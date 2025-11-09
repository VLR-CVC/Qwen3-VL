# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.utils import select_model_class
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

import time

from enum import Enum

from torch.distributed import init_process_group
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed._composable import replicate
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import Color


def save_model(trainer, output_dir):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    del state_dict
    step = trainer.step

    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step:06d}")
    torch.save(cpu_state_dict, os.path.join(checkpoint_dir, "pytorch_model.bin"))


def compile_model(model: torch.nn.Module):
    model.visual = torch.compile(
        model.visual, fullgraph=True
    )
    model.language_model = torch.compile(
        model.language_model, fullgraph=True
    )
    model.visual.merger = torch.compile(
        model.visual.merger, fullgraph=True
    )

    model = torch.compile(
        model, fullgraph=True
    ) 


def apply_ddp(
        model,
        dp_mesh,
        enable_compile=True,
        enable_compile_autograd=True,
):
    if enable_compile:
        if enable_compile_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, dp_mesh, bucket_cap_mb=100)

    logger.info("applied DDP to the model")


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def loss_fn(pred, labels):
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


class Trainer(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, *args, **kwargs):
        attn_implementation = "flash_attention_2"
        logger.info("starting finetune job")

        torch.distributed.init_process_group(backend="nccl")

        local_rank = int(os.environ["LOCAL_RANK"])

        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        os.makedirs(training_args.output_dir, exist_ok=True)

        self.model, data_args = select_model_class(model_args, data_args, training_args, attn_implementation)

        self.device = torch.device(f"cuda:{local_rank}")

        compile_model(self.model.to(self.device).to(torch.bfloat16))
        logger.info("model compiled with torch.compile")

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        )

        set_model(model_args, self.model)

        self.data_module = make_supervised_data_module(self.processor, data_args=data_args)

        dataset = self.data_module['train_dataset']
        collator = self.data_module['data_collator']

        self.sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
        )

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            collate_fn=collator,
            num_workers=64,
            pin_memory=True,
            persistent_workers=True,
            sampler=self.sampler,
        )

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )

        self.step = 0
        self.tokens_seen = 0
        self.tokens_seen_assistant = 0
        self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()
        self.color = Color()

    def batch_generator(self, data_module):
        data_iter = iter(self.data_loader)

        while True:
            data_start_time = time.perf_counter()
            try:
                batch = next(data_iter)

            except StopIteration:
                raise Exception("DataLoader ran out of data.")

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)

            labels = batch.pop("labels")
            _ = batch.pop("attention_mask", None)

            ntokens_batch = labels.numel()
            ntokens_batch_assistant = (labels != -100).sum().item()

            self.tokens_seen_assistant += ntokens_batch_assistant
            self.tokens_seen += ntokens_batch
            self.ntokens_since_last_log += ntokens_batch

            self.data_time_delta = time.perf_counter() - data_start_time

            yield batch, labels

    def log(self, global_loss):

        time_delta = time.perf_counter() - self.time_last_log

        tps = self.ntokens_since_last_log / time_delta

        color = self.color

        data_time_pct = (self.data_time_delta / time_delta) * 100

        logger.info(
            f"{color.red}step {self.step} "
            f"{color.green}loss {global_loss:.4f} "
            f"{color.blue}tps {tps:.2f} "
            f"{color.reset}"
            f"time_delta {self.train_step_delta:.2f}s "
            f"data_time_pct {data_time_pct:.2f}%"
        )

        self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()

    def train_step(self, data_iterator):
        self.optimizer.zero_grad()
        batch, labels = next(data_iterator)
        # we use the labels directly

        for _ in range(self.training_args.gradient_accumulation_steps):
            with torch.autocast("cuda", torch.bfloat16):
                outputs = self.model(
                    labels=labels,
                    **batch
                )
                loss = outputs.loss
            total_loss = loss / self.training_args.gradient_accumulation_steps
            total_loss.backward()

        if self.training_args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training_args.max_grad_norm
            )
        self.optimizer.step()
        self.train_step_delta = time.perf_counter() - self.time_last_log

        self.log(total_loss.item())


    def train(self):
        data_iterator = self.batch_generator(self.data_module)

        try:
            while True:
                self.step += 1
                self.train_step(data_iterator)
        except Exception as e:
            logger.info(f"exception: {e}")
        except KeyboardInterrupt:
            logger.info("keyboard interrupt received...")

        logger.info(f"tokens seen: {self.tokens_seen}")
        logger.info(f"assistant tokens seen: {self.tokens_seen_assistant}")
        logger.info("Training completed")

if __name__ == "__main__":
    init_logger()

    trainer = Trainer()
    trainer.train()
