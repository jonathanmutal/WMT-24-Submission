#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a Transformers model on text translation.
"""
import argparse
import json
import logging
import math
import os
import random

import datasets
import evaluate
import numpy as np
import torch
import shutil

from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from sacrebleu.metrics import BLEU, CHRF, TER

from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    NllbTokenizer,
    NllbTokenizerFast,
    MarianTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from dataclasses import dataclass
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.42.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--train_source_file", type=str, default=None, help="A plain file containing the source sentences."
    )
    parser.add_argument(
        "--train_target_file", type=str, default=None, help="A plain file containing the target data."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=80,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=80,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help=(
            "Whether to pad all samples to model maximum sentence "
            "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--validation_source_file", type=str, default=None, help="A validation file containing the validation data."
    )
    parser.add_argument(
        "--validation_target_file", type=str, default=None, help="A validation file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch. it saves the best models",
    )
    parser.add_argument(
        "--resume_from_checkpoint_path",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint path.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="If the training should continue from the checkpoint.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to enable early stopping.",
    )
    parser.add_argument(
        "--early_stopping_steps",
        type=int,
        default=None,
        help=(
            "The early stopping steps "
        ),
     )
    parser.add_argument(
        "--save_last_model_steps",
        type=int,
        default=10000,
        help=(
            "Each number of steps to save the model"
        ),
     )
    parser.add_argument(
        "--save_total_checkpoints",
        type=int,
        default=1,
        help=(
            "Number of checkpoints to save the model. If early stopping, it will always save the best model in best_model folder."
        ),
     )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help=(
            "training in a multilingal fashion. In this case we need to use the attribute multilingual_files"
        ),
     )
    parser.add_argument(
        "--multilingual_files",
        type=str,
        default=None,
        help=(
            "training in a multilingal fashion in the target. \
            we need to specify the token in the KEY and path in the Value. For \
            instance, \
            {oci_Latn: [{\"source\" : ./data/occitan/source.es, \"target\": ./data/occitan/target.oc}]}"
        ),
     )
    args = parser.parse_args()

    # Sanity checks
    if args.early_stopping and args.early_stopping_steps is None:
        raise ValueError("Need for early_stopping_steps.")

    if not args.early_stopping:
        args.early_stopping = True
        args.early_stopping_steps = 100000

    if args.train_source_file is None and \
       args.validation_source_file is None:
        raise ValueError("Need either a task name or a training/validation source file.")

    if args.train_target_file is None and \
       args.validation_target_file is None:
        raise ValueError("Need either a task name or a training/validation target file.")

    ## if val_max_target_length is not set
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    # sanity check for multilingual
    if args.multilingual and args.multilingual_files is None:
        raise ValueError("Please, provide the data inside the multilingual_files parameter \
                         for the task in json format.")

    if args.multilingual and args.multilingual_files:
        print(args.multilingual_files)
        args.multilingual_files = json.loads(args.multilingual_files)
        ## validating if it has the src and tgt keys.
        for key in args.multilingual_files.keys():
            for index, pair_files in enumerate(args.multilingual_files[key]):
                if not "source" in pair_files:
                    raise ValueError(f"Please, provide the source file (in source key) for the \
                                      file {index} in the {key} language")
                if not "target" in pair_files:
                    raise ValueError(f"Please, provide the target file (in target key) for the \
                                       file {index} in the {key} language")
    return args


@dataclass
class Trainer():
    train_source_file: str
    train_target_file : str
    validation_source_file: str
    validation_target_file: str

    output_dir: str

    model_name_or_path: str = None
    config_name: str = None
    tokenizer_name: str = None
    use_slow_tokenizer: bool = False
    model_type: str = None

    source_lang: str = None
    target_lang: str = None
    source_prefix: str = None
    max_source_length: int = 64
    max_target_length: int = 64
    max_length: int = 64
    val_max_target_length: int = None
    num_beams: int = 1
    pad_to_max_length: bool = False
    ignore_pad_token_for_loss: bool = True

    preprocessing_num_workers: int = None
    overwrite_cache: bool = True

    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 3e-5
    lr_scheduler_type: SchedulerType = "linear"
    num_warmup_steps: int = 0

    weight_decay: float = 0.0
    num_train_epochs: int = 3
    max_train_steps: int = None
    gradient_accumulation_steps: int = 1
    early_stopping: bool = False
    early_stopping_steps: int = 5
    checkpointing_steps: int = 5000
    save_last_model_steps : int = 10000
    save_total_checkpoints : int = 1

    seed: int = 111

    trust_remote_code: bool = False
    resume_from_checkpoint: bool = False
    resume_from_checkpoint_path: str = None
    with_tracking: bool = True
    report_to: str = "TensorBoard"

    multilingual: bool = False
    multilingual_files : dict[list] = None

    def __post_init__(self):
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
        # in the environment
        self.accelerator =(
           Accelerator(log_with=self.report_to, project_dir=self.output_dir) if self.with_tracking else Accelerator()
        )
        # set the logging for the training
        self.set_logging()
        # set the metrics
        self.set_metrics()
        # to set the seed and to be reproducible
        if self.set_seed is not None:
            self.set_seed()
        # if resuming from checkpoint, we get the last trainable checkpoint
        if self.resume_from_checkpoint:
            self.get_last_checkpoint()
        # to intialize the tokenizer
        self.set_tokenizer()
        # in the case we want to train from scratch
        self.set_model_configuration()
        # downloading and loading the model
        self.set_model()
        # load the data
        self.load_dataset()
        # preprocessing the dataset
        self.preprocess_data()
        # set data loader
        self.set_dataloader()
        # calculate number of training steps
        self.calculate_number_of_training_steps()
        # set the optimizer
        self.set_optimizer()
        # set the learning rate scheduler
        self.set_lr_scheduler()
        # set device
        self.set_device()
        # number of steps to validation. We only save the best models according to the metric
        if self.checkpointing_steps is not None and self.checkpointing_steps.isdigit():
            self.checkpointing_steps = int(self.checkpointing_steps)
        # enable tracking
        if self.with_tracking:
            self.set_tracker()
        # if early_stopping
        if self.early_stopping:
            self.set_early_stopping()

    def set_logging(self):
        self.logger = logger
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            filename=f"{self.output_dir}/train.log",
            filemode='w',
            encoding="utf-8-sig",
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

    def set_metrics(self):
        # TODO: do something more general
         self.bleu = BLEU()

    def set_seed(self):
        # If passed along, set the training seed now.
        set_seed(self.seed)

    def set_tokenizer(self):
        if self.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, use_fast=not self.use_slow_tokenizer, trust_remote_code=self.trust_remote_code
            )
        elif self.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, use_fast=not self.use_slow_tokenizer, trust_remote_code=self.trust_remote_code
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        # For translation we set the codes of our source and target languages (only useful for mBART and nllb, the others will
        # ignore those attributes).
        if isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast, NllbTokenizer, NllbTokenizerFast)):
            if self.source_lang is not None:
                self.tokenizer.src_lang = self.source_lang
            if self.target_lang is not None:
                self.tokenizer.tgt_lang = self.target_lang

        # if it is multilingual, we check if the special token is in the vocabulary
        # otherwise, we add it
        if self.multilingual:
            special_tokens_to_add = []
            for key in self.multilingual_files.keys():
                special_tokens_to_add.append(key)

            ## by default we will add the tokens and initizitilie randomly the embedding layers
            if special_tokens_to_add:
                self.tokenizer.add_tokens(new_tokens=special_tokens_to_add, special_tokens=True)

                if self.accelerator.is_main_process:
                    self.logger.warning(f"we add these tokens to the tokenizer: {special_tokens_to_add}")


    def set_model_configuration(self):
        """ Load pretrained model configuration"""
        if self.config_name:
            self.config = AutoConfig.from_pretrained(self.config_name, trust_remote_code=self.trust_remote_code)
        elif self.model_name_or_path:
            self.config = AutoConfig.from_pretrained(self.model_name_or_path, trust_remote_code=self.trust_remote_code)
        else:
            self.config = CONFIG_MAPPING[self.model_type]()
            self.logger.warning("You are instantiating a new config instance from scratch.")

    def set_model(self):
        """
        In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
        """
        if self.model_name_or_path:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_name_or_path),
                config=self.config,
                trust_remote_code=self.trust_remote_code,
            )
        else:
            self.logger.info("Training new model from scratch")
            self.model = AutoModelForSeq2SeqLM.from_config(self.config, trust_remote_code=self.trust_remote_code)

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            if self.accelerator.is_main_process:
                self.logger.warning(f"we randomly initiate {len(self.tokenizer) - embedding_size} tokens for training.")

        # Set decoder_start_token_id
        if self.model.config.decoder_start_token_id is None and \
           isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast, NllbTokenizer, NllbTokenizerFast)):
            assert (
                self.target_lang is not None and self.source_lang is not None
            ), "mBart and Nllb requires --target_lang and --source_lang"
            if isinstance(self.tokenizer, MBartTokenizer, NllbTokenizer):
                self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[self.target_lang]
            else:
                self.model.config.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)

        self.logger.info("****** Configuration model ******* ")
        self.logger.info(self.model.config)
        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    def load_dataset(self):
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        def load_parallel_dataset(source_file_name: str, target_file_name: str):
            def load_corpus(filepath: str):
                corpus = []
                with open(filepath, mode="r") as c:
                    for line in c:
                        corpus.append(line.strip())
                return corpus
            source_sents = load_corpus(source_file_name)
            target_sents = load_corpus(target_file_name)

            assert len(source_sents) == len(target_sents),\
                    f"{source_file_name} and {target_file_name} do no have the same number of sentences"

            parallel_dataset = Dataset.from_dict(
                {'source': source_sents,
                 'target': target_sents}
            )
            return parallel_dataset

        self.raw_dataset_train = load_parallel_dataset(self.train_source_file, self.train_target_file)
        self.raw_dataset_valid = load_parallel_dataset(self.validation_source_file, self.validation_target_file)

        if self.multilingual:
            self.raw_dataset_multilingual = {}
            for key in self.multilingual_files.keys():
                parallel_datasets = []
                for parallel_data in self.multilingual_files[key]:
                    parallel_datasets.append(
                        load_parallel_dataset(parallel_data["source"], parallel_data["target"])
                    )
                self.raw_dataset_multilingual[key] = concatenate_datasets(parallel_datasets)

    def preprocess_data(self):
        prefix = self.source_prefix if self.source_prefix is not None else ""

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = self.raw_dataset_train.column_names
        padding = "max_length" if self.pad_to_max_length else False

        def preprocess_dataset(dataset: Dataset, source_lang: str = None, target_lang: str = None):
            def preprocess_function(examples: dict):
                self.tokenizer.src_lang = source_lang
                self.tokenizer.tgt_lang = target_lang
                inputs = examples["source"]
                targets = examples["target"]
                ## for marianNMT, we need to add the token for the tokenizer
                if target_lang and isinstance(self.tokenizer, MarianTokenizer):
                    targets = [target_lang + " " + target for target in targets]
                inputs = [prefix + inp for inp in inputs]
                model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=padding, truncation=True)

                # Tokenize targets with the `text_target` keyword argument
                labels = self.tokenizer(text_target=targets, max_length=self.max_target_length, padding=padding, truncation=True)

                # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
                # padding in the loss.
                if padding == "max_length" and self.ignore_pad_token_for_loss:
                    labels["input_ids"] = [
                        [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                    ]

                model_inputs["labels"] = labels["input_ids"]

                return model_inputs

            return dataset.map(
                preprocess_function,
                batched=True,
                batch_size=64,
                num_proc=self.preprocessing_num_workers,
                remove_columns=self.raw_dataset_train.column_names,
                load_from_cache_file=not self.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        with self.accelerator.main_process_first():
            self.train_dataset = preprocess_dataset(self.raw_dataset_train, source_lang=self.source_lang, target_lang=self.target_lang)
            self.eval_dataset = preprocess_dataset(self.raw_dataset_valid, source_lang=self.source_lang, target_lang=self.target_lang)
            ## if it is multilingual, we preprocess each language with each special token
            if self.multilingual:
                multilingual_datasets = []
                for key in self.raw_dataset_multilingual.keys():
                    multilingual_datasets.append(
                        preprocess_dataset(self.raw_dataset_multilingual[key],
                                           source_lang=self.source_lang,
                                           target_lang=key)
                    )
                ## joining all the datasets together
                self.train_dataset = concatenate_datasets([self.train_dataset] + multilingual_datasets)

        # Log a few random samples from the training set:
        for index in random.sample(range(len(self.train_dataset)), 3):
            self.logger.info(f"Sample {index} of the training set: {self.train_dataset[index]}.")
            self.logger.info(f"Input: {self.tokenizer.decode(self.train_dataset[index]['input_ids'], skip_special_tokens=False)}")
            self.logger.info(f"Output: {self.tokenizer.decode(self.train_dataset[index]['labels'], skip_special_tokens=False)}")

        if self.multilingual:
            self.logger.info(f"Sample {index} of the multilingual data set: {multilingual_datasets[0][0]}.")
            self.logger.info(f"Input: {self.tokenizer.decode(multilingual_datasets[0][0]['input_ids'], skip_special_tokens=False)}")
            self.logger.info(f"Output: {self.tokenizer.decode(multilingual_datasets[0][0]['labels'], skip_special_tokens=False)}")

    def set_dataloader(self):
        # DataLoaders creation:
        label_pad_token_id = -100 if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if self.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if self.accelerator.use_fp16 else None,
            )

        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, collate_fn=data_collator, batch_size=self.per_device_train_batch_size
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size
        )

    def calculate_number_of_training_steps(self):
        # math around the number of training steps.
        self.overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * self.num_update_steps_per_epoch
            self.overrode_max_train_steps = True

    def set_optimizer(self):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

    def set_lr_scheduler(self):
        self.lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )

    def set_device(self):
        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler
        )
        if self.resume_from_checkpoint:
            ### TODO: remove this part after saving the model using accelerator.save_model.
            if torch.cuda.is_available():
                device =  torch.device("cuda")
            self.accelerator.print(f"Resumed from checkpoint: {self.checkpoint_path}")
            #self.accelerator.load_state(self.checkpoint_path)
            ### loading optimizer
            optimizer_state = torch.load(f"{self.checkpoint_path}/optimizer.bin", map_location=device)
            self.optimizer.load_state_dict(optimizer_state)
            logger.info("All optimizer states loaded successfully")
            ### loading scheduler
            scheduler_state = torch.load(f"{self.checkpoint_path}/scheduler.bin", map_location=device)
            self.lr_scheduler.load_state_dict(scheduler_state)
            logger.info("The scheduler was loaded successfully")
            ### loading random state
            try:
               states = torch.load(input_dir.joinpath(f"{self.checkpoint_path}/random_states_0.pkl"))
               random.setstate(states["random_state"])
               np.random.set_state(states["numpy_random_seed"])
               torch.set_rng_state(states["torch_manual_seed"])
               torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
               logger.info("All random states loaded successfully")
            except Exception:
                logger.info("Could not load random states")

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        # During the preparation of the dataloader, especially when using libraries like accelerate, the size of the dataloader can change.
        # This can happen due to data shuffling, filtering, or other preprocessing steps that might alter the number of batches in each epoch.
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.max_train_steps = self.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self.max_train_steps / self.num_update_steps_per_epoch)

    def set_tracker(self):
        # We need to initialize the trackers we use, and also store our configuration.
        # We initialize the trackers only on main process because `accelerator.log`
        # only logs on main process and we don't want empty logs/runs on other processes.
        experiment = {}
        if self.accelerator.is_main_process:
            experiment_config = vars(self)
            # TensorBoard cannot log Enums, need the raw value
            experiment["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            experiment["lr"] = experiment_config["learning_rate"]
            experiment["warmup_steps"] = experiment_config["num_warmup_steps"]
            experiment["model_name"] = experiment_config["model_name_or_path"]
            if self.source_lang and self.target_lang:
                experiment["languages"] = experiment_config["source_lang"] + "-" + experiment_config["target_lang"]
            experiment["max_length"] = str(experiment_config["max_source_length"]) + "-" + str(experiment_config["max_target_length"])
            experiment["num_beams"] = experiment_config["num_beams"]
            experiment["batch_size"] = experiment_config["per_device_train_batch_size"]
            experiment["epochs"] = experiment_config["num_train_epochs"]
            experiment["weight_decay"] = experiment_config["weight_decay"]
            experiment["early_stopping_steps"] = experiment_config["early_stopping_steps"]
            experiment["valid_steps"] = experiment_config["checkpointing_steps"]

            self.accelerator.init_trackers("logs_tracker", experiment)

    def set_early_stopping(self):
        # load bleu
        # TODO: do something more general
        self.best_bleu = 0.0
        self.bleu_early_stopping = 0

        ### setting the earlystopping step from last checkpoint
        if self.resume_from_checkpoint:
            self.logger.info(f"Resuming early stopping steps from {self.checkpoint_path}/earlystopping")
            with open(f"{self.checkpoint_path}/earlystopping") as f:
                best_bleu, bleu_early_stopping  = f.readlines()[0].strip().split(' ')
            self.best_bleu, self.bleu_early_stopping = float(best_bleu), int(bleu_early_stopping)

    ### auxiliar methods for training to save, write predictions and validate
    def save_model(self, name=""):
        self.logger.info(f" Saving model to {self.output_dir}/{name}")
        self.accelerator.wait_for_everyone()
        # TODO : remove next iterations
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            f"{self.output_dir}/{name}",
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save
        )
#        self.accelerator.save_model(self.model, f"{self.output_dir}/{name}")
        self.accelerator.save_state(f"{self.output_dir}/{name}/state")
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(f"{self.output_dir}/{name}")

        if self.save_total_checkpoints and name:
            main_name, step = name.split("_")
            # get the directiories that begin with step
            dirs = [f for f in os.scandir(self.output_dir) if f.is_dir() and f.name.startswith(main_name)]
            # sort the directories by time of saves
            dirs.sort(key=os.path.getctime)
            # rm the directories
            for dir_to_rm in dirs[:-self.save_total_checkpoints]:
                shutil.rmtree(dir_to_rm)

        with open(f"{self.output_dir}/{name}/earlystopping", "w") as f:
            f.write(f"{self.best_bleu} {self.bleu_early_stopping}")

    def write_prediction(self, predictions):
        with open(f"{self.output_dir}/prediction", mode="w") as o:
            try:
                o.write("\n".join(predictions))
            except:
                self.logger.info("Error writing output file")

    def validate(self):
        self.model.eval()
        gen_kwargs = {
            "max_length": self.val_max_target_length if self.val_max_target_length is not None else self.config.max_length,
            "num_beams": self.num_beams,
        }
        if isinstance(self.tokenizer, (NllbTokenizerFast)):
            gen_kwargs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        samples_seen = 0
        predictions = []
        references = []
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                generated_tokens = self.accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not self.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = self.accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id)

                generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()
                labels = self.accelerator.gather(labels).cpu().numpy()

                if self.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                # If we are in a multiprocess environment, the last batch has duplicates
                if self.accelerator.num_processes > 1:
                    if step == len(self.eval_dataloader) - 1:
                        decoded_preds = decoded_preds[: len(self.eval_dataloader.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(self.eval_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)
                references.extend(decoded_labels)
                predictions.extend(decoded_preds)
        score = {
            'BLEU': self.bleu.corpus_score(hypotheses=predictions, references=[references]).score
        }
        return score, predictions

    def get_last_checkpoint(self):
        # Potentially load in the weights and states from a previous save
        if self.resume_from_checkpoint_path is not None and \
            ("epoch_" in self.resume_from_checkpoint_path or \
            "step_" in self.resume_from_checkpoint_path or \
            "bestmodel_" in self.resume_from_checkpoint_path):
            self.checkpoint_path = self.resume_from_checkpoint_path
        else:
            # Get the most recent checkpoint
            # Only get the directories with step
            dirs = [
                f for f in os.scandir(self.resume_from_checkpoint_path)
                if f.is_dir() and (f.name.startswith("step") or f.name.startswith('bestmodel'))
            ]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1].path  # Sorts folders by date modified, most recent checkpoint is the last
            self.checkpoint_path = path


    def train(self):
        # Train!
        total_batch_size = self.per_device_train_batch_size * self.accelerator.num_processes * self.gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.max_train_steps}")
        self.logger.info(f"  Max source length = {self.max_source_length}")
        self.logger.info(f"  Max target length = {self.max_target_length}")
        self.logger.info(f"  Max generation target length = {self.val_max_target_length}")
        self.logger.info(f"  Multilingual approach = {self.multilingual}")

        progress_bar = tqdm(range(self.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Only show the progress bar once on each machine.
        if self.resume_from_checkpoint:
            #self.model.load_state_dict(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            path = os.path.basename(self.checkpoint_path)
            training_difference = os.path.splitext(path)[0]
            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * self.num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                file_name = "step_" if "step_" in training_difference else "bestmodel_"
                resume_step = int(training_difference.replace(file_name, "")) * self.gradient_accumulation_steps
                starting_epoch = resume_step // len(self.train_dataloader)
                completed_steps = resume_step // self.gradient_accumulation_steps
                resume_step -= starting_epoch * len(self.train_dataloader)

            if self.accelerator.is_main_process:
                self.logger.info(f"***** Resume configuration *****")
                self.logger.info(f"  Resumed from checkpoint = {self.checkpoint_path}")
                self.logger.info(f"  Starting Epoch = {starting_epoch}")
                self.logger.info(f"  Completed Steps = {completed_steps}")
                self.logger.info(f"  Early Stopping = {self.best_bleu} {self.bleu_early_stopping}")

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)
        score, prediction = self.validate()
        self.logger.info(f"Validation Step 0 -- BLEU: {score['BLEU']:.2f}")

        for epoch in range(starting_epoch, self.num_train_epochs):
            self.model.train()
            if self.with_tracking:
                total_loss = 0
            if self.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step)
            else:
                active_dataloader = self.train_dataloader
            for step, batch in enumerate(active_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if self.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / self.gradient_accumulation_steps
                self.accelerator.backward(loss)
                if step % self.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                if self.accelerator.is_main_process:
                   if self.with_tracking:
                       self.accelerator.log(
                           {
                               "step": step,
                               "loss": loss.detach().float().item(),
                               "cost": total_loss.item() / len(self.train_dataloader),
                               "lr": self.lr_scheduler.get_last_lr()[0],
                               "input_shape": batch["input_ids"].shape
                           },
                           step=completed_steps,
                       )
                       self.logger.info(f"Epoch : {int(epoch)+1} -- Step {completed_steps} -- Cost : {total_loss.item() / (step+1):.4f} -- lr : {self.lr_scheduler.get_last_lr()[0]:.14f} -- Loss Batch : {loss:.4f} -- Input shape : {batch['input_ids'].shape}")

                ## printing the progress and doing validation
                if isinstance(self.checkpointing_steps, int):
                    if completed_steps % self.checkpointing_steps == 0:
                        score, prediction = self.validate()
                        if score["BLEU"] > self.best_bleu:
                            self.best_bleu = score["BLEU"]
                            if self.accelerator.is_main_process:
                                self.save_model(f"bestmodel_{completed_steps}")
                            self.bleu_early_stopping = 0
                            self.write_prediction(prediction)
                        self.bleu_early_stopping += 1
                        if self.accelerator.is_main_process:
                            self.logger.info(f"Validation Step {completed_steps} -- BLEU: {score['BLEU']:.2f} (stalled {self.bleu_early_stopping} : {self.best_bleu:.2f}) -- Loss: {total_loss.item() / len(self.train_dataloader):.6f}")
                            if self.with_tracking:
                                self.accelerator.log(
                                    {
                                        "Cost": total_loss.item() / len(self.train_dataloader),
                                        "BLEU": score['BLEU'],
                                        "best_bleu": self.best_bleu,
                                        "stalled": self.bleu_early_stopping,
                                    },
                                    step=completed_steps,
                                )
                        ## begin to the state of train again
                        self.model.train()

                # if we save the model each X steps for the checkpoint
                if self.save_total_checkpoints and completed_steps % self.save_last_model_steps == 0:
                   if self.accelerator.is_main_process:
                       self.save_model(f"step_{completed_steps}")

                # if we finish the epochs or we reach the patient limit
                # of improvement
                if completed_steps >= self.max_train_steps or (self.early_stopping and self.bleu_early_stopping == self.early_stopping_steps):
                    break

                if self.checkpointing_steps == "epoch":
                    if self.accelerator.is_main_process:
                        self.save_model(f"epoch_{epoch}")

        if self.with_tracking:
            self.accelerator.end_training()
        if self.output_dir is not None:
            if self.accelerator.is_main_process:
                self.save_model(f"step_{completed_steps}")

        with open(os.path.join(self.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_bleu": self.best_bleu}, f)


def main():
    # Parse the arguments
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_translation_no_trainer", args)

    # Handle the repository creation
    #accelerator.wait_for_everyone()
    print(vars(args))
    Trainer(**vars(args)).train()

if __name__ == "__main__":
    main()
