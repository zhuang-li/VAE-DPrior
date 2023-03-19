# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import logging
import math
import os, transformers, torch
import pickle
import sys

from transformers.data.data_collator import DataCollatorForCLModeling, DataCollatorForTopicClusterLanguageModeling, DataCollatorForProtoClusterLanguageModeling

from PrefixTuning.transformers.examples.control.tokenization import WordTokenizer

sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(sys_path)
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sys_path)
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(sys_path)
print (sys.path)
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from transformers.data.datasets.language_modeling import LineByLineUnderstandTextDataset, \
    LineByLineUnderstandFrontTextDataset, LineByLineCLDataset, LineByLineTopicDataset, LineByLineNoRelDataset, \
    LineByLineProtoDataset, LineByLinePrivacyDataset

from .train_control import PrefixTuning, ClassificationHead, PrefixEmbTuning
from transformers.file_utils import cached_path

import glob

path = os.path.abspath(transformers.__file__)

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWeightedLanguageModeling,  # modified
    DataCollatorForEmbMatchLanguageModeling,  # modified
    DataCollatorForTopicLanguageModeling,  # modified
    DataCollatorForLengthLanguageModeling,  # modified
    DataCollatorForKeywordLanguageModeling,  # modified
    DataCollatorForData2TextLanguageModeling,  # modified
    DataCollatorForText2DataLanguageModeling,  # modified
    DataCollatorForWritingPromptsLanguageModeling,  # modified
    DataCollatorForClassificationSentimentLanguageModeling,  # modified
    DataCollatorForSumLanguageModeling,  # modified
    HfArgumentParser,
    LineByLineTextDataset,
    LineByLineWithWeightTextDataset,  # modified
    LineByLineEmbMatchTextDataset,  # modified
    LineByLineTopicTextDataset,  # modified
    LineByLineKeywordTextDataset,  # modified
    LineByLineLengthTextDataset,  # modified
    LineByLineData2TextTextDataset,  # modified
    LineByLineLemma2TextTextDataset,  # modified
    LineByLineText2DataTextDataset,  # modified
    LineByLineTriplesTextDataset,  # modified
    LineByLineWebNLGTextDataset,  # modified
    LineByLineWritingPromptsTextDataset,  # modified
    LineByLineSentimentTextDataset,  # modified
    LineByLineClassificationSentimentTextDataset,  # modified
    LineByLineClassificationTopicTextDataset,
    LineByLineSumTextDataset,  # modified
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    Trainer_Prefix,
    TrainingArguments,
    set_seed,
    GPT2LMHeadModel,
    BertTokenizerFast,
    BertModel,
    AutoModelForSequenceClassification,
    GPT2LMHeadModelAdapter, BertTokenizer,
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    prefixModel_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The prefix model checkpoint for weights initialization. "
                    "Leave None if you want to train a model from scratch."
        },
    )

    prefix_mode: Optional[str] = field(
        default='activation',
        metadata={
            "help": "activation or embedding"
        },
    )



    preseqlen: Optional[int] = field(
        default=0,
        metadata={
            "help": "preseqlen for how many tokens of prefix should we include."
        },
    )

    optim_prefix: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether we are optimizing the prefix directly, or optimize another amortized function that "
                    "genrate the prefix."
        },
    )



    tuning_mode: Optional[str] = field(
        default='finetune',
        metadata={
            "help": "whether it's doing prefixtune or finetune."
        },
    )

    objective_mode: Optional[int] = field(
        default=0,
        metadata={
            "help": "In prefixtuning setting, the objective function... "
        },
    )

    top_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": "In finetuning setting, if we only tune the top k layers. "
        },
    )

    adapter_design: Optional[int] = field(
        default=2,
        metadata={
            "help": "For Baseline of the adapter module... (1) means using the NLG adapter reference. "
                    "(2) means using a design similar to adapter module"
        },
    )

    adapter_bottleneck: Optional[int] = field(
        default=100,
        metadata={
            "help": "For baseline adapter module: the mid dim of the adapter. "
        },
    )

    parametrize_emb: Optional[str] = field(
        default='MLP',
        metadata={
            "help": "MLP or Emb to parametrize when we optimize for the embeddings."
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "dropout rate for the prefix tuning model. "
        },
    )

    init_random: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to init a random embedding, or use GPT2 embedding for the prefix tuning model. "
        },
    )

    use_dropout: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to use dropout of GPT2 on trainer. "
        },
    )

    mid_dim: Optional[int] = field(
        default=512,
        metadata={
            "help": "the mid dim."
        },
    )

    dataless_sample_size: Optional[int] = field(
        default=8,
        metadata={
            "help": "the size of samples for each class in dataless training."
        },
    )

    gumbel: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the gumbel softmax trick in training."
        },
    )

    replay_buffer: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the replay buffer in training."
        },
    )

    training_obj: Optional[int] = field(
        default=0,
        metadata={
            "help": "use a specified training objective"
        },
    )


    dataless_sample_length: Optional[int] = field(
        default=20,
        metadata={
            "help": "the length of samples for each class in dataless training."
        },
    )

    dataless_control_type: Optional[int] = field(
        default=0,
        metadata={
            "help": "the type of control in dataless training."
        },
    )

    dataless_usebaseline: Optional[str] = field(
        default='yes',
        metadata={
            "help": "use baseline in dataless training."
        },
    )


    dataless_discri_model_path: Optional[str] = field(
        default='textattack/roberta-base-imdb',
        metadata={
            "help": "the path to discri_model and discri_tokenizer"
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    task_mode: Optional[str] = field(
        default=None, metadata={"help": "The task mode"}
    )

    format_mode: Optional[str] = field(
        default='cat', metadata={"help": "The mode of data2text format (cat, peek, nopeek)"}
    )

    lowdata_token: Optional[str] = field(
        default='summarize', metadata={"help": "The token to be prepended at initialization time. "}
    )

    use_lowdata_token: Optional[str] = field(
        default='yes', metadata={"help": "Whether we should use the lowdata token and pass it to the prefixTuning Model "
                                         "for the initialization trick.  "}
    )

    dataless: Optional[str] = field(
        default='no', metadata={"help": "Whether we are training or loading dataless model."}
    )

    train_embs: Optional[str] = field(
        default='no', metadata={"help": "whether the train word embeddings"}
    )

    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )

    train_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for training data. "}
    )

    val_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for dev data. "}
    )

    # controlprefix: Optional[str] = field(
    #     default="yes", metadata={"help": "The control mode"}
    # )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    sentence_tokenizer: WordTokenizer,
    label_id_List,
    label_list,
    sentence_list,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    iterations: int = 0,
    classes_per_it:int = 0,
    sample_per_class:int = 0,
    max_seq_length:int = 0,
    data_type:str = None
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        if args.task_mode == 'cl':
            dataset = LineByLineCLDataset(tokenizer=tokenizer,sentence_tokenizer = sentence_tokenizer, label_id_List= label_id_List,
                                          label_list=label_list, sentence_list=sentence_list,
                                                   block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                   eos_tok=tokenizer.eos_token, max_source_length = max_seq_length,
                                                   max_target_length = max_seq_length)
        elif args.task_mode == 'no_hidden' and data_type=='privacy':
            dataset = LineByLinePrivacyDataset(tokenizer=tokenizer,sentence_tokenizer = sentence_tokenizer, label_id_List= label_id_List,
                                          label_list=label_list, sentence_list=sentence_list,
                                                   block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                   eos_tok=tokenizer.eos_token, max_source_length = max_seq_length,
                                                   max_target_length = max_seq_length)
        elif data_type=='privacy':
            dataset = LineByLinePrivacyDataset(tokenizer=tokenizer,sentence_tokenizer = sentence_tokenizer, label_id_List= label_id_List,
                                          label_list=label_list, sentence_list=sentence_list,
                                                   block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                   eos_tok=tokenizer.eos_token, max_source_length = max_seq_length,
                                                   max_target_length = max_seq_length)
        elif args.task_mode == 'topic' or args.task_mode == 'no_hidden' or args.task_mode == 'casual_lens':
            dataset = LineByLineTopicDataset(tokenizer=tokenizer,sentence_tokenizer = sentence_tokenizer, label_id_List= label_id_List,
                                          label_list=label_list, sentence_list=sentence_list,
                                                   block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                   eos_tok=tokenizer.eos_token, max_source_length = max_seq_length,
                                                   max_target_length = max_seq_length)
        elif args.task_mode == 'no_rel':
            dataset = LineByLineNoRelDataset(tokenizer=tokenizer,sentence_tokenizer = sentence_tokenizer, label_id_List= label_id_List,
                                          label_list=label_list, sentence_list=sentence_list,
                                                   block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                   eos_tok=tokenizer.eos_token, max_source_length = max_seq_length,
                                                   max_target_length = max_seq_length)
        elif args.task_mode == 'proto' or args.task_mode == 'proto_reg':
            dataset = LineByLineProtoDataset(tokenizer=tokenizer,sentence_tokenizer = sentence_tokenizer, label_id_List= label_id_List,
                                          label_list=label_list, sentence_list=sentence_list,
                                                   block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                   eos_tok=tokenizer.eos_token, max_source_length = max_seq_length,
                                                   max_target_length = max_seq_length, iterations = iterations, classes_per_it = classes_per_it, sample_per_class = sample_per_class)

        return dataset

    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
            cache_dir=cache_dir,
        )


def initilize_gpt2(model_args, data_args):
    #model_args.model_name_or_path = 'gpt2'
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(sys_path)
    print(sys.path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(sys_path, '.cache/huggingface')

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    config._my_arg_tune_mode = model_args.tuning_mode

    # 0 means the regular token level objective, which is sum / output_len
    # 1 means the sentence level objective, which is sum
    # 2 means our buggy version which is sum/max_batch(input_len +output_len)
    # 3 means our buggy version which is sum/max_batch(output_len)
    # 4 means our buggy version which is sum/(input_len +output_len)
    config._objective_mode = model_args.objective_mode
    config._my_arg_task_mode = data_args.task_mode

    if model_args.tuning_mode in ['finetune', 'adaptertune', 'finetune-top']:
        print('objective is 0 because of finetune')
    elif model_args.tuning_mode == 'prefixtune':
        print('objective is {}'.format(config._objective_mode))


    if model_args.model_name_or_path:
        print("Reading from pre-trained model====================================")
        print(config.return_dict)
        config.return_dict = True
        model = GPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=model_args.cache_dir
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)



    print('adapting the size of the model embedding to include [PAD]')
    print('len(tokenizer) = ', len(tokenizer))
    print(tokenizer.pad_token, tokenizer.pad_token_id)
    if tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens(
            {'pad_token': '[PAD]'})
    if tokenizer.eos_token is None:

        tokenizer.eos_token = tokenizer.sep_token
        #tokenizer.eos_token_id = tokenizer.sep_token_id
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.cls_token
        #tokenizer.bos_token_id = tokenizer.cls_token_id
    print('len(tokenizer) = ', len(tokenizer))
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.bos_token, tokenizer.bos_token_id)
    embedding_layer = model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def train_prefix(label_id_List, label_list, sentence_list, sentence_tokenizer, gpt2, tokenizer, model_args, data_args, training_args, prefix_path, ratio, task_mode=None, aug_epoch=0,aug_iter=0, examples_per_class=0, classes_per_episode=0, num_support=0, max_length=0, warm_epoch=0, prefixModel_name_or_path=None, preseqlen=100,tuning_mode='prefixtune', device='cuda:0',encoder_config=None, data_type = None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model_args.tuning_mode = tuning_mode
    model_args.prefixModel_name_or_path = prefixModel_name_or_path
    training_args.output_dir = prefix_path
    sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(sys_path)
    print(sys.path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(sys_path, '.cache/huggingface')
    print(int(300 * ratio))
    training_args.num_train_epochs = int(aug_epoch)
    #print(training_args)
    #training_args.device = device
    #setattr(training_args, 'device', device)
    print(training_args.device)
    training_args.per_device_train_batch_size = examples_per_class*classes_per_episode
    print("training args", file=sys.stderr)
    print(training_args, file=sys.stderr)

    print("model args", file=sys.stderr)
    print(model_args, file=sys.stderr)

    print("model arg length")
    print(model_args.preseqlen)
    model_args.preseqlen = preseqlen
    encoder_embed = encoder_config['hidden_size'] * 2
    data_args.task_mode = task_mode
    alpha = 1000
    training_args.save_steps = 100000
    training_args.warmup_steps = int(aug_iter/training_args.n_gpu) * warm_epoch
    n_support = 0
    iterations = 0
    sample_per_class = 0
    classes_per_it = 0
    if data_args.task_mode == 'proto' or data_args.task_mode == 'proto_reg':
        #encoder_embed = 0
        print("devices number =====================")
        print(training_args.n_gpu)
        training_args.per_device_train_batch_size = 1
        iterations = aug_iter - aug_iter%training_args.n_gpu



        sample_per_class = examples_per_class
        classes_per_it = classes_per_episode
        n_support = num_support

    # Set seed
    set_seed(training_args.seed)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    ####################### bert encoder #########################################
    """
    sentence_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) , '.cache/huggingface'))
    sentence_encoder = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) , '.cache/huggingface')).to(training_args.device)
    for param in sentence_encoder.parameters():
        param.requires_grad = False


    file_path = data_args.train_data_file

    src_file = '{}/trn.source'.format(file_path)
    tgt_file = '{}/trn.target'.format(file_path)

    src_lines = []
    with open(src_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and not line.isspace():
                src_lines.append(line.split('\t')[0])

    tgt_lines = []
    with open(tgt_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and not line.isspace():
                tgt_lines.append(line)

    assert len(src_lines) == len(tgt_lines)
    proto_type_dict = {}
    for index, tgt_line in enumerate(tgt_lines):
        dis_tgt = sentence_tokenizer(tgt_line, add_special_tokens=True, truncation=True, max_length=100,
                                is_split_into_words=False, return_tensors="pt")['input_ids']
        dis_tgt = dis_tgt.to(training_args.device)
        # print (type(sentence_encoder))
        # print (sentence_encoder(dis_tgt))
        sentences_embed = sentence_encoder(dis_tgt, return_dict=True).pooler_output.squeeze()
        #print(sentences_embed.size())
        if src_lines[index] in proto_type_dict:
            proto_type_dict[src_lines[index]].append(sentences_embed)
        else:
            proto_type_dict[src_lines[index]] = []
            proto_type_dict[src_lines[index]].append(sentences_embed)

    proto_type_embed = {}
    proto_dict = {}
    for label_id, sentences_embed in proto_type_dict.items():
        gpu_data = torch.stack(sentences_embed).mean(dim=0)
        proto_type_embed[int(label_id)] = gpu_data
        data = torch.stack(sentences_embed).cpu().numpy()
        #print(data.shape)
        mean = np.mean(data, axis=0)
        if not label_id in proto_dict:
            proto_dict[label_id] = {'mean': None, 'cov': None}
        proto_dict[label_id]['mean'] = mean
        cov = np.cov(data, rowvar=0)
        proto_dict[label_id]['cov'] = cov
    #print(len(proto_dict.items()))
    torch.save(proto_dict, os.path.join(training_args.output_dir, "proto_type_dict.bin"))
    """
    ####################### bert encoder #########################################


    if model_args.tuning_mode == 'prefixtune' or model_args.tuning_mode == 'bothtune':  # prefixtune
        if model_args.tuning_mode == 'prefixtune':
            for param in gpt2.base_model.parameters():
                param.requires_grad = False
        elif model_args.tuning_mode == 'bothtune':
            for param in gpt2.base_model.parameters():
                param.requires_grad = True

        #gpt2 = model

        print('loading the prefix model from ', model_args.prefixModel_name_or_path)
        # print(bool(".ckpt" in model_args.prefixModel_name_or_path))
        if model_args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif model_args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if model_args.prefixModel_name_or_path is not None:
            config2 = AutoConfig.from_pretrained(model_args.prefixModel_name_or_path, cache_dir=model_args.cache_dir)
            # print(config2)

            if model_args.prefix_mode == 'embedding':
                model = PrefixEmbTuning.from_pretrained(
                    model_args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                    config=config2,
                    cache_dir=model_args.cache_dir,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                    use_infix=(data_args.format_mode == 'infix')
                )

            elif model_args.prefix_mode == 'activation':

                model = PrefixTuning.from_pretrained(
                    model_args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                    config=config2,
                    cache_dir=model_args.cache_dir,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                    use_infix=(data_args.format_mode == 'infix'),
                    encoder_config=encoder_config
                )
            else:
                assert False, "invalid prefix mode"
            discri_labels = None
        else:

            # should clone the config and construct it.
            config_prefix = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
            config_prefix._my_arg_tune_mode = model_args.tuning_mode
            config_prefix._my_arg_task_mode = data_args.task_mode
            config_prefix._my_arg_control = True
            config_prefix.train_weights = data_args.train_embs
            config_prefix.optim_prefix = optim_prefix_bool
            config_prefix.preseqlen = model_args.preseqlen
            config_prefix.use_infix = (data_args.format_mode == 'infix')
            config_prefix.format_mode = data_args.format_mode
            config_prefix.prefix_dropout = model_args.prefix_dropout
            config_prefix.vocab_size = len(tokenizer)
            config_prefix.lowdata = ('lowdata' in training_args.output_dir)
            if config_prefix.lowdata and data_args.use_lowdata_token == 'yes':
                config_prefix.lowdata_token = tokenizer([data_args.lowdata_token],
                                                        add_prefix_space=True)['input_ids']  # return_tensors='np',
                print(data_args.lowdata_token)
                print(config_prefix.lowdata_token)

            # some extra stuff.
            config_prefix.init_random = model_args.init_random
            config_prefix.mid_dim = model_args.mid_dim

            config_prefix.encoder_embed = encoder_embed
            if data_args.task_mode == 'proto' or data_args.task_mode == 'proto_reg':
                config_prefix.n_support = n_support
                config_prefix.classes_per_it = classes_per_it
            print('training the prefix model from scratch. ')
            #if model_args.prefix_mode == 'embedding':

                # specific parametrization for embedding.
            #    config_prefix.parametrize_emb = model_args.parametrize_emb
            #    model = PrefixEmbTuning(config_prefix, model_gpt2=gpt2)

            #elif model_args.prefix_mode == 'activation':
            #    model = PrefixTuning(config_prefix, model_gpt2=gpt2)
            #else:
            #    assert False, "invalid prefix mode"
            print('Not in dataless setting, loading the control code. ')

            discri_labels = None

                # should clone the config and construct it.
            config_prefix = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
            config_prefix._my_arg_tune_mode = model_args.tuning_mode
            config_prefix._my_arg_task_mode = data_args.task_mode
            config_prefix._my_arg_control = True
            config_prefix.train_weights = data_args.train_embs
            config_prefix.optim_prefix = optim_prefix_bool
            config_prefix.preseqlen = model_args.preseqlen
            config_prefix.use_infix = (data_args.format_mode == 'infix')
            config_prefix.format_mode = data_args.format_mode
            config_prefix.prefix_dropout = model_args.prefix_dropout
            config_prefix.vocab_size = len(tokenizer)
            config_prefix.lowdata = ('lowdata' in training_args.output_dir)
            if config_prefix.lowdata and data_args.use_lowdata_token == 'yes':
                config_prefix.lowdata_token = tokenizer([data_args.lowdata_token],
                                                        add_prefix_space=True)['input_ids']  # return_tensors='np',
                print(data_args.lowdata_token)
                print(config_prefix.lowdata_token)

            # some extra stuff.
            config_prefix.init_random = model_args.init_random
            config_prefix.mid_dim = model_args.mid_dim
            config_prefix.encoder_embed = encoder_embed
            config_prefix.n_support = n_support
            config_prefix.classes_per_it = classes_per_it
            print('training the prefix model from scratch. ')
            if model_args.prefix_mode == 'embedding':
                config_prefix.parametrize_emb = model_args.parametrize_emb

                model = PrefixEmbTuning(config_prefix, model_gpt2=gpt2)

            elif model_args.prefix_mode == 'activation':
                model = PrefixTuning(config_prefix, model_gpt2=gpt2, encoder_config=encoder_config)

            else:
                assert False, "invalid prefix mode"

    #model.encoder_embed = encoder_embed
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer,sentence_tokenizer=sentence_tokenizer,  label_id_List = label_id_List,
    label_list = label_list,
    sentence_list = sentence_list, cache_dir=model_args.cache_dir, iterations = iterations, sample_per_class = sample_per_class, classes_per_it = classes_per_it, max_seq_length=max_length, data_type=data_type) if training_args.do_train else None
    )

    if data_args.task_mode == 'cl':
        data_collator = DataCollatorForCLModeling(
            tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
            format_mode=data_args.format_mode
        )
    elif data_args.task_mode in ['topic', 'no_rel', 'no_hidden', 'privacy', 'casual_lens']:
        data_collator = DataCollatorForTopicClusterLanguageModeling(tokenizer = tokenizer, sentence_tokenizer = sentence_tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
            format_mode=data_args.format_mode)
    elif data_args.task_mode == 'proto' or data_args.task_mode == 'proto_reg':
        data_collator = DataCollatorForProtoClusterLanguageModeling(tokenizer = tokenizer, sentence_tokenizer = sentence_tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
            format_mode=data_args.format_mode)

    # set prefix tuning extra parameters
    #model.proto_type_embed = proto_type_embed
    #model.task_mode = data_args.task_mode

    if (model_args.tuning_mode == 'prefixtune'):
        trainer = Trainer_Prefix(
            model=model,
            tokenizer=tokenizer,
            discri_labels=discri_labels,
            model_gpt2=gpt2,
            args=training_args,
            prediction_loss_only=True,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
            task_mode=data_args.task_mode,
            use_dropout=(model_args.use_dropout == 'yes'),
            alpha = alpha
        )
    elif (model_args.tuning_mode == 'bothtune'):
        print('BOTH TUNE for trainer prefix. ')
        trainer = Trainer_Prefix(
            model=model,
            tokenizer=tokenizer,
            discri_labels=discri_labels,
            model_gpt2=gpt2,
            args=training_args,
            prediction_loss_only=True,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
            task_mode=data_args.task_mode,
            use_dropout=(model_args.use_dropout == 'yes'),
            both_tune=True,
            alpha=alpha
        )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        if not (data_args.dataless == 'yes'):
            trainer.train(model_path=model_path)
        elif False:
            trainer.train_dataless(model_path=model_path, verbose=True)
        else:
            trainer.train_amortized_pplm(model_path=model_path, verbose=True)

        if 'lowdata' not in training_args.output_dir:
            trainer.save_model()

            if model_args.tuning_mode == 'bothtune':
                gpt2_dir = os.path.join(training_args.output_dir, 'gpt2')
                print("saving GPT2  model to", gpt2_dir)
                print(gpt2)
                gpt2.save_pretrained(gpt2_dir)

    return model

