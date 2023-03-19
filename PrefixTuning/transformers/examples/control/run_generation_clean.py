#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import argparse
import logging
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import json
import os
import sys
import nltk
from nltk import TweetTokenizer

sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(sys_path)
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(sys_path)
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(sys_path)
print (sys.path)

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    BertForMaskedLM, BertModel,
    BertTokenizer, BertTokenizerFast, AutoConfig,
    set_seed,
    GPT2LMHeadModelAdapter,
)

from .train_control import PrefixTuning, PrefixEmbTuning

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}

def read_e2e_files(path, tokenizer, lowdata_token=None):
    file_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src, tgt = line.strip().split('||')
            # URGENT CHANGE
            # src =  src + ' {}'.format(' summarize :')
            if lowdata_token is None:
                src = ' {} {}'.format(src, tokenizer.bos_token)
                # src =  src + ' {}'.format(tokenizer.bos_token)
            else:
                src = ' {} {} {}'.format(lowdata_token, src, tokenizer.bos_token)
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    return file_dict

def read_wp_files(path, tokenizer):
    file_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src, tgt = line.strip().split('|||')
            src = src + ' {}'.format(tokenizer.bos_token)
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    return file_dict


def read_classifySentiment_files(path, tokenizer):
    file_dict = []
    with open(path, 'r') as f:
        for line in f:
            tgt, src = line.strip().split('|||')
            src = src.replace("< br / >", "\n")
            src = ' {} {}'.format(src, tokenizer.bos_token)
            file_dict.append((src, tgt))
    return file_dict

def read_classifyTopic_files(path, tokenizer):
    file_dict = []
    with open(path, 'r') as f:
        for line in f:
            if (len(line) > 0 and not line.isspace()
                    and len(line.split('||')) == 2):
                tgt, src = line.strip().split('||')
            else:
                continue
            src = ' {} {}'.format(src, tokenizer.bos_token)
            file_dict.append((src, tgt))
    return file_dict

def read_sum_files(path, tokenizer, max_source_length, max_target_length):
    src_file = 'relation_data/test.source'
    tgt_file = 'relation_data/test.target'

    file_dict = {}

    src_lines = []
    src_ids = []
    with open(src_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and not line.isspace():
                src_lines.append(line.split('\t')[1])
                src_ids.append(line.split('\t')[0])
    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = [line.split('\t')[1] for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    print(tgt_file, len(tgt_lines), '\n', src_file, len(src_lines))
    src_full_list = []
    for src, tgt in zip(src_lines, tgt_lines):
        src_bpe = tokenizer.encode(
            src, add_special_tokens=False, truncation=True, max_length=max_source_length,
            is_split_into_words=False
        )

        src_full = src_bpe + [tokenizer.bos_token_id] # add the bos token.
        #src_full = src_bpe

        # print(len(src_full), src_full)
        src_full = tuple(src_full)
        src_full_list.append(src_full)
        if src_full not in file_dict:
            file_dict[src_full] = [tgt]
        else:
            print('should not happen')
            file_dict[src_full].append(tgt)
    #print(file_dict)
    return src_full_list, src_ids

def read_webnlg_files(path, tokenizer):
    file_dict = {}

    with open(path) as f:
        lines_dict = json.load(f)

    full_rela_lst = []
    full_src_lst = []
    # full_tgt_lst = []
    total_count = 0
    for i, example in enumerate(lines_dict['entries']):
        sents = example[str(i + 1)]['lexicalisations']
        triples = example[str(i + 1)]['modifiedtripleset']

        rela_lst = []
        temp_triples = ''
        for j, tripleset in enumerate(triples):
            subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
            rela_lst.append(rela)
            if i > 0:
                temp_triples += ' | '
            temp_triples += '{} : {} : {}'.format(subj, rela, obj)

        temp_triples = ' {} {}'.format(temp_triples, tokenizer.bos_token)


        for sent in sents:
            if True: #sent["comment"] == 'good'
                if (temp_triples,tuple(rela_lst)) not in file_dict:
                    file_dict[(temp_triples,tuple(rela_lst))] = []
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(tuple(rela_lst))
                file_dict[(temp_triples,tuple(rela_lst))].append(sent["lex"])


    print(len(file_dict), len(full_src_lst))
    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)

    return file_dict


def read_triples_files2(path, tokenizer):
    file_src = []
    file_tgt = []

    with open(path) as f:
        lines_dict = json.load(f)

    print(len(lines_dict))
    full_rela_lst = []
    full_src_lst = []
    for example in lines_dict:
        rela_lst = []
        temp_triples = ''
        for i, tripleset in enumerate(example['tripleset']):
            subj, rela, obj = tripleset
            rela = rela.lower()
            rela_lst.append(rela)
            if i > 0:
                temp_triples += ' | '
            temp_triples += '{} : {} : {}'.format(subj, rela, obj)

        temp_triples = ' {} {}'.format(temp_triples, tokenizer.bos_token)

        file_src.append((temp_triples, tuple(rela_lst)))
        # file_tgt

        for sent in example['annotations']:
            if (temp_triples, tuple(rela_lst)) not in file_dict:
                file_dict[(temp_triples, tuple(rela_lst))] = []
                full_src_lst.append(temp_triples)
                full_rela_lst.append(tuple(rela_lst))
            file_dict[(temp_triples, tuple(rela_lst))].append(sent['text'])

    print(len(file_dict), len(full_src_lst))
    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)
    return file_dict

def read_triples_files(path, tokenizer):
    file_dict = {}

    with open(path) as f:
        lines_dict = json.load(f)

    print(len(lines_dict))
    full_rela_lst = []
    full_src_lst = []
    for example in lines_dict:
        rela_lst = []
        temp_triples = ''
        for i, tripleset in enumerate(example['tripleset']):
            subj, rela, obj = tripleset
            rela = rela.lower()
            rela_lst.append(rela)
            if i > 0:
                temp_triples += ' | '
            temp_triples += '{} : {} : {}'.format(subj, rela, obj)

        temp_triples = ' {} {}'.format(temp_triples, tokenizer.bos_token)

        for sent in example['annotations']:
            if (temp_triples, tuple(rela_lst)) not in file_dict:
                file_dict[(temp_triples, tuple(rela_lst))] = []
                full_src_lst.append(temp_triples)
                full_rela_lst.append(tuple(rela_lst))
            file_dict[(temp_triples, tuple(rela_lst))].append(sent['text'])

    print(len(file_dict), len(full_src_lst))
    assert len(full_rela_lst) == len(full_src_lst)
    assert len(full_rela_lst) == len(file_dict)
    return file_dict

# def write_e2e_corr(prompt_lst, file_dict, corr_path):
#     with open(corr_path, 'w') as f:
#         for x in prompt_lst:
#             for line in file_dict[x]:
#                 print(line, file=f)
#             print('', file=f)
#     return

def write_e2e_corr(prompt_lst, file_dict, corr_path):
    print(len(prompt_lst))
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            for line in file_dict[x]:
                if not line.strip():
                    print('PROBLEM', line,'PROBLEM',file_dict[x] )
                else:
                    print(line, file=f)
            print('', file=f)

    # buf = [[]]
    # with open(corr_path, 'r') as fh:
    #     for line in fh:
    #         line = line.strip()
    #         if True:
    #             # print(line)
    #             if not line:
    #                 buf.append([])
    #             else:
    #                 buf[-1].append(line)
    #         else:
    #             buf.append(line)
    # if not buf[-1]:
    #     del buf[-1]
    #
    # print(buf[:3])
    #
    # print(len(buf))

    return

def write_e2e_src(prompt_lst, corr_path):
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            print(x, file=f)
    return



def get_emb(sent_lst, word_lst, num_layer=1):
    # load bert
    tokenizer_bert = BertTokenizerFast.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained('bert-large-uncased', return_dict=True).cuda()
    for param in model.parameters():
        param.requires_grad = False

    device = model.device

    edited_sent = []
    chosen_word = []
    with torch.no_grad():
        computed_ = 0
        mid_ = 300
        full_score = []
        while computed_ < len(sent_lst):
            temp_sent = sent_lst[computed_:computed_ + mid_]
            temp_word = word_lst[computed_:computed_ + mid_]
            temp_input = tokenizer_bert(temp_sent, return_tensors="pt", padding=True,
                                        is_split_into_words=False, return_offsets_mapping=True, add_special_tokens=True)
            input_ids = temp_input["input_ids"]
            # print(temp_input.keys())
            mask_input = temp_input['attention_mask']
            bsz, seqlen = input_ids.shape

            # print(input_ids.shape)

            cand_idx = tokenizer_bert(temp_word, add_special_tokens=False)['input_ids']
            # print(cand_idx)
            # if BPE has multiple subwords.
            cand_idx = torch.tensor([i[-1] for i in cand_idx])  # bsz
            # print(cand_idx)
            cand_idx2 = cand_idx.unsqueeze(1).expand(bsz, seqlen)

            mask = (input_ids == cand_idx2)
            # print(mask.sum(dim=1))
            # print(mask.nonzero())

            # what if the occurence of a subword is not in the primary word?

            # if has multiple occurence? only taking the first one.
            mask = (mask.cumsum(dim=1) == 1) & mask
            # print(mask)
            # print(mask.sum(dim=1))
            # print(mask.nonzero())
            mask_idx = mask.nonzero()

            # print(input_ids.shape)

            edit_temp = []
            keep_mask = []
            word_temp = []
            for i, (sent1, word1) in enumerate(zip(temp_sent, temp_word)):
                # TODO: could check against the offests and make final changes!
                temp_idx1 = temp_input["offset_mapping"][i][mask_idx[i, 1]]
                # print(word1, sent1)
                # print(sent1[temp_idx1[0]:temp_idx1[1]])
                sent1 = sent1.split()
                widx = sent1.index(word1)
                by_tokenl = sum([len(l) + 1 for l in sent1[:widx]])
                by_tokenr = sum([len(l) + 1 for l in sent1[:widx + 1]]) - 1
                # print(by_tokenl, by_tokenr, temp_idx1)
                if by_tokenl != temp_idx1[0].item() and by_tokenr != temp_idx1[1].item():
                    # print('dangerous')
                    # print(sent1, word1, by_tokenl, by_tokenr, temp_idx1)
                    # simple option: delete it form input_ids
                    keep_mask.append(False)
                    continue
                else:
                    keep_mask.append(True)
                new_sent = [word1, '[BOS]'] + sent1[:widx] + ['[', sent1[widx], ']'] + sent1[widx + 1:] + ['[EOS]']
                assert len(new_sent) == len(sent1) + 5
                edit_temp.append(new_sent)
                word_temp.append(word1)

            keep_mask = torch.tensor(keep_mask)
            # print(keep_mask.shape, input_ids.shape, mask.shape, 'hi')
            input_ids = input_ids[keep_mask]
            mask = mask[keep_mask]
            mask_input = mask_input[keep_mask]
            # print(input_ids.shape, mask.shape, len(edit_temp))
            assert input_ids.size(0) == len(edit_temp)

            edited_sent += edit_temp
            chosen_word += word_temp
            # print(len(edited_sent), len(chosen_word))

            outputs = model(input_ids.to(device), attention_mask=mask_input.to(device), output_hidden_states=True)

            if num_layer > 1:
                all_hidden_states = outputs.hidden_states
                selected_all_hidden_states = [ii[mask] for ii in all_hidden_states[-num_layer:]]
                # print([ii.shape for ii in selected_all_hidden_states])
                hidden_layer = torch.stack(selected_all_hidden_states, dim=1)
                # print(hidden_layer.shape, selected_all_hidden_states[0].shape)
                # print('all hidden', selected_all_hidden_states.shape)

            else:
                last_hidden_states = outputs.last_hidden_state
                hidden_layer = last_hidden_states[mask].unsqueeze(1)


            computed_ += mid_
            full_score.append(hidden_layer.cpu())

        full_score = torch.cat(full_score, dim=0)

    return full_score, edited_sent, chosen_word

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def read_doc_for_embmatch(file_name, num_layer):
    word_lst = []
    sent_lst = []
    with open(file_name, 'r') as f:
        for line in f:
            word, sent = line.strip().split('||')
            word_lst.append(word)
            sent_lst.append(sent)

    emb_match, sent_cleaned_lst, chosen_word = get_emb(sent_lst, word_lst, num_layer=num_layer)
    prompt_text_lst = [word + ' [BOS]' for word in chosen_word]
    return prompt_text_lst, emb_match.split(1), sent_cleaned_lst

def load_prefix_model(args, prefixModel_name_or_path, gpt2, config, encoder_config):
    args.device = config['device']
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    if args.optim_prefix == 'yes':
        optim_prefix_bool = True
    elif args.optim_prefix == 'no':
        optim_prefix_bool = False
    else:
        assert False, "model_args.optim_prefix should be either yes or no"

    if prefixModel_name_or_path is not None:

        plmm_config = AutoConfig.from_pretrained(prefixModel_name_or_path, cache_dir=args.cache_dir)
        #print(config)

        if args.prefix_mode == 'embedding':
            model = PrefixEmbTuning.from_pretrained(
                prefixModel_name_or_path,
                from_tf=bool(".ckpt" in prefixModel_name_or_path, ),
                config=plmm_config,
                model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=args.preseqlen,
                use_infix=(args.format_mode == 'infix')
            )

        elif args.prefix_mode == 'activation':
            print("sequence length ========================")
            args.preseqlen = 100
            print (args.preseqlen)
            model = PrefixTuning.from_pretrained(
                prefixModel_name_or_path,
                from_tf=bool(".ckpt" in prefixModel_name_or_path, ),
                config=plmm_config,
                model_gpt2=gpt2, optim_prefix=plmm_config.optim_prefix, preseqlen=plmm_config.preseqlen,
                use_infix=(args.format_mode == 'infix'),
                encoder_config=encoder_config
            )

        model.to(args.device)
    else:
        assert False, "prefixModel_name_or_path is NONE."

    return model

def generate_instances(args, prefix_model, label_ids, label_tokens, gpt2, proto_type_embed):
    nltk_tokenizer = TweetTokenizer()
    args.num_return_sequences = 5
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args.seed)

    # Initialize the model and tokenizer
    if args.tuning_mode == 'prefixtune':

        print('loading from PrefixTuning.', args.prefixModel_name_or_path,)
        if args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            assert False, 'shouldn not init config from scratch. '
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        try:
            args.model_type = args.model_type.lower()
            model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        if args.model_name_or_path:
            print('loading the trained tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        elif args.tokenizer_name:
            print('loading from the init tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

        config._my_arg_tune_mode = args.tuning_mode
        config._my_arg_task_mode = args.task_mode
        config._objective_mode = args.objective_mode
        print(config)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        add_pad = False

        if args.model_name_or_path == 'gpt2-medium':
            if args.task_mode == 'dataless':
                print(args.tuning_mode, 'dataless setting, so no new tokens at all.')
                print('We do not add special tokens to the tokenizer, instead, we just finetune on <|endoftext|>')

                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

            elif add_pad:
                print('extending the size of word embeddings. to include the [PAD] ')
                num_added_tokens = tokenizer.add_special_tokens(
                    {'pad_token': '[PAD]'})
                embedding_layer = model.resize_token_embeddings(len(tokenizer))
            else:
                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        if args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if args.prefixModel_name_or_path is not None:
            model = prefix_model
            config = model.config
        else:
            assert False, "prefixModel_name_or_path is NONE."

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    decode_mode = 'beam'

    text_instances = []
    for prompt_idx, label_token in enumerate(label_tokens):
        label_id = label_ids[prompt_idx]
        label_token_bpe = tokenizer.encode(
            label_token, add_special_tokens=False, truncation=True, max_length=100,
            is_split_into_words=False
        )

        label_token_full = label_token_bpe + [tokenizer.bos_token_id] # add the bos token.
        print(label_token_full)
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        elif args.task_mode == 'cnndm' or args.task_mode == 'xsum':
            # already processed
            encoded_prompt = torch.LongTensor(label_token_full).unsqueeze(0)
            print(encoded_prompt.shape)
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            print('****************',prefix)
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        if args.task_mode == 'embMatch' and args.control_dataless != 'yes':
            print(' '.join(sent_cleaned_lst[prompt_idx]))
            emb_match_temp = emb_match[prompt_idx].to(model.device).expand(args.num_return_sequences, -1, -1)
        else:
            emb_match_temp = None

        if args.control_mode == 'yes' and args.control_dataless != 'yes':
            # URGENT, check whether the next line is necessary?
            # control_code = torch.LongTensor(control_codes[prompt_idx]).to(model.device).unsqueeze(0).expand(args.num_return_sequences, -1)
            control_code = None
            pass

        else:
            control_code = None

        if args.tuning_mode == 'prefixtune' or args.tuning_mode == 'bothtune':
            print(config.optim_prefix, optim_prefix_bool)
            print('control code is ', control_code)
            print('prompt idx is ', prompt_idx)
            print('label id is ',label_id)
            print('label tokens are ', label_token)
            for h in proto_type_embed[label_id]:
                #mean = proto_dict[str(label_id)]['mean']
                #logv = proto_dict[str(label_id)]['cov']
                #h = torch.from_numpy(np.random.multivariate_normal(mean, logv).astype(np.float32))

                #print(h.size())
                proto_type =  h.unsqueeze(0).unsqueeze(0).to(model.device)
                prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=1, sentences_embed=proto_type)

                prompt = [x.expand(-1, args.num_return_sequences , -1, -1, -1) for x in prompt]

                #print(decode_mode)
                #print(prompt[0].size())
                if decode_mode == 'nucleus':
                    output_sequences = gpt2.generate(
                        input_ids=input_ids,
                        emb_match=None,
                        control_code=None,
                        past_key_values=prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=0.8,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=True,
                        num_return_sequences=args.num_return_sequences,
                    )
                elif decode_mode == 'beam':
                    output_sequences = gpt2.generate(
                        input_ids=input_ids,
                        emb_match=None,
                        control_code=None,
                        past_key_values=prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        min_length=5,
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=0.9, #top_p=0.5,
                        repetition_penalty=args.repetition_penalty, ##args.repetition_penalty,
                        do_sample=False,
                        num_beams=args.num_return_sequences,
                        bad_words_ids=[[628], [198]] if True else None,
                        num_return_sequences=args.num_return_sequences,
                    )
                    # print(output_sequences)

                elif decode_mode == 'greedy':
                    output_sequences = gpt2.generate(
                        input_ids=input_ids,
                        emb_match=None,
                        control_code=None,
                        past_key_values=prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        min_length=5,
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=0.5,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=False,
                        bad_words_ids=[[628], [198]] if True else None,
                        num_return_sequences=1,
                    )


                # Remove the batch dimension when returning multiple sequences
                if len(output_sequences.shape) > 2:
                    output_sequences.squeeze_()

                output_sequences_list = output_sequences.tolist()
                #print("original")
                #print(output_sequences_list)
                df = pd.DataFrame(output_sequences_list)
                df = df.drop_duplicates()
                output_sequences_list = df.values.tolist()
                #print("filter")
                #print(output_sequences_list)
                index_arr = np.arange(len(output_sequences_list))
                np.random.shuffle(index_arr)
                index_arr_ids = list(index_arr)

                output_sequences_list = [output_sequences_list[i] for i in index_arr_ids]
                #print("after")
                #print(output_sequences_list)
                generated_sequences = []

                for generated_sequence_idx, generated_sequence in enumerate(output_sequences_list):
                    # args.stop_token = tokenizer.eos_token
                    #generated_sequence = generated_sequence.tolist()
                    #print(generated_sequence)
                    #if 50257 in generated_sequence:
                    #    continue
                    print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                    #eos_index = generated_sequence.index(50256)
                    #print(generated_sequence)
                    generated_sequence = [token_id for token_id in generated_sequence if token_id != 50257]
                    #print(generated_sequence)
                    # Decode text
                    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                    #print(text)
                    text_output = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                    idx = text_output.find(tokenizer.eos_token)
                    if idx >= 0:
                        text_output = text_output[:idx]
                    text_output = text_output.strip()

                    if args.task_mode == 'topic' or args.task_mode == 'sentiment':
                        text_output = prompt_text + ' ' + text_output + ' [SPECIAL_END]'

                    if text_output:
                        text_output = " ".join(nltk_tokenizer.tokenize(text_output))
                        print(text_output)
                        generated_sequences.append((label_id, text_output))
                        #break
                    else:
                        print('Error')
                        #raise ValueError
                #assert len(generated_sequences) == 1
                text_instances.extend(generated_sequences)
    return text_instances



def generate_optimus_instances(args, prefix_model, label_ids, label_tokens, gpt2, topic_embed_dict, num_return_sequences=1):
    print("===================================================================================")
    print(args.prefixModel_name_or_path)
    nltk_tokenizer = TweetTokenizer()
    args.num_return_sequences = num_return_sequences
    args.device = prefix_model.device
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args.seed)

    # Initialize the model and tokenizer
    if args.tuning_mode == 'prefixtune':

        print('loading from PrefixTuning.', args.prefixModel_name_or_path,)
        if args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            assert False, 'shouldn not init config from scratch. '
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        try:
            args.model_type = args.model_type.lower()
            model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        if args.model_name_or_path:
            print('loading the trained tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        elif args.tokenizer_name:
            print('loading from the init tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

        config._my_arg_tune_mode = args.tuning_mode
        config._my_arg_task_mode = args.task_mode
        config._objective_mode = args.objective_mode
        print(config)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        add_pad = False

        if args.model_name_or_path == 'gpt2-medium':
            if args.task_mode == 'dataless':
                print(args.tuning_mode, 'dataless setting, so no new tokens at all.')
                print('We do not add special tokens to the tokenizer, instead, we just finetune on <|endoftext|>')

                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

            elif add_pad:
                print('extending the size of word embeddings. to include the [PAD] ')
                num_added_tokens = tokenizer.add_special_tokens(
                    {'pad_token': '[PAD]'})
                embedding_layer = model.resize_token_embeddings(len(tokenizer))
            else:
                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        if args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if args.prefixModel_name_or_path is not None:
            model = prefix_model
            config = model.config
        else:
            assert False, "prefixModel_name_or_path is NONE."

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    decode_mode = 'beam'

    text_instances = []

    for prompt_idx, label_token in enumerate(label_tokens):

        label_id = label_ids[prompt_idx]

        label_token_bpe = tokenizer.encode(
            label_token, add_special_tokens=False, truncation=True, max_length=100,
            is_split_into_words=False
        )
        if config._my_arg_task_mode == 'no_rel':
            label_token_full = [tokenizer.bos_token_id]  # add the bos token.
        else:
            label_token_full = label_token_bpe + [tokenizer.bos_token_id] # add the bos token.


        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        elif args.task_mode == 'cnndm' or args.task_mode == 'xsum':
            # already processed
            encoded_prompt = torch.LongTensor(label_token_full).unsqueeze(0)
            print(encoded_prompt.shape)
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            print('****************',prefix)
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        if args.task_mode == 'embMatch' and args.control_dataless != 'yes':
            print(' '.join(sent_cleaned_lst[prompt_idx]))
            emb_match_temp = emb_match[prompt_idx].to(model.device).expand(args.num_return_sequences, -1, -1)
        else:
            emb_match_temp = None

        if args.control_mode == 'yes' and args.control_dataless != 'yes':
            # URGENT, check whether the next line is necessary?
            # control_code = torch.LongTensor(control_codes[prompt_idx]).to(model.device).unsqueeze(0).expand(args.num_return_sequences, -1)
            control_code = None
            pass

        else:
            control_code = None

        if args.tuning_mode == 'prefixtune' or args.tuning_mode == 'bothtune':
            print(config.optim_prefix, optim_prefix_bool)
            print('control code is ', control_code)
            print('prompt idx is ', prompt_idx)
            print('label id is ',label_id)
            print('label tokens are ', label_token)

            topic_embeds = topic_embed_dict[label_token]

            batch_index_arr = np.arange(len(topic_embeds))
            batch_size = 96
            batch_num = int(np.ceil(len(topic_embeds) / float(batch_size)))
            #print(batch_num)
            for batch_id in range(batch_num):
                #print (batch_id)
                #print(batch_size)
                batch_ids = batch_index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
                current_bsz = len(batch_ids)
                #print(current_bsz)
                batch_topic_embeds = [topic_embeds[i] for i in batch_ids]
                #print(len(batch_topic_embeds))
                topic_embed = torch.cat(batch_topic_embeds, dim=0)
            #for topic_index, topic_embed in enumerate(topic_embeds):
                #print(topic_embed.size())
                topic_embed = topic_embed.unsqueeze(1)
                prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=topic_embed.size(0), topic_embed=topic_embed)
                #print(prompt[0].size())
                prompt = [x.expand(-1, args.num_return_sequences*current_bsz , -1, -1, -1) for x in prompt]
                expand_input_ids = input_ids.expand(current_bsz, input_ids.size(1))
                #print(input_ids.size())
                #print(prompt.size())

                #print(decode_mode)
                #print(prompt[0].size())
                #print(expand_input_ids)
                if decode_mode == 'nucleus':
                    output_sequences = gpt2.generate(
                        input_ids=expand_input_ids,
                        emb_match=None,
                        control_code=None,
                        past_key_values=prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=0.8,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=True,
                        num_return_sequences=args.num_return_sequences,
                    )
                elif decode_mode == 'beam':
                    output_sequences = gpt2.generate(
                        input_ids=expand_input_ids,
                        emb_match=None,
                        control_code=None,
                        past_key_values=prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        min_length=5,
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=0.9, #top_p=0.5,
                        repetition_penalty=args.repetition_penalty, ##args.repetition_penalty,
                        do_sample=False,
                        num_beams=args.num_return_sequences,
                        bad_words_ids=[[628], [198]] if True else None,
                        num_return_sequences=args.num_return_sequences,
                    )
                    # print(output_sequences)

                elif decode_mode == 'greedy':
                    output_sequences = gpt2.generate(
                        input_ids=expand_input_ids,
                        emb_match=None,
                        control_code=None,
                        past_key_values=prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        min_length=5,
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=0.5,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=False,
                        bad_words_ids=[[628], [198]] if True else None,
                        num_return_sequences=1,
                    )


                # Remove the batch dimension when returning multiple sequences
                if len(output_sequences.shape) > 2:
                    output_sequences.squeeze_()

                output_sequences_list = output_sequences.tolist()
                #print("original")
                #print(output_sequences_list)
                df = pd.DataFrame(output_sequences_list)
                df = df.drop_duplicates()
                output_sequences_list = df.values.tolist()
                #print("filter")
                #print(output_sequences_list)
                #index_arr = np.arange(len(output_sequences_list))
                #np.random.shuffle(index_arr)
                #index_arr_ids = list(index_arr)

                #output_sequences_list = [output_sequences_list[i] for i in index_arr_ids]
                #print("after")
                #print(output_sequences_list)
                generated_sequences = []

                for generated_sequence_idx, generated_sequence in enumerate(output_sequences_list):
                    # args.stop_token = tokenizer.eos_token
                    #generated_sequence = generated_sequence.tolist()
                    #print(generated_sequence)
                    #if 50257 in generated_sequence:
                    #    continue
                    #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                    #eos_index = generated_sequence.index(50256)
                    #print(generated_sequence)
                    generated_sequence = [token_id for token_id in generated_sequence if token_id != 50257]
                    #print(generated_sequence)
                    # Decode text
                    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                    #print(text)
                    text_output = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                    idx = text_output.find(tokenizer.eos_token)
                    if idx >= 0:
                        text_output = text_output[:idx]
                    text_output = text_output.strip()

                    if args.task_mode == 'topic' or args.task_mode == 'sentiment':
                        text_output = prompt_text + ' ' + text_output + ' [SPECIAL_END]'

                    if text_output:
                        text_output = " ".join(nltk_tokenizer.tokenize(text_output))
                        #print(text_output)
                        generated_sequences.append((label_id, text_output))
                        #break
                    else:
                        generated_sequences.append((label_id, label_token))
                        #raise ValueError
                #assert len(generated_sequences) == 1
                text_instances.extend(generated_sequences)
    print(text_instances)
    return text_instances



def generate_topic_instances(args, prefix_model, label_ids, label_tokens, gpt2, topic_embeds, label_embeds, aug_num, bert_tokenizer=None, mode='aug', vae='ivae', original_sentences=None,dup_score=[]):
    print("generate topic instances =======================================================================================================")
    nltk_tokenizer = TweetTokenizer()
    if mode=='aug':
        args.num_return_sequences = 1
    else:
        args.num_return_sequences = 1
        args.length = 128
    args.device = prefix_model.device
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args.seed)

    # Initialize the model and tokenizer
    if args.tuning_mode == 'prefixtune':

        print('loading from PrefixTuning.', args.prefixModel_name_or_path,)
        if args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            assert False, 'shouldn not init config from scratch. '
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        try:
            args.model_type = args.model_type.lower()
            model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        if args.model_name_or_path:
            print('loading the trained tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        elif args.tokenizer_name:
            print('loading from the init tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

        config._my_arg_tune_mode = args.tuning_mode
        config._my_arg_task_mode = args.task_mode
        config._objective_mode = args.objective_mode
        print(config)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        add_pad = False

        if args.model_name_or_path == 'gpt2-medium':
            if args.task_mode == 'dataless':
                print(args.tuning_mode, 'dataless setting, so no new tokens at all.')
                print('We do not add special tokens to the tokenizer, instead, we just finetune on <|endoftext|>')

                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

            elif add_pad:
                print('extending the size of word embeddings. to include the [PAD] ')
                num_added_tokens = tokenizer.add_special_tokens(
                    {'pad_token': '[PAD]'})
                embedding_layer = model.resize_token_embeddings(len(tokenizer))
            else:
                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        if args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if args.prefixModel_name_or_path is not None:
            model = prefix_model
            config = model.config
        else:
            assert False, "prefixModel_name_or_path is NONE."

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    decode_mode = 'beam'

    text_instances = []
    label_dict = {}

    total_count = 0
    dup_count = 0

    for prompt_idx, label_token in enumerate(label_tokens):
        label_id = label_ids[prompt_idx]

        label_dict[label_id] = label_token

        label_embed_ori = label_embeds[label_id]
        label_embed_list = []
        if isinstance(label_embed_ori, list):
            label_embed_list = label_embed_ori
        else:
            label_embed_list.append(label_embed_ori)

        if isinstance(topic_embeds, dict):
            topic_embeds = topic_embeds[label_id]

        for label_embed in label_embed_list:
            label_embed = label_embed.unsqueeze(0).unsqueeze(0).to(model.device)
            label_token_bpe = tokenizer.encode(
                label_token, add_special_tokens=True, truncation=True, max_length=100,
                is_split_into_words=False
            )
            if config._my_arg_task_mode == 'no_rel':
                label_token_full = [tokenizer.bos_token_id]  # add the bos token.
            else:
                label_token_full = label_token_bpe + [tokenizer.bos_token_id] # add the bos token.
            print(label_token_full)
            requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
                preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

                if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                    tokenizer_kwargs = {"add_space_before_punct_symbol": True}
                else:
                    tokenizer_kwargs = {}

                encoded_prompt = tokenizer.encode(
                    preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
                )
            elif args.task_mode == 'cnndm' or args.task_mode == 'xsum':
                # already processed
                encoded_prompt = torch.LongTensor(label_token_full).unsqueeze(0)
                print(encoded_prompt.shape)
            else:
                prefix = args.prefix if args.prefix else args.padding_text
                print('****************',prefix)
                encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(args.device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            if args.task_mode == 'embMatch' and args.control_dataless != 'yes':
                print(' '.join(sent_cleaned_lst[prompt_idx]))
                emb_match_temp = emb_match[prompt_idx].to(model.device).expand(args.num_return_sequences, -1, -1)
            else:
                emb_match_temp = None

            if args.control_mode == 'yes' and args.control_dataless != 'yes':
                # URGENT, check whether the next line is necessary?
                # control_code = torch.LongTensor(control_codes[prompt_idx]).to(model.device).unsqueeze(0).expand(args.num_return_sequences, -1)
                control_code = None
                pass

            else:
                control_code = None

            if args.tuning_mode == 'prefixtune' or args.tuning_mode == 'bothtune':
                print(config.optim_prefix, optim_prefix_bool)
                print('control code is ', control_code)
                print('prompt idx is ', prompt_idx)
                print('label id is ',label_id)
                print('label tokens are ', label_token)


                batch_index_arr = np.arange(len(topic_embeds))
                if mode=='aug':
                    batch_size = 50
                else:
                    batch_size = 200
                batch_num = int(np.ceil(len(topic_embeds) / float(batch_size)))
                # print(batch_num)
                for batch_id in range(batch_num):
                    # print (batch_id)
                    # print(batch_size)
                    batch_ids = batch_index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
                    current_bsz = len(batch_ids)
                    # print(current_bsz)
                    #print(topic_embeds[0].size())
                    batch_topic_embeds = [topic_embeds[i].unsqueeze(0).expand(args.num_return_sequences, label_embed.size(2)) for i in batch_ids]
                    if original_sentences is not None:
                        batch_original_sentences = [original_sentences[i] for i in batch_ids]
                    # print(len(batch_topic_embeds))
                    batch_topic_embed = torch.cat(batch_topic_embeds, dim=0)
                    # for topic_index, topic_embed in enumerate(topic_embeds):
                    expand_topic_embed = batch_topic_embed.unsqueeze(1)
                    #print(expand_topic_embed.size())

                    expand_label_embed = label_embed.expand(current_bsz*args.num_return_sequences,label_embed.size(1), label_embed.size(2))

                    #print(expand_label_embed.size())

                    prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=current_bsz*args.num_return_sequences, sentences_embed=expand_label_embed,
                                              topic_embed=expand_topic_embed)
                    #prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=topic_embed.size(0), topic_embed=topic_embed)
                    #print(prompt[0].size())
                    #print(prompt[0].unsqueeze(1).size())
                    #print(prompt[0].unsqueeze(1).expand(-1, args.num_return_sequences, -1, -1, -1, -1).size())
                    #print(prompt[0].unsqueeze(1).expand(-1, args.num_return_sequences, -1, -1, -1, -1).reshape(prompt[0].size(0),args.num_return_sequences*current_bsz, prompt[0].size(3),prompt[0].size(4),prompt[0].size(5)).size())
                    #size_0 = prompt[0].size(0)
                    #size_1 = prompt[0].size(1)
                    #size_2 = prompt[0].size(2)
                    #size_3 = prompt[0].size(3)
                    #size_4 = prompt[0].size(4)
                    #prompt = [x.unsqueeze(1).expand(-1, args.num_return_sequences, -1, -1, -1, -1).reshape(size_0,args.num_return_sequences*current_bsz, size_2,size_3,size_4) for x in prompt]
                    expand_input_ids = input_ids.expand(current_bsz, input_ids.size(1))
                    # print(input_ids.size())


                #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                #topic_embed_matrix = torch.stack(topic_embeds).view(len(topic_embeds), -1).squeeze().to(model.device)
                #label_embed_compare = label_embed.squeeze().unsqueeze(0)
                #sim = cos(topic_embed_matrix, label_embed_compare)
                #print(sim.size())
                #values, indices = torch.topk(-sim, 50)
                #min_index_set = set(indices.tolist())
                #for topic_index, topic_embed in enumerate(topic_embeds):
                    #if not topic_index in min_index_set:
                    #    continue
                    #mean = proto_dict[str(label_id)]['mean']
                    #logv = proto_dict[str(label_id)]['cov']
                    #h = torch.from_numpy(np.random.multivariate_normal(mean, logv).astype(np.float32))

                    #print(h.size())
                    #print(label_embed.size())
                    #print(topic_embed.size())

                    #topic_embed = topic_embed.unsqueeze(0).unsqueeze(0).to(model.device)

                    #print(label_embed.size())
                    #print(topic_embed.size())

                    #prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=1, sentences_embed=label_embed, topic_embed=topic_embed)

                    #prompt = [x.expand(-1, args.num_return_sequences , -1, -1, -1) for x in prompt]

                    #print(decode_mode)
                    #print(prompt[0].size())
                    if decode_mode == 'nucleus':
                        output_sequences = gpt2.generate(
                            input_ids=expand_input_ids,
                            emb_match=None,
                            control_code=None,
                            past_key_values=prompt,
                            max_length=args.length + len(encoded_prompt[0]),
                            temperature=args.temperature,
                            top_k=args.k,
                            top_p=0.8,
                            repetition_penalty=args.repetition_penalty,
                            do_sample=True,
                            num_return_sequences=args.num_return_sequences,
                        )
                    elif decode_mode == 'beam':
                        #print (args.repetition_penalty)
                        #print (args.temperature)
                        #print (args.length)
                        #print (args.k)
                        output_sequences = gpt2.generate(
                            input_ids=expand_input_ids,
                            emb_match=None,
                            control_code=None,
                            past_key_values=prompt,
                            max_length=args.length + len(encoded_prompt[0]),
                            min_length=10,
                            temperature=args.temperature,
                            top_k=args.k,
                            top_p=0.9, #top_p=0.5,
                            repetition_penalty=args.repetition_penalty, ##args.repetition_penalty,
                            do_sample=False,
                            num_beams=args.num_return_sequences,
                            bad_words_ids=[[628], [198]] if True else None,
                            num_return_sequences=args.num_return_sequences
                        )
                        # print(output_sequences)

                    elif decode_mode == 'greedy':
                        output_sequences = gpt2.generate(
                            input_ids=expand_input_ids,
                            emb_match=None,
                            control_code=None,
                            past_key_values=prompt,
                            max_length=args.length + len(encoded_prompt[0]),
                            min_length=5,
                            temperature=args.temperature,
                            top_k=args.k,
                            top_p=0.5,
                            repetition_penalty=args.repetition_penalty,
                            do_sample=False,
                            bad_words_ids=[[628], [198]] if True else None,
                            num_return_sequences=1,
                        )


                    # Remove the batch dimension when returning multiple sequences
                    if len(output_sequences.shape) > 2:
                        output_sequences.squeeze_()

                    output_sequences_list = output_sequences.tolist()
                    if mode=='aug':
                        #print("original")
                        #print(output_sequences_list)

                        origin_total_count = len(output_sequences_list)

                        total_count += origin_total_count

                        df = pd.DataFrame(output_sequences_list)
                        df = df.drop_duplicates()
                        output_sequences_list = df.values.tolist()

                        dup_count += origin_total_count - len(output_sequences_list)
                        #print("filter")
                        #print(output_sequences_list)
                        index_arr = np.arange(len(output_sequences_list))
                        np.random.shuffle(index_arr)
                        index_arr_ids = list(index_arr)

                        output_sequences_list = [output_sequences_list[i] for i in index_arr_ids]
                    #print("after")
                    #print(output_sequences_list)
                    generated_sequences = []

                    for generated_sequence_idx, generated_sequence in enumerate(output_sequences_list):
                        #if original_sentences is not None:
                        #    sentence_id = int(generated_sequence_idx/args.num_return_sequences)
                        #    print(batch_original_sentences[sentence_id])
                        # args.stop_token = tokenizer.eos_token
                        #generated_sequence = generated_sequence.tolist()
                        #print(generated_sequence)
                        #if 50257 in generated_sequence:
                        #    continue
                        #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                        #eos_index = generated_sequence.index(50256)
                        #print(generated_sequence)
                        generated_sequence = [token_id for token_id in generated_sequence if token_id != 50257]
                        #print(generated_sequence)
                        # Decode text
                        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                        #print(text)
                        text_output = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                        #print (text_output)
                        copy_text = str(text_output)
                        idx = text_output.find(tokenizer.eos_token)
                        if idx >= 0:
                            #print (id)
                            #print (text_output)
                            text_output = text_output[:idx]
                            #print (text_output)
                        text_output = text_output.strip()

                        if args.task_mode == 'topic' or args.task_mode == 'sentiment':
                            text_output = prompt_text + ' ' + text_output + ' [SPECIAL_END]'

                        if text_output:
                            text_output = " ".join([token for token in nltk_tokenizer.tokenize(text_output)])
                            #print(text_output)
                            generated_sequences.append((label_id, text_output))
                            #break
                        else:
                            print ("ERROR")
                            print (copy_text)
                            #if not mode=='aug':
                            generated_sequences.append((label_id, "ERROR TEXT"))
                    #assert len(generated_sequences) == 1
                    text_instances.extend(generated_sequences)
    print ("====================== length before filter =========================================================")
    print (len(text_instances))
    if mode == 'aug':

        best_text_instances = []

        question_dict = {}

        for label_id, utterance in text_instances:
            if label_id in question_dict:
                question_dict[label_id].append((label_id, utterance))
            else:
                question_dict[label_id] = []
                question_dict[label_id].append((label_id, utterance))
        for label_id, question_list in question_dict.items():
            #if isinstance(label_embeds[0], list):
            #    question_list_index_arr = np.arange(len(question_list))
            #    np.random.shuffle(question_list_index_arr)
            #    question_list_index_arr_ids = list(question_list_index_arr)
            #    best_text_instances.extend([best_text_instances[i] for i in question_list_index_arr_ids[:aug_num]])
            #else:
            best_text_instances.extend(select_best_instance_from_a_list(question_list, label_embeds, model, bert_tokenizer, args, label_dict, prefix_model, return_num=aug_num,vae=vae, mode=mode))

        index_arr = np.arange(len(best_text_instances))
        np.random.shuffle(index_arr)
        index_arr_ids = list(index_arr)

        random_examples = [best_text_instances[i] for i in index_arr_ids]
        dup_score.append(dup_count/total_count)
        return random_examples

    else:
        print ("====================== final length =========================================================")
        print (len(text_instances))
        nested_text_instances = [text_instances[i:i + args.num_return_sequences] for i in range(0, len(text_instances), args.num_return_sequences)]
        best_text_instances = []

        for question_list in nested_text_instances:
            best_text_instances.extend(select_best_instance_from_a_list(question_list, label_embeds, model, bert_tokenizer, args, label_dict,
                                             prefix_model, return_num=1,vae=vae, mode=mode))
        return best_text_instances


def generate_privacy_topic_instances(args, prefix_model,  gpt2, label_embeds, topic_embeds, model_name_or_path):
    print(
        "generate topic instances =======================================================================================================")
    nltk_tokenizer = TweetTokenizer()
    args.num_return_sequences = 5
    args.length = 128
    args.device = prefix_model.device
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.model_name_or_path = model_name_or_path

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args.seed)

    # Initialize the model and tokenizer
    if args.tuning_mode == 'prefixtune':

        print('loading from PrefixTuning.', args.prefixModel_name_or_path, )
        if args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            assert False, 'shouldn not init config from scratch. '
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        try:
            args.model_type = args.model_type.lower()
            model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")
        #path = os.path.abspath(transformers.__file__)

        from transformers import AutoTokenizer
        if args.model_name_or_path:
            print('loading the trained tokenizer')
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        elif args.tokenizer_name:
            print('loading from the init tokenizer')
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)

        if tokenizer.pad_token is None:
            num_added_tokens = tokenizer.add_special_tokens(
                {'pad_token': '[PAD]'})
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.sep_token
            # tokenizer.eos_token_id = tokenizer.sep_token_id
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.cls_token

        #breakpoint()
        config._my_arg_tune_mode = args.tuning_mode
        config._my_arg_task_mode = args.task_mode
        config._objective_mode = args.objective_mode
        print(config)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        add_pad = False

        if args.model_name_or_path == 'gpt2-medium':
            if args.task_mode == 'dataless':
                print(args.tuning_mode, 'dataless setting, so no new tokens at all.')
                print('We do not add special tokens to the tokenizer, instead, we just finetune on <|endoftext|>')

                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

            elif add_pad:
                print('extending the size of word embeddings. to include the [PAD] ')
                num_added_tokens = tokenizer.add_special_tokens(
                    {'pad_token': '[PAD]'})
                embedding_layer = model.resize_token_embeddings(len(tokenizer))
            else:
                print(tokenizer.eos_token_id)
                print(tokenizer.eos_token)
                print(tokenizer.pad_token_id)
                tokenizer.pad_token = tokenizer.eos_token
                print(tokenizer.pad_token, tokenizer.pad_token_id)

        print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

        if args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if args.prefixModel_name_or_path is not None:
            model = prefix_model
            config = model.config
        else:
            assert False, "prefixModel_name_or_path is NONE."

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    decode_mode = 'beam'

    text_instances = {}


    for label_token, topic_embeds in topic_embeds.items():

        label_embed = label_embeds[label_token]

        #label_embed = label_embed.unsqueeze(0).unsqueeze(0).to(model.device)
        label_token_bpe = tokenizer.encode(
            label_token, add_special_tokens=True, truncation=True, max_length=100,
            is_split_into_words=False
        )

        #breakpoint()
        label_token_full = label_token_bpe + [tokenizer.bos_token_id]  # add the bos token.
        print(label_token_full)
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        elif args.task_mode == 'cnndm' or args.task_mode == 'xsum':
            # already processed
            encoded_prompt = torch.LongTensor(label_token_full).unsqueeze(0)
            print(encoded_prompt.shape)
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            print('****************', prefix)
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        if args.task_mode == 'embMatch' and args.control_dataless != 'yes':
            print(' '.join(sent_cleaned_lst[prompt_idx]))
            emb_match_temp = emb_match[prompt_idx].to(model.device).expand(args.num_return_sequences, -1, -1)
        else:
            emb_match_temp = None

        if args.control_mode == 'yes' and args.control_dataless != 'yes':
            # URGENT, check whether the next line is necessary?
            # control_code = torch.LongTensor(control_codes[prompt_idx]).to(model.device).unsqueeze(0).expand(args.num_return_sequences, -1)
            control_code = None
            pass

        else:
            control_code = None

        if args.tuning_mode == 'prefixtune' or args.tuning_mode == 'bothtune':
            print(config.optim_prefix, optim_prefix_bool)
            print('control code is ', control_code)
            print('label tokens are ', label_token)

            batch_index_arr = np.arange(len(topic_embeds))
            batch_size = 40
            batch_num = int(np.ceil(len(topic_embeds) / float(batch_size)))
            # print(batch_num)
            for batch_id in range(batch_num):
                # print (batch_id)
                # print(batch_size)
                batch_ids = batch_index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
                current_bsz = len(batch_ids)
                # print(current_bsz)
                # print(topic_embeds[0].size())
                batch_topic_embeds = [
                    topic_embeds[i].unsqueeze(0).expand(args.num_return_sequences, topic_embeds[i].size(-1)) for i in
                    batch_ids]
                # print(len(batch_topic_embeds))
                batch_topic_embed = torch.cat(batch_topic_embeds, dim=0)
                # for topic_index, topic_embed in enumerate(topic_embeds):
                expand_topic_embed = batch_topic_embed.unsqueeze(1)
                # print(expand_topic_embed.size())

                #print(len(label_embed_list))

                batch_label_embed = [
                    label_embed.unsqueeze(0).expand(args.num_return_sequences, label_embed.size(-1)) for i in
                    batch_ids]

                expand_label_embed = torch.cat(batch_label_embed, dim=0).unsqueeze(1)

                    #label_embed.expand(current_bsz * args.num_return_sequences, label_embed.size(1),
                    #                                    label_embed.size(2))

                # print(expand_label_embed.size())

                prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=current_bsz * args.num_return_sequences,
                                          sentences_embed=expand_label_embed,
                                          topic_embed=expand_topic_embed)
                # prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=topic_embed.size(0), topic_embed=topic_embed)
                # print(prompt[0].size())
                # print(prompt[0].unsqueeze(1).size())
                # print(prompt[0].unsqueeze(1).expand(-1, args.num_return_sequences, -1, -1, -1, -1).size())
                # print(prompt[0].unsqueeze(1).expand(-1, args.num_return_sequences, -1, -1, -1, -1).reshape(prompt[0].size(0),args.num_return_sequences*current_bsz, prompt[0].size(3),prompt[0].size(4),prompt[0].size(5)).size())
                # size_0 = prompt[0].size(0)
                # size_1 = prompt[0].size(1)
                # size_2 = prompt[0].size(2)
                # size_3 = prompt[0].size(3)
                # size_4 = prompt[0].size(4)
                # prompt = [x.unsqueeze(1).expand(-1, args.num_return_sequences, -1, -1, -1, -1).reshape(size_0,args.num_return_sequences*current_bsz, size_2,size_3,size_4) for x in prompt]
                expand_input_ids = input_ids.expand(current_bsz, input_ids.size(1))
                # print(input_ids.size())

                # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                # topic_embed_matrix = torch.stack(topic_embeds).view(len(topic_embeds), -1).squeeze().to(model.device)
                # label_embed_compare = label_embed.squeeze().unsqueeze(0)
                # sim = cos(topic_embed_matrix, label_embed_compare)
                # print(sim.size())
                # values, indices = torch.topk(-sim, 50)
                # min_index_set = set(indices.tolist())
                # for topic_index, topic_embed in enumerate(topic_embeds):
                # if not topic_index in min_index_set:
                #    continue
                # mean = proto_dict[str(label_id)]['mean']
                # logv = proto_dict[str(label_id)]['cov']
                # h = torch.from_numpy(np.random.multivariate_normal(mean, logv).astype(np.float32))

                # print(h.size())
                # print(label_embed.size())
                # print(topic_embed.size())

                # topic_embed = topic_embed.unsqueeze(0).unsqueeze(0).to(model.device)

                # print(label_embed.size())
                # print(topic_embed.size())

                # prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=1, sentences_embed=label_embed, topic_embed=topic_embed)

                # prompt = [x.expand(-1, args.num_return_sequences , -1, -1, -1) for x in prompt]

                #breakpoint()
                # print(decode_mode)
                # print(prompt[0].size())
                #expand_input_ids = expand_input_ids.cpu()
                #prompt = (prompt[0].cpu(),prompt[1].cpu())
                #prompt[1] = prompt[1].cpu()
                #gpt2 = gpt2.cpu()
                #print(expand_input_ids)
                if decode_mode == 'nucleus':
                    output_sequences = gpt2.generate(
                        input_ids=expand_input_ids,
                        emb_match=None,
                        control_code=None,
                        past_key_values=prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=0.8,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=True,
                        num_return_sequences=args.num_return_sequences,
                    )
                elif decode_mode == 'beam':
                    # print (args.repetition_penalty)
                    # print (args.temperature)
                    # print (args.length)
                    # print (args.k)
                    output_sequences = gpt2.generate(
                        input_ids=expand_input_ids,
                        emb_match=None,
                        control_code=None,
                        past_key_values=prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        min_length=5,
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=0.8,  # top_p=0.5,
                        repetition_penalty=args.repetition_penalty,  ##args.repetition_penalty,
                        do_sample=False,
                        num_beams=args.num_return_sequences,
                        bad_words_ids=[[628], [198]] if True else None,
                        num_return_sequences=args.num_return_sequences,
                        no_repeat_ngram_size=2
                    )
                    # print(output_sequences)

                elif decode_mode == 'greedy':
                    output_sequences = gpt2.generate(
                        input_ids=expand_input_ids,
                        emb_match=None,
                        control_code=None,
                        past_key_values=prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        min_length=5,
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=0.5,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=False,
                        bad_words_ids=[[628], [198]] if True else None,
                        num_return_sequences=1,
                    )

                # Remove the batch dimension when returning multiple sequences
                if len(output_sequences.shape) > 2:
                    output_sequences.squeeze_()

                output_sequences_list = output_sequences.tolist()
                # print("after")
                # print(output_sequences_list)
                generated_sequences = []

                for generated_sequence_idx, generated_sequence in enumerate(output_sequences_list):
                    # if original_sentences is not None:
                    #    sentence_id = int(generated_sequence_idx/args.num_return_sequences)
                    #    print(batch_original_sentences[sentence_id])
                    # args.stop_token = tokenizer.eos_token
                    # generated_sequence = generated_sequence.tolist()
                    # print(generated_sequence)
                    # if 50257 in generated_sequence:
                    #    continue
                    # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                    # eos_index = generated_sequence.index(50256)
                    # print(generated_sequence)
                    generated_sequence = [token_id for token_id in generated_sequence if token_id != 50257]
                    # print(generated_sequence)
                    # Decode text
                    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                    # print(text)
                    text_output = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                    # print (text_output)
                    copy_text = str(text_output)
                    idx = text_output.find(tokenizer.eos_token)
                    if idx >= 0:
                        # print (id)
                        # print (text_output)
                        text_output = text_output[:idx]
                        # print (text_output)
                    text_output = text_output.strip()

                    if args.task_mode == 'topic' or args.task_mode == 'sentiment':
                        text_output = prompt_text + ' ' + text_output + ' [SPECIAL_END]'

                    if text_output:
                        text_output = " ".join([token for token in nltk_tokenizer.tokenize(text_output)])
                        # print(text_output)
                        generated_sequences.append(text_output)
                        # break
                    else:
                        generated_sequences.append("ERROR TEXT")
                # assert len(generated_sequences) == 1
                if label_token in text_instances:
                    text_instances[label_token].extend(generated_sequences)
                else:
                    text_instances[label_token] = []
                    text_instances[label_token].extend(generated_sequences)
    return text_instances


def select_best_instance_from_a_list(text_instances, label_embeds, model, bert_tokenizer, args, label_dict, prefix_model, return_num, vae, mode):
    score_list = []
    for instance_index, (label_id, instance) in enumerate(text_instances):
        label_embed = label_embeds[label_id]
        if isinstance(label_embed, list):
            label_embed = torch.stack(label_embeds[label_id]).mean(0)
        label_embed = label_embed.to(model.device).squeeze().unsqueeze(0)

        instance_bpe = bert_tokenizer(
            instance, return_tensors='pt', padding=True, truncation=True, max_length=128
        )['input_ids']

        instance_tensor = instance_bpe.to(args.device)

        label_instance_bpe = bert_tokenizer(
            label_dict[label_id], return_tensors='pt', padding=True, truncation=True, max_length=128
        )['input_ids']

        label_instance_tensor = label_instance_bpe.to(args.device)

        # print(instance_tensor)
        instance_embed = prefix_model.sentence_encoder(input_ids=instance_tensor, src_input_ids=label_instance_tensor,vae=vae).squeeze().unsqueeze(0)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim_score = cos(label_embed, instance_embed).item()
        if not mode == 'aug':
            #print (instance[-1])
            if instance[-1] in ['!', '?', '.']:
                sim_score = sim_score*1.1
            sent_list = nltk.sent_tokenize(instance)
            if len(sent_list[-1].split(' ')) < 4:
                #print (sent_list[-1])
                sim_score = sim_score*0.9
                
        score_list.append(sim_score)
    sorted_index = [i[0] for i in sorted(enumerate(score_list), key=lambda x: x[1], reverse=True)]

    best_text_instances = [text_instances[indx] for indx in sorted_index[:return_num]]
    print(best_text_instances)
    print(len(best_text_instances))
    return best_text_instances





