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
import copy
import numpy as np
import torch
import json

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
import sys, os

from train_control import PrefixTuning, PrefixEmbTuning

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
    src_file = 'understanding_test_data/test.source'
    tgt_file = 'understanding_test_data/test.target'

    file_dict = {}

    label_id_tokens = {}
    label_ids = []
    with open(src_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and not line.isspace():
                label_id = line.split('\t')[0]
                label_tokens = line.split('\t')[1]
                label_ids.append(label_id)
                label_id_tokens[int(label_id)] = label_tokens

    src_bpe_list = []
    for i in range(len(label_id_tokens.items())):
        src = label_id_tokens[i]
        print(src)
        src_bpe = tokenizer.encode(
            src, add_special_tokens=False, truncation=True, max_length=max_source_length,
            is_split_into_words=False
        )
        full = src_bpe # add the bos token.
        src_bpe_list.append(full)
    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    #print(tgt_file, len(tgt_lines), '\n', src_file, len(src_lines))
    src_full_list = []
    for tgt in tgt_lines:
        src_full = []
        tgt_bpe = tokenizer.encode(
            tgt, add_special_tokens=False, truncation=True, max_length=max_source_length,
            is_split_into_words=False
        )
        for src_bpe in src_bpe_list:
            src_full.append(tgt_bpe + [tokenizer.bos_token_id] + src_bpe + [tokenizer.eos_token_id])
        src_full_list.append(src_full)

    #print(file_dict)
    return src_full_list, label_ids, tgt_lines

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained tokenizer or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--prefixModel_name_or_path",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained PrefixTuning Model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--task_mode", type=str, default="embMatch")
    parser.add_argument("--control_mode", type=str, default="yes")
    parser.add_argument("--prefix_mode", type=str, default="activation")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--gen_dir", type=str, default="e2e_results_conv")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--tuning_mode", type=str, default="finetune", help="prefixtune or finetune")
    parser.add_argument("--objective_mode", type=int, default=0)
    parser.add_argument("--format_mode", type=str, default="peek", help="peek, cat, nopeek, or infix")
    parser.add_argument("--optim_prefix", type=str, default="no", help="optim_prefix")
    parser.add_argument("--preseqlen", type=int, default=5, help="preseqlen")

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--control_dataless", type=str, default="no", help="control dataless mode")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

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
        model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
        model.to(args.device)

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

        gpt2 = model

        if args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if args.prefixModel_name_or_path is not None:

            config = AutoConfig.from_pretrained(args.prefixModel_name_or_path, cache_dir=args.cache_dir )
            print(config)

            if args.prefix_mode == 'embedding':
                model = PrefixEmbTuning.from_pretrained(
                    args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in args.prefixModel_name_or_path, ),
                    config=config,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=args.preseqlen,
                    use_infix=(args.format_mode == 'infix')
                )

            elif args.prefix_mode == 'activation':

                model = PrefixTuning.from_pretrained(
                    args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in args.prefixModel_name_or_path, ),
                    config=config,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=args.preseqlen,
                    use_infix=(args.format_mode == 'infix')
                )

            model.to(args.device)
        else:
            assert False, "prefixModel_name_or_path is NONE."

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)
    sentence_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.cache/huggingface'))
    sentence_encoder = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.cache/huggingface')).to(
        model.device)

    if args.task_mode == 'xsum':
        src_full_list, label_ids, tgt_lines = read_sum_files("test_path", tokenizer, 100, 100)

        #prompt_text_lst = list(prompt_text_dict.keys())
        if args.prefixModel_name_or_path is not None:
            temp = os.path.basename(args.prefixModel_name_or_path)
        else:
            temp = os.path.basename(args.model_name_or_path)
        proto_dict = torch.load(os.path.join(args.prefixModel_name_or_path, "proto_type_dict.bin"))
        # print(prompt_text_dict)
        split_file = 'test'  # test
        decode_mode = 'beam'
        curr_dir = os.path.join('',
                                args.gen_dir,
                                '{}_{}_{}'.format(temp, split_file, decode_mode))

        print(curr_dir)
        gold_dir = os.path.join('',
                                args.gen_dir,
                                '{}_{}_{}'.format(temp, split_file, 'gold'))
        #print(gold_dir)
        #write_e2e_corr(prompt_text_lst, prompt_text_dict, gold_dir)

        out_handle = open(curr_dir, 'w')

    true_count = 0
    for instance_idx, src_full in enumerate(src_full_list):
        #tgt_line = tgt_lines[instance_idx]
        #dis_tgt = sentence_tokenizer(tgt_line, add_special_tokens=True, truncation=True, max_length=100,
        #                        is_split_into_words=False, return_tensors="pt")['input_ids']
        #dis_tgt = dis_tgt.to(model.device)
        # print (type(sentence_encoder))
        # print (sentence_encoder(dis_tgt))
        #sentences_embed = sentence_encoder(dis_tgt, return_dict=True).pooler_output.squeeze()

        loss_values = []
        for label_indx, prompt_text in enumerate(src_full):
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

                sep_idx = prompt_text.index(tokenizer.bos_token_id) + 1

                label = copy.deepcopy(prompt_text)
                label[:sep_idx] = [-100] * sep_idx
                print(label)
                label = torch.LongTensor(label).unsqueeze(0)
                #print("======================")
                print(prompt_text)
                encoded_prompt = torch.LongTensor(prompt_text).unsqueeze(0)
                #print(encoded_prompt.size())
                #print(encoded_prompt.shape)
            else:
                prefix = args.prefix if args.prefix else args.padding_text
                print('****************',prefix)
                encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
            label = label.to(args.device)
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
                if args.task_mode == 'webnlg' or args.task_mode == 'triples':
                    src = prompt_text_lst[prompt_idx].split()[:-1]
                    src = ' '.join(src)
                    cat = prompt_rela_lst[prompt_idx]
                    print(cat)
                    src_cat = tokenizer(cat, add_special_tokens=True, truncation=True, is_split_into_words=True)['input_ids']
                    src = tokenizer(src, add_special_tokens=True, truncation=True, is_split_into_words=False)['input_ids']

                    mode = None
                    if 'cat2' in args.prefixModel_name_or_path or 'cat' in args.prefixModel_name_or_path:
                        mode = 'cat'
                    elif 'nopeek' in args.prefixModel_name_or_path or 'nop' in args.prefixModel_name_or_path:
                        mode = 'nopeek'
                    elif 'peek' in args.prefixModel_name_or_path or 'pee' in args.prefixModel_name_or_path:
                        mode = 'peek'
                    elif 'tune_y_' in args.prefixModel_name_or_path or config.optim_prefix:
                        mode = 'instruction_based'
                        # assert False, "prefixtune20 shouldn't be processed here."
                    else:
                        if args.format_mode == 'infix':
                            mode = 'infix'
                        else:
                            assert False, "Given that it's in prefix tuning mode, need to specify a valid prefix mode, " \
                                          "(cat, nopeek, peek)"

                    print(mode)

                    if mode == 'cat':
                        cc = src_cat
                    elif mode == 'peek' or mode == 'nopeek':
                        cc = src
                    elif mode == 'infix':
                        cc = src

                    if mode == 'nopeek' or mode == 'infix':
                        input_pp = tokenizer.bos_token
                        encoded_prompt = tokenizer(input_pp, add_special_tokens=True, truncation=True, return_tensors="pt", is_split_into_words=False)['input_ids'].to(model.device)
                        input_ids = encoded_prompt

                    if mode in ['cat', 'peek', 'nopeek', 'infix']:
                        control_code = torch.LongTensor(cc).to(model.device).unsqueeze(0)
                    elif mode == 'instruction_based':
                        control_code = None
                    else:
                        assert False, "invalid mode type."

                    if config.optim_prefix:
                        control_code = None


                print(config.optim_prefix, optim_prefix_bool)
                print('control code is ', control_code)
                print('current label idx is ', label_indx)
                #label_id = src_ids[label_indx]
                #print('label id is ',label_id)

                mean = proto_dict[str(label_indx)]['mean']
                #logv = proto_dict[str(label_id)]['cov']
                #h = torch.from_numpy(np.random.multivariate_normal(mean, logv).astype(np.float32))
                #print(h.size())
                proto_type =  torch.from_numpy(mean).unsqueeze(0).unsqueeze(0).to(model.device)
                #proto_type =  sentences_embed.unsqueeze(0).unsqueeze(0).to(model.device)
                prompt = model.get_prompt(control_code, gpt2=gpt2, bsz=1, sentences_embed=proto_type)

                #prompt = [x.expand(-1, args.num_return_sequences , -1, -1, -1) for x in prompt]

                loss = gpt2(
                    input_ids=input_ids,
                    past_key_values=prompt,
                    labels = label,
                    return_dict=True,
                ).loss
                loss_value = loss.item()
                loss_values.append(loss_value)
        print(loss_values)
        predict_label_id = loss_values.index(min(loss_values))
        gold_label_id = label_ids[instance_idx]
        print(predict_label_id)
        print(gold_label_id)
        if int(predict_label_id) == int(gold_label_id):
            true_count+=1
    acc = true_count/len(src_full_list)
    print("acc is : ", acc)

    out_handle.close()

if __name__ == "__main__":
    main()

