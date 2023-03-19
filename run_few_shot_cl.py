import argparse
import pickle
from collections import Counter

import nltk
import numpy as np
import spacy as spacy
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as st
import sys
import random
import time
import json
import os

import sacrebleu

import wmd
from tqdm import tqdm, trange
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from sklearn.cluster import KMeans




sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(sys_path)
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForMaskedLM, MarianTokenizer, \
    MarianMTModel, MarianConfig, GPT2Config, T5ForConditionalGeneration, T5Config, T5Tokenizer
#sys.path.append(os.path.join(sys_path, 'autoencoders'))
#sys.path.append(os.path.join(sys_path, 'autoencoders/generate_samples.py'))
#sys.path.append(os.path.join(sys_path, 'autoencoders/utils.py'))
#sys.path.append(os.path.join(sys_path, 'autoencoders/noise.py'))
#print(sys.path)
print("=========================")
print(sys.path)
from autoencoders.adversarial import CondAdv, Adversarial
from PrefixTuning.transformers.examples.control.run_generation_clean import load_prefix_model, \
    generate_topic_instances, generate_optimus_instances
from PrefixTuning.transformers.examples.control.run_language_modeling_clean import train_prefix, initilize_gpt2, \
    ModelArguments, DataTrainingArguments, TrainingArguments

from autoencoders.generate_samples import train_lambda, train_few_shot, train_adv, train_casual_lens_adv
from autoencoders.utils import sample_k

import lifelong
from lifelong.model.encoder import lstm_encoder
from lifelong.model.module import proto_softmax_layer
from lifelong.data.sampler import data_sampler, topic_data_sampler
from lifelong.utils import set_seed
from lifelong.utils import outputer
from data_loader import get_data_loader
# -------------------------------------------------
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--few_shot_num", type=int, default=0,
                    help="few shot number")

parser.add_argument("--topic_num", type=int, default=800,
                    help="topic number")

parser.add_argument("--disentangle_loss", type=int, default=0,
                    help="disen loss type")
parser.add_argument("--data_type", choices=['tarced', 'caption', 'empathetic', 'goemotion','fewrel'], default="tarced",
                    help="data we use")
parser.add_argument("--prefix_model_name", type=str, default="tarced_model",
                    help="prefix model name")
parser.add_argument("--vae", type=str, default="ivae",
                    help="vae names")

parser.add_argument("--prior", type=str, default="prior",
                    help="use prior or posterior")

parser.add_argument("--data_aug_method",
                    choices=['eda', 'few_shot', 'prefix', 'back_translate', 'none', 'cbert', 'lambda', 'optimus', 'casual_lens'],
                    default='prefix',
                    help="The data aug method")

parser.add_argument("--saved_model_prefix_dir", type=str, default="prefix directory",
                    help="prefix dir")

params = parser.parse_args()

print("*******************************************")
print("This process has the PID", os.getpid())


if params.data_type == 'tarced':
    if params.data_aug_method == 'optimus':
        from config.config_optimus import CONFIG_tarced as config
    elif params.data_aug_method == 'casual_lens':
        from config.config_casual_lens import CONFIG_tarced as config
    else:
        from config.config import CONFIG_tarced as config
elif params.data_type == 'caption':
    if params.data_aug_method == 'optimus':
        from config.config_optimus import CONFIG_caption as config
    elif params.data_aug_method == 'casual_lens':
        from config.config_casual_lens import CONFIG_caption as config
    else:
        from config.config import CONFIG_caption as config
elif params.data_type == 'empathetic':
    if params.data_aug_method == 'optimus':
        from config.config_optimus import CONFIG_empathetic as config
    elif params.data_aug_method == 'casual_lens':
        from config.config_casual_lens import CONFIG_empathetic as config
    else:
        from config.config import CONFIG_empathetic as config
elif params.data_type == 'goemotion':
    if params.data_aug_method == 'optimus':
        from config.config_optimus import CONFIG_goemotion as config
    else:
        from config.config import CONFIG_goemotion as config
elif params.data_type == 'fewrel':
    if params.data_aug_method == 'optimus':
        from config.config_optimus import CONFIG_FEWREL as config
    elif params.data_aug_method == 'casual_lens':
        from config.config_casual_lens import CONFIG_FEWREL as config
    else:
        from config.config import CONFIG_FEWREL as config

config['aug'] = params.data_aug_method

config['few_shot_num'] = params.few_shot_num
config['prefix_model_name'] = params.prefix_model_name
config['disentangle_loss'] = params.disentangle_loss
config['vae'] = params.vae
config['topic_num'] = params.topic_num
#os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_id']
emar_prefix = os.path.dirname(os.path.abspath(__file__))

# f = open(emar_prefix + "config/config.json", "r")
# config = json.loads(f.read())

# print (torch.cuda.is_available())
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_CACHE'] = os.path.join(sys_path, '.cache/huggingface')

print(os.environ['TRANSFORMERS_CACHE'])

dir_prefix = config['dir_prefix']

logging.basicConfig(filename=os.path.join(emar_prefix, 'log_fewrel/log_{0}_{1}_{2}.log'.format(config['aug'], config['data_type'], config['few_shot_num'])),
                    level=logging.ERROR)
log = logging.getLogger(__name__)
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
log.info(config)


# f.close()


def evaluate_model(config, model, test_set, num_class):
    # print ("====================evaluate model===================", file=sys.stderr)
    # print(model)
    # print(model.sentence_encoder)
    model.eval()
    data_loader = get_data_loader(config, test_set, False, False)
    num_correct = 0
    total = 0.0
    for step, (labels, neg_labels, sentences, lengths) in enumerate(data_loader):
        # print(len(labels), file=sys.stderr)
        logits, rep = model(sentences, lengths)
        distances = model.get_mem_feature(rep)
        logits = logits
        short_logits = distances
        for index, logit in enumerate(logits):
            score = short_logits[index]  # logits[index] + short_logits[index] + long_logits[index]

            #print(type(score))
            #print(type(neg_labels))

            total += 1.0
            golden_score = score[labels[index]]

            max_neg_score = -2147483647.0
            for i in neg_labels[index]:  # range(num_class):
                if (i != labels[index]) and (score[i] > max_neg_score):
                    max_neg_score = score[i]
            if golden_score > max_neg_score:
                num_correct += 1
    # print ("====================ending evaluate model===================", file=sys.stderr)
    return num_correct / total


def evaluate_model_p5(config, model, test_set, num_class):
    # print ("====================evaluate model===================", file=sys.stderr)
    # print(model)
    # print(model.sentence_encoder)
    model.eval()
    data_loader = get_data_loader(config, test_set, False, False)
    num_correct = 0
    candidate_num_correct = 0
    pfive_correct = 0
    pten_correct = 0
    total = 0.0
    for step, (labels, two_neg_labels, sentences, lengths) in enumerate(data_loader):
        # print(len(labels), file=sys.stderr)
        logits, rep = model(sentences, lengths)
        distances = model.get_mem_feature(rep)
        logits = logits
        short_logits = distances
        for index, logit in enumerate(logits):
            score = short_logits[index]  # logits[index] + short_logits[index] + long_logits[index]
            #======================================================================================

            candidate_neg_labels = two_neg_labels[1]
            candidate_golden_score = score[labels[index]]

            candidate_max_neg_score = -2147483647.0
            for i in candidate_neg_labels[index]:  # range(num_class):
                if (i != labels[index]) and (score[i] > candidate_max_neg_score):
                    candidate_max_neg_score = score[i]
            if candidate_golden_score > candidate_max_neg_score:
                candidate_num_correct += 1



            #print(type(score))
            #print(type(neg_labels))
            # ======================================================================================
            #print(type(score))
            #print(type(neg_labels))
            neg_labels = two_neg_labels[0]
            total += 1.0
            #print(labels[index])
            if labels[index] in neg_labels[index]:
                neg_labels[index] = neg_labels[index][neg_labels[index]!=labels[index]]
            golden_score = score[labels[index]]

            neg_scores = score[neg_labels[index]]

            #print(neg_labels[index].shape)

            #print(len(neg_labels[index]))
            #print(score[neg_labels[index]].shape)

            sorted_neg_scores = sorted(neg_scores, reverse=True)

            max_neg_score = sorted_neg_scores[0]

            fiveth_max_neg_score = sorted_neg_scores[4]

            tenth_max_neg_score = sorted_neg_scores[10]

            if golden_score > max_neg_score:
                num_correct += 1

            if golden_score > fiveth_max_neg_score:
                pfive_correct +=1

            if golden_score > tenth_max_neg_score:
                pten_correct +=1
    # print ("====================ending evaluate model===================", file=sys.stderr)
    return num_correct / total, pfive_correct/total, pten_correct/total, candidate_num_correct/total



def prune_examples(config, model, generated_test_set, raw_questions, prune_number):
    # print ("====================evaluate model===================", file=sys.stderr)
    # print(model)
    # print(model.sentence_encoder)
    model.eval()
    data_loader = get_data_loader(config, generated_test_set, False, False)

    count = 0
    pruned_examples = []
    pruned_questions = []
    for step, (labels, neg_labels, sentences, lengths) in enumerate(data_loader):
        # print(len(labels), file=sys.stderr)
        logits, rep = model(sentences, lengths)
        distances = model.get_mem_feature(rep)
        logits = logits
        short_logits = distances
        for index, logit in enumerate(logits):
            score = short_logits[index]  # logits[index] + short_logits[index] + long_logits[index]
            golden_score = score[labels[index]]
            pruned_examples.append((golden_score, generated_test_set[count]))
            pruned_questions.append((golden_score, raw_questions[count]))
            count += 1

    sorted_examples = sorted(pruned_examples, key=lambda elem: elem[0],reverse=True)
    sorted_questions = sorted(pruned_questions, key=lambda elem: elem[0],reverse=True)

    pruned_new_set = sorted_examples[:prune_number]

    pruned_new_questions = sorted_questions[:prune_number]

    return [e for index, e in pruned_new_set], [e for index, e in pruned_new_questions]



def get_memory(config, model, proto_set):
    memset = []
    resset = []
    rangeset = [0]
    for i in proto_set:
        memset += i
        rangeset.append(rangeset[-1] + len(i))
    data_loader = get_data_loader(config, memset, False, False)
    features = []
    #print(model)
    for step, (labels, neg_labels, sentences, lengths) in enumerate(tqdm(data_loader)):
        #print("load features")
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
        #print("add features")
    #print("out loop")
    features = np.concatenate(features)
    protos = []
    # print("proto_instaces:%d" % len(features))
    for i in range(len(proto_set)):
        protos.append(torch.tensor(features[rangeset[i]:rangeset[i + 1], :].mean(0, keepdims=True)))
    protos = torch.cat(protos, 0)
    return protos


def select_whole_data(mem_set, proto_set, sample_set, raw_mem_set, raw_train_set):
    for index, instance in enumerate(sample_set):
        raw_instance = raw_train_set[index]
        raw_mem_set.append(raw_instance)
        mem_set.append(instance)
        proto_set[instance[0]].append(instance)

    return mem_set, raw_mem_set


# Use K-Means to select what samples to save, similar to at_least = 0
def select_data(mem_set, proto_set, config, model, sample_set, num_sel_data, raw_mem_set, raw_train_set):
    data_loader = get_data_loader(config, sample_set, False, False)
    features = []
    labels = []
    for step, (labels, neg_labels, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)
    num_clusters = min(num_sel_data, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:, i])
        instance = sample_set[sel_index]
        mem_set.append(instance)
        raw_instance = raw_train_set[sel_index]
        raw_mem_set.append(raw_instance)

        proto_set[instance[0]].append(instance)
    return mem_set, raw_mem_set


# Use K-Means to select what samples to save
def select_data_twice(mem_set, proto_set, config, model, sample_set, num_sel_data, at_least=3):
    data_loader = get_data_loader(config, sample_set, False, False)
    features = []
    for step, (_, _, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)
    num_clusters = min(num_sel_data, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)
    rel_info = {}
    rel_alloc = {}
    for index, instance in enumerate(sample_set):
        if not instance[0] in rel_info:
            rel_info[instance[0]] = []
            rel_alloc[instance[0]] = 0
        rel_info[instance[0]].append(index)
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:, i])
        instance = sample_set[sel_index]
        rel_alloc[instance[0]] += 1
    rel_alloc = [(i, rel_alloc[i]) for i in rel_alloc]
    at_least = min(at_least, num_sel_data // len(rel_alloc))
    while True:
        rel_alloc = sorted(rel_alloc, key=lambda num: num[1], reverse=True)
        if rel_alloc[-1][1] >= at_least:
            break
        index = 0
        while rel_alloc[-1][1] < at_least:
            if rel_alloc[index][1] <= at_least:
                index = 0
            rel_alloc[-1][1] += 1
            rel_alloc[index][1] -= 1
            index += 1
    print(rel_alloc)
    for i in rel_alloc:
        label = i[0]
        num = i[1]
        tmp_feature = features[rel_info[label]]
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(tmp_feature)

    mem_set.append(instance)
    proto_set[instance[0]].append(instance)
    return mem_set


def train_simple_model(config, model, train_set, epochs):
    data_loader = get_data_loader(config, train_set)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, neg_labels, sentences, lengths) in enumerate(tqdm(data_loader, disable=True)):
            model.zero_grad()
            # print(sentences.size())
            # print(labels.size())
            logits, _ = model(sentences, lengths)
            labels = labels.to(config['device'])
            loss = criterion(logits, labels)
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
        # print("The average loss of current epoch: ", np.array(losses).mean())
    return model


def train_model(config, model, mem_set, epochs, current_proto):
    data_loader = get_data_loader(config, mem_set, batch_size=5)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        # current_proto = get_memory(config, model, proto_memory)
        model.set_memorized_prototypes(current_proto)
        losses = []
        for step, (labels, neg_labels, sentences, lengths) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            logits, rep = model(sentences, lengths)
            logits_proto = model.mem_forward(rep)
            labels = labels.to(config['device'])
            loss = (criterion(logits_proto, labels))
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
    return model


def interval(data):
    """
    data: 1-dim np array
    """
    interv = st.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=st.sem(data))
    mean = np.mean(data)
    interv = interv - mean
    return mean, interv


def merge(original_sentence, unigram, k=None):
    top_k = []
    mask_idx = []
    result = []
    original_sentence = original_sentence.split(' ')
    if k == None:
        k = len(original_sentence) // 2
    for i in range(k):
        top_k.append(unigram[i][0].split(' '))
        for j in range(len(top_k[i])):
            if top_k[i][j] == '<mask>':
                mask_idx.append(j)
    for i in range(len(original_sentence)):
        if i not in mask_idx:
            result.append(original_sentence[i])
        else:
            if len(result) == 0 or result[-1] != "<mask>":
                result.append("<mask>")
    return ' '.join(result)


def convert_text_to_list(classifier_tok, sample, max_length):
    tokens = classifier_tok.tokenize(sample)
    length = min(len(tokens), max_length)
    tokens = classifier_tok.convert_tokens_to_ids(tokens, unk_id=classifier_tok.vocab['[UNK]'])

    if (len(tokens) > max_length):
        tokens = tokens[:max_length]
    return tokens, length


def similar_score(classifier_tok, generated_sentences, model, label, criterion, max_length, return_seq=10):
    sentences = []
    lengths = []
    for sample in generated_sentences:
        tokens, length = convert_text_to_list(classifier_tok, sample, max_length)

        sentences.append(torch.LongTensor(tokens))
        lengths.append(torch.LongTensor(length))

    labels = torch.LongTensor(len(sentences)).fill_(label).to(config['device'])

    logits, _ = model(sentences, lengths)
    labels = labels.to(config['device'])
    loss = criterion(logits, labels)
    loss = loss.view(-1, return_seq)
    return loss.mean(dim=-1)


def mask_sentence(sentence, num_of_mask, sampling=False, sampling_time=100):
    sentence = sentence.split(' ')
    sentence_len = len(sentence)
    masked_sentences = []

    if not sampling:
        for mask_start in range(sentence_len - num_of_mask + 1):
            masked_sentence = [word for word in sentence]
            for masked_position in range(mask_start, mask_start + num_of_mask):
                masked_sentence[masked_position] = '<mask>'
            masked_sentence = ' '.join(masked_sentence)
            masked_sentences.append(masked_sentence)

    else:
        for _ in range(sampling_time):
            random_mask = random.sample(range(sentence_len), num_of_mask)
            masked_sentence = []
            for i in range(sentence_len):
                if i not in random_mask:
                    masked_sentence.append(sentence[i])
                else:
                    masked_sentence.append('<mask>')
            masked_sentences.append(' '.join(masked_sentence))
    return masked_sentences


def merge_sentence_masks(sentence):
    merged_sentence = []
    flag = True
    for word in sentence.split(' '):
        if word == '<mask>':
            if flag == True:
                merged_sentence.append('<mask>')
                flag = False
        else:
            merged_sentence.append(word)
            flag = True
    return ' '.join(merged_sentence)


def slot_filling(tok, sentence, pretrained_model, num_beams=1, return_seq=10, do_sample=False):
    splitted_sentence = [sent.split(' ') for sent in sentence]
    min_length = min(map(len, splitted_sentence))
    max_length = max(map(len, splitted_sentence))
    batch = tok(sentence, return_tensors='pt', padding=True, truncation=True)

    # print(batch['input_ids'].size(-1))
    generated_ids = pretrained_model.generate(batch['input_ids'].to(config['device']), min_length=int(min_length),
                                              max_length=int(max_length * 2), num_beams=num_beams,
                                              early_stopping=True, do_sample=do_sample, num_return_sequences=return_seq)
    return tok.batch_decode(generated_ids, skip_special_tokens=True)


def generate_mask(gram, original_sentences, labels, classifier_tok, pretrained_tok, classifier_model, criterion,
                  pretrained_model, max_length, sampling=False, return_seq=1):
    masked_sentences = []
    results = []

    for original_sentence in original_sentences:
        if not sampling:
            masked_sentences.append(mask_sentence(original_sentence, gram))
        else:
            gram = len(original_sentence.split(' ')) // 3
            masked_sentences.append(mask_sentence(original_sentence, num_of_mask=gram, sampling=True))
    for i in tqdm(range(len(original_sentences))):

        merged_masked_sentences = [merge_sentence_masks(sentence) for sentence in masked_sentences[i]]
        # for sent_idx, sentence in enumerate(masked_sentences[i]):
        filling_result = slot_filling(pretrained_tok, merged_masked_sentences, pretrained_model, num_beams=1,
                                      return_seq=return_seq)
        scores = similar_score(classifier_tok, filling_result, classifier_model, labels[i], criterion, max_length,
                               return_seq=return_seq)
        if sampling == True:
            result = [(masked_sentence, scores[sent_idx].item()) for sent_idx, masked_sentence in
                      enumerate(merged_masked_sentences)]
        else:
            result = [(masked_sentence, scores[sent_idx].item()) for sent_idx, masked_sentence in
                      enumerate(masked_sentences[i])]
        # result.append((masked_sentences,score))
        result.sort(key=lambda x: x[1])
        results.append(result)
    return results


def generate_templates(gram, original_sentences, labels, classifier_tok, pretrained_tok, classifier_model, criterion,
                       pretrained_model, max_length, sampling=True):
    n_gram_masks = generate_mask(gram, original_sentences, labels, classifier_tok, pretrained_tok, classifier_model,
                                 criterion, pretrained_model, max_length, sampling=sampling)
    if sampling == True:
        masked_sentences = []
        for mask_idx, original_sentence in enumerate(original_sentences):
            if n_gram_masks[mask_idx][0][1] < 1:
                masked_sentences.append(n_gram_masks[mask_idx][0][0])
                # print(n_gram_masks[mask_idx][0][0])
            else:
                masked_sentences.append(original_sentence)
    else:
        masked_sentences = []
        for mask_idx, original_sentence in enumerate(original_sentences):
            masked_sentences.append(merge(original_sentence, n_gram_masks[mask_idx]))
    return masked_sentences


def aug_memory(mem_data, templates, pre_trained_tok, pretrained_model, classifier_tok, max_length, proto_set,
               raw_questions, augmenter, option):
    # assert len(mem_data) == len(raw_questions)
    copied_mem_data = [[label, list(neg_label), list(token_list), length] for label, neg_label, token_list, length in
                       mem_data]
    copied_proto_set = [[] for i in range(len(proto_set))]

    if not option == 'prompt':
        templates = raw_questions

    for temp_id, template in enumerate(tqdm(templates)):
        rand_num = random.random()
        if rand_num < 0.5:
            if option == 'prompt':
                filling_result = \
                    slot_filling(pre_trained_tok, [template], pretrained_model, num_beams=1, return_seq=1,
                                 do_sample=True)[
                        0]
            elif option == 'glove_insert' or option == 'gpt2_sentence_insert' or option == 'back_translate':
                filling_result = augmenter.augment(raw_questions[temp_id])
            token_list, length = convert_text_to_list(classifier_tok, filling_result, max_length)
            copied_mem_data[temp_id][2] = token_list
            copied_mem_data[temp_id][3] = length
    for i in range(len(copied_proto_set)):
        copied_proto_set[i] = [proto_set[i][0]]
    for instance in copied_mem_data:
        copied_proto_set[instance[0]].append(instance)
    return copied_mem_data, copied_proto_set

def random_sample_list(sample_list):
    random.shuffle(sample_list)
    return sample_list[0]

def aug_memory_nlaug(mem_data, raw_questions, max_length, classifier_tok, augmenter, shot, aug_num, aug_method):
    # assert len(mem_data) == len(raw_questions)
    #copied_mem_data = [[label, list(neg_label), list(token_list), length] for label, neg_label, token_list, length in
    #                   mem_data]

    #copied_proto_set = [[] for i in range(len(proto_set))]

    tokenizer = nltk.TweetTokenizer()
    #source_text = " ".join(tokenizer.tokenize(source_text))
    #target_text = " ".join(tokenizer.tokenize(target_text))

    aug_data = []
    n = int(aug_num/shot)
    raw_question_data = []
    for temp_id, template in enumerate(tqdm(raw_questions)):
        if aug_method == 'eda' or aug_method == 'cbert':
            aug = random_sample_list(augmenter)
        else:
            aug = augmenter

        label_id, raw_question = template
        if aug_method == 'back_translate':
            #current_time = time.time()
            translated = aug['forward_model'].generate(**aug['forward_tokenizer'](raw_question, return_tensors="pt", padding=True))
            #print(translated)
            de_text = aug['forward_tokenizer'].decode(translated.tolist()[0], skip_special_tokens=True)
            #print(time.time()-current_time)
            #current_time = time.time()
            #print(de_text)
            back_translated = aug['backward_model'].generate(**aug['backward_tokenizer'](de_text, return_tensors="pt", padding=True), do_sample=True, num_return_sequences=n)
            filling_results = [aug['backward_tokenizer'].decode(t, skip_special_tokens=True) for t in back_translated.tolist()]
            #print(time.time()-current_time)
            #current_time = time.time()
            #print(raw_question)
            #print(filling_results)
            #print(filling_results)
            #print(raw_question)
        else:
            filling_results = aug.augment(raw_question, n=n)
        #print(len(filling_results))
        for filling_result in filling_results:
            #print(filling_result)
            filling_result = " ".join(tokenizer.tokenize(filling_result))
            #print(filling_result)
            token_list, length = convert_text_to_list(classifier_tok, filling_result, max_length)
            #print(token_list)
            #copied_mem_data[temp_id][2] = token_list
            #copied_mem_data[temp_id][3] = length
            label = mem_data[temp_id][0]
            assert label_id == label
            aug_instance = [label, ([],[]), list(token_list), length]
            aug_data.append(aug_instance)
            raw_question_data.append((label, filling_result))

    #for i in range(len(copied_proto_set)):
    #    copied_proto_set[i] = [proto_set[i][0]]
    #for instance in copied_mem_data:
    #    copied_proto_set[instance[0]].append(instance)
    #print(len(aug_data))
    return aug_data, raw_question_data

def pretrain_nlaug_few_shot(current_relations, question_data, max_length, classifier_tok, plmm, plmm_tokenizer, aug_num, sampler, config):
    aug_data = []
    #print(current_relations)
    tokenizer = nltk.TweetTokenizer()
    label_dict = {}
    for label_id, question in question_data:
        if label_id in label_dict:
            label_dict[label_id].append(question)
        else:
            label_dict[label_id] = []
            label_dict[label_id].append(question)

    raw_question_data = []
    for index, label_id in enumerate(tqdm(current_relations)):
        for i in range(aug_num):
            prompt = sampler.id2rel[label_id] + ": "
            if len(question_data) > 0:
                prompt = prompt + (" [SEP] ".join(sample_k(label_dict[label_id], 3))).strip()
            #print(prompt)
            gen_ids = plmm.generate(**plmm_tokenizer(prompt, return_tensors="pt").to(config['device']), do_sample=True, num_return_sequences=1)
            filling_results = [plmm_tokenizer.decode(t, skip_special_tokens=True) for t in gen_ids.tolist()]

            for filling_result in filling_results:
                #print(sampler.id2rel[label_id])
                #print(filling_result)
                filling_result = " ".join(tokenizer.tokenize(filling_result))
                sentence = filling_result
                token_list, length = convert_text_to_list(classifier_tok, sentence, max_length)
                if len(token_list) <=2:
                    continue
                #print(token_list)
                #copied_mem_data[temp_id][2] = token_list
                #copied_mem_data[temp_id][3] = length
                aug_instance = [label_id, ([],[]), list(token_list), length]
                aug_data.append(aug_instance)
                raw_question_data.append((label_id, filling_result))

    print(len(aug_data))

    return aug_data, raw_question_data




def pretrain_nlaug(current_relations, max_length, classifier_tok, plmm, plmm_tokenizer, aug_num, sampler, class_model, config):


    aug_data = []

    tokenizer = nltk.TweetTokenizer()
    #print(current_relations)
    prune_number = aug_num*len(current_relations)

    raw_question_data = []
    print(len(current_relations))
    for index, label_id in enumerate(tqdm(current_relations)):
        prompt = sampler.id2rel[label_id] + " [SEP]"
        gen_ids = plmm.generate(**plmm_tokenizer(prompt, return_tensors="pt").to(config['device']), do_sample=True , num_return_sequences=aug_num*2)
        filling_results = [plmm_tokenizer.decode(t, skip_special_tokens=True) for t in gen_ids.tolist()]

        for filling_result in filling_results:
            sentence = filling_result.split('[SEP]')[1]

            sentence = " ".join(tokenizer.tokenize(sentence))

            token_list, length = convert_text_to_list(classifier_tok, sentence, max_length)
            if len(token_list) <=2:
                continue
            #print(token_list)
            #copied_mem_data[temp_id][2] = token_list
            #copied_mem_data[temp_id][3] = length
            aug_instance = [label_id, ([],[]), list(token_list), length]
            aug_data.append(aug_instance)
            raw_question_data.append((label_id, filling_result))
    print(len(raw_question_data))
    pruned_data, pruned_question_data = prune_examples(config, class_model, aug_data, raw_question_data, prune_number)
    print(len(pruned_data))
    print(len(pruned_question_data))
    return pruned_data, pruned_question_data



def store_sentence_rep(rep_path, config, data_set, label_id_List, label_list, sentence_list, model):
    data_loader = get_data_loader(config, data_set, False, False, batch_size=1)
    features = []
    for step, (labels, neg_labels, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append((label_id_List[step], label_list[step], sentence_list[step], feature))
    torch.save(features, rep_path)


# Use K-Means to select what samples to save, similar to at_least = 0
def cluster_prototype_stored_instance(eposide_proto_set, features, num_clusters, label_id):
    # data_loader = get_data_loader(config, data_set, False, False, batch_size=1)
    features = np.concatenate(features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    for i in range(num_clusters):
        eposide_proto_set[label_id].append(torch.from_numpy(kmeans.cluster_centers_[i]).to(config['device']))

def posterior_compress_topic_embedding_1(sentence_tokenizer, train_questions, num_clusters, topic_model, device):
    features = []
    for question in train_questions:
        question_ids = sentence_tokenizer(question, add_special_tokens=True, truncation=True,
                                          max_length=100,
                                          is_split_into_words=False, return_tensors='pt')['input_ids'].to(device)

        #if topic_model.vae:
        #    logits, mu, z_var = topic_model(question_ids)
        #else:
        logits = topic_model(question_ids)

        features.append(logits.cpu().detach().numpy())

    features = np.concatenate(features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    cluster_centers = np.concatenate(kmeans.cluster_centers_).reshape(num_clusters, -1)
    return torch.from_numpy(cluster_centers).to(device)


def sample_from_topic_embedding(num_clusters, topic_embedding, device, number):
    # dropout = torch.nn.Dropout(p=0.5)
    # dropout.eval()
    # topic_embedding = dropout(topic_embedding)

    # for i in topic_embedding.size(0):
    #    print(topic_embedding.weight[i])

    features = topic_embedding.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    cluster_centers = np.concatenate(kmeans.cluster_centers_).reshape(num_clusters, -1)
    return torch.from_numpy(cluster_centers).to(device)


def get_proto_stored_instance(rep_instances, label_proto_list, num_sel):
    label_proto_dict = {}
    label_instance_dict = {}
    vectors = []
    for instance in rep_instances:
        label_id = instance[0]
        if label_id in label_instance_dict:
            label_instance_dict[label_id].append(instance[3])
        else:
            label_instance_dict[label_id] = []
            label_instance_dict[label_id].append(instance[3])
        vectors.append(instance[3])

    for label_id, instance in label_instance_dict.items():
        label_proto_dict[label_id] = []

    label_id_list = list(label_instance_dict.keys())
    label_num = num_sel // len(label_instance_dict.keys())
    extra = num_sel % label_num
    cluster_num_list = [label_num] * len(label_instance_dict.keys())
    # print(cluster_num_list)
    cluster_num_list[:extra] = [num + 1 for num in cluster_num_list[:extra]]
    # print(cluster_num_list)
    assert sum(cluster_num_list) == num_sel, " sum of cluster is {0}".format(sum(cluster_num_list))
    for index, cluster_num in enumerate(cluster_num_list):
        label_id = label_id_list[index]
        cluster_prototype_stored_instance(label_proto_dict, vectors, 10, label_id)
    label_proto_list.append(label_proto_dict)


def calculate_distribution(sentence_tokenizer, train_questions, topic_model, device):
    features = []
    topic_features = []
    topic_model.starting_flag = True
    #topic_dis = torch.zeros(topic_model.topic_embedding.weight.size(0))
    for question in train_questions:
        question_ids = sentence_tokenizer(question, add_special_tokens=True, truncation=True,
                                          max_length=100,
                                          is_split_into_words=False, return_tensors='pt')['input_ids'].to(device)

        #if topic_model.vae:
        #    logits, mu, z_var = topic_model(question_ids)
        #else:
        logits = topic_model(question_ids)

        # print(logits.size(), file=sys.stderr)

        features.append(logits.cpu().detach().numpy())

        topic_loss, topic_prior_embed, mse_reg_loss, topic_prior = topic_model.VQ_forward(logits)
        topic_features.append(topic_prior_embed.cpu().detach().numpy())
        #if topic_prior is not None:
        #    print(torch.argmax(topic_prior))
        # print(topic_prior.cpu().detach())
    #topic_dis = topic_dis / len(train_questions)
    #print(topic_dis)

    features = np.concatenate(features)
    kmeans = KMeans(n_clusters=topic_model.topic_embedding.weight.size(0), random_state=0).fit(features)
    print("======================= kmeans labels ==========================================", file=sys.stderr)
    for i in range(topic_model.topic_embedding.weight.size(0)):
        print(kmeans.labels_[i])

    topic_features = np.concatenate(topic_features)
    kmeans = KMeans(n_clusters=topic_model.topic_embedding.weight.size(0), random_state=0).fit(topic_features)
    print("======================= topic kmeans labels ==========================================", file=sys.stderr)
    for i in range(topic_model.topic_embedding.weight.size(0)):
        print(kmeans.labels_[i])


def initilize_topic_embedding(sentence_tokenizer, train_questions, num_clusters, topic_model, device, label_embed = None):
    features = []
    for question in train_questions:
        question_ids = sentence_tokenizer(question, add_special_tokens=True, truncation=True,
                                          max_length=100,
                                          is_split_into_words=False, return_tensors='pt')['input_ids'].to(device)

        #if topic_model.vae:
        #    logits, mu, z_var = topic_model(question_ids)
        #else:
        if label_embed is not None:
            logits =  topic_model.vamp_vae_forward(input_ids=question_ids, z2_logits=label_embed)
        else:
            logits = topic_model(question_ids)

        features.append(logits.cpu().detach().numpy())

    features = np.concatenate(features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    # print("======================= kmeans labels ==========================================")
    # print(kmeans.labels_)
    # unique_labels, label_counts = np.unique(kmeans.labels_,return_counts=True)
    # for i in range(num_clusters):
    #    print(label_counts[i])
    cluster_centers = np.concatenate(kmeans.cluster_centers_).reshape(num_clusters, -1)
    return torch.from_numpy(cluster_centers).to(device)


def initilize_topic_embedding_vae(sentence_tokenizer, train_questions, num_clusters, topic_model, device):

    cluster_centers = np.random.normal(size=(num_clusters,512)) #np.concatenate(kmeans.cluster_centers_).reshape(num_clusters, -1)
    topic_model.topic_embedding.weight.data.copy_(torch.from_numpy(cluster_centers).float().to(device))


def compress_topic_embedding_1(sentence_tokenizer, train_questions, num_clusters, topic_model, device):
    features = []
    for id in range(num_clusters):
        topic_embed = prefix_model.topic_encoder.get_topic_embedding()

        features.append(topic_embed.cpu().detach().numpy())

    features = np.concatenate(features)
    cluster_centers = features.reshape(num_clusters, -1)
    return torch.from_numpy(cluster_centers).to(device)


def compress_topic_embedding(num_clusters, topic_embedding, device):
    # dropout = torch.nn.Dropout(p=0.5)
    # dropout.eval()
    # topic_embedding = dropout(topic_embedding)

    # for i in topic_embedding.size(0):
    #    print(topic_embedding.weight[i])

    features = topic_embedding.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    cluster_centers = np.concatenate(kmeans.cluster_centers_).reshape(num_clusters, -1)
    return torch.from_numpy(cluster_centers).to(device)


# Use K-Means to select what samples to save, similar to at_least = 0
def cluster_prototype(eposide_proto_set, config, model, data_set, num_clusters, label_id):
    data_loader = get_data_loader(config, data_set, False, False, batch_size=1)
    features = []
    for step, (labels, neg_labels, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    for i in range(num_clusters):
        eposide_proto_set[label_id].append(torch.from_numpy(kmeans.cluster_centers_[i]).to(config['device']))


def get_proto(train_data, label_proto_list, model, num_sel, config):
    label_proto_dict = {}
    label_instance_dict = {}
    for instance in train_data:
        label_id = instance[0]
        if label_id in label_instance_dict:
            label_instance_dict[label_id].append(instance)
        else:
            label_instance_dict[label_id] = []
            label_instance_dict[label_id].append(instance)

    for label_id, instance in label_instance_dict.items():
        label_proto_dict[label_id] = []

    label_id_list = list(label_instance_dict.keys())
    label_num = num_sel // len(label_instance_dict.keys())
    extra = num_sel % label_num
    cluster_num_list = [label_num] * len(label_instance_dict.keys())
    # print(cluster_num_list)
    cluster_num_list[:extra] = [num + 1 for num in cluster_num_list[:extra]]
    # print(cluster_num_list)
    assert sum(cluster_num_list) == num_sel, " sum of cluster is {0}".format(sum(cluster_num_list))
    for index, cluster_num in enumerate(cluster_num_list):
        label_id = label_id_list[index]
        cluster_prototype(label_proto_dict, config, model, label_instance_dict[label_id], cluster_num, label_id)
    label_proto_list.append(label_proto_dict)


def prefix_resample_data(args, gpt2, prefix_model, label_ids, id2rel, label_embeds, topic_embeds, tokenizer, max_length, aug_num,
                         bert_tokenizer=None,vae='ivae', dup_score=[]):
    # copied_mem_data = []
    # copied_proto_set = [[] for i in range(len(proto_memory))]
    print("prefix sample==============================================================================",
          file=sys.stderr)
    pseduo_data = []



    generated_instances = generate_topic_instances(args, prefix_model, label_ids,
                                                   [id2rel[label_id] for label_id in label_ids], gpt2,
                                                   label_embeds=label_embeds, topic_embeds=topic_embeds,
                                                   bert_tokenizer=bert_tokenizer, aug_num=aug_num,vae=vae, dup_score=dup_score)
    # print("===============")
    # print(len(generated_instances))
    for label_id, instance in generated_instances:
        token_list, length = convert_text_to_list(tokenizer, instance, max_length)
        pseduo_data.append([label_id, ([], []), list(token_list), length])
    # assert len(copied_mem_data) % config['task_memory_size'] == 0, "length of memory is {0}".format(
    #    len(copied_mem_data))
    # for i in range(len(copied_proto_set)):
    #    copied_proto_set[i] = [proto_memory[i][0]]
    # for instance in copied_mem_data:
    #    copied_proto_set[instance[0]].append(instance)
    # return copied_mem_data, copied_proto_set
    return pseduo_data, generated_instances


def optimus_resample_data(gpt2, prefix_model, label_ids, id2rel, topic_embeds, tokenizer, max_length):
    # copied_mem_data = []
    # copied_proto_set = [[] for i in range(len(proto_memory))]
    print("prefix sample==============================================================================",
          file=sys.stderr)
    pseduo_data = []
    generated_instances = generate_optimus_instances(generation_args, prefix_model, label_ids, [id2rel[label_id] for label_id in label_ids], gpt2,
                                                   topic_embeds)

    # print("===============")
    # print(len(generated_instances))
    for label_id, instance in generated_instances:
        token_list, length = convert_text_to_list(tokenizer, instance, max_length)
        pseduo_data.append([label_id, ([], []), list(token_list), length])
    # assert len(copied_mem_data) % config['task_memory_size'] == 0, "length of memory is {0}".format(
    #    len(copied_mem_data))
    # for i in range(len(copied_proto_set)):
    #    copied_proto_set[i] = [proto_memory[i][0]]
    # for instance in copied_mem_data:
    #    copied_proto_set[instance[0]].append(instance)
    # return copied_mem_data, copied_proto_set
    return pseduo_data, generated_instances

def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_label_embeds_prior(current_relations, label_model, sentence_tokenizer, id2rel, device, training_data):
    label_instance_dict = {}

    for label_id in current_relations:
        label_instance_dict[label_id] = []

    for instance in training_data:
        label = instance[0]
        # if label in label_instance_dict:
        label_instance_dict[label].append(instance)
        # else:
        # label_instance_dict[label] = []
        # label_instance_dict[label].append(instance)

    label_embeds = {}
    label_ids = list(set(current_relations))
    label_tokens = [id2rel[idx] for idx in label_ids]
    for indx, label_token in enumerate(label_tokens):
        label_id = label_ids[indx]
        label_bpe = sentence_tokenizer(label_token, add_special_tokens=True, truncation=True,
                                       max_length=100,
                                       is_split_into_words=False, return_tensors='pt')['input_ids'].to(device)

        # (self, input_ids, attn=None, temp=-1,
        label_embeds[label_id] = []
        if config['vae'] in ['ivae', 'ivae_nocond', 'ivae_nocond_fix']:
            label_embeds[label_id].append(label_model.gaussian_hidden2mean(
                label_model(input_ids=label_bpe, src_input_ids=label_bpe, vae='auto').squeeze()))
            weight = 1
            coff = 0.1
            for instance in label_instance_dict[label_id]:
                instance_bpe = sentence_tokenizer(instance[1], add_special_tokens=True, truncation=True,
                                                  max_length=100,
                                                  is_split_into_words=False, return_tensors='pt')['input_ids'].to(
                    device)
                # if label_model.vae:
                #    logits, label_embed, var = label_model(input_ids=instance_bpe, src_input_ids= label_bpe, temp=0.5)
                # else:
                label_embed = coff * label_model.gaussian_hidden2mean(
                label_model(input_ids=instance_bpe, src_input_ids=label_bpe, vae='auto').squeeze()) #label_model(input_ids=instance_bpe, src_input_ids=label_bpe,
                                     #            vae=config['vae']).squeeze()
                weight += coff
                label_embeds[label_id].append(label_embed.squeeze())
            print(torch.stack(label_embeds[label_id]).size())
            label_embeds[label_id] = torch.stack(label_embeds[label_id]).sum(0) / weight

        elif config['vae'] == 'vae':
            label_embed = torch.from_numpy(np.random.normal(size=(15, 512))).float().to(device)
            for i in range(15):
                label_embeds[label_id].append(label_embed[i].squeeze())
        elif config['vae'] == 'vamp_vae':
            label_embed = label_model.get_vamp_memory_and_var()
            perm = torch.randperm(label_embed.size(0))
            idx = perm[:15]
            random_label_embed = label_embed[idx]
            for i in range(15):
                label_embeds[label_id].append(random_label_embed[i].squeeze())
        elif config['vae'] == 'vq_vae' or config['vae'] == 'c_vae':
            label_embed = label_model.topic_embedding.weight
            perm = torch.randperm(label_embed.size(0))
            idx = perm[:15]
            random_label_embed = label_embed[idx]
            for i in range(15):
                label_embeds[label_id].append(random_label_embed[i].squeeze())

    return label_ids, label_embeds


def get_label_embeds_posterior(current_relations, label_model, sentence_tokenizer, id2rel, device, training_data):
    label_instance_dict = {}

    for label_id in current_relations:
        label_instance_dict[label_id] = []

    for instance in training_data:
        label = instance[0]
        # if label in label_instance_dict:
        label_instance_dict[label].append(instance)
        # else:
        # label_instance_dict[label] = []
        # label_instance_dict[label].append(instance)

    label_embeds = {}
    label_ids = list(set(current_relations))
    label_tokens = [id2rel[idx] for idx in label_ids]
    for indx, label_token in enumerate(label_tokens):
        label_id = label_ids[indx]
        label_bpe = sentence_tokenizer(label_token, add_special_tokens=True, truncation=True,
                                       max_length=100,
                                       is_split_into_words=False, return_tensors='pt')['input_ids'].to(device)

        # (self, input_ids, attn=None, temp=-1,
        label_embeds[label_id] = []
        label_embeds[label_id].append(label_model(input_ids=label_bpe, src_input_ids=label_bpe, vae=config['vae']).squeeze())
        weight = 1
        coff = 0.1
        for instance in label_instance_dict[label_id]:
            instance_bpe = sentence_tokenizer(instance[1], add_special_tokens=True, truncation=True,
                                              max_length=100,
                                              is_split_into_words=False, return_tensors='pt')['input_ids'].to(
                device)
            # if label_model.vae:
            #    logits, label_embed, var = label_model(input_ids=instance_bpe, src_input_ids= label_bpe, temp=0.5)
            # else:
            label_embed = coff * label_model(input_ids=instance_bpe, src_input_ids=label_bpe, vae=config['vae']).squeeze() #label_model(input_ids=instance_bpe, src_input_ids=label_bpe, vae=config['vae']).squeeze()
            weight += coff
            label_embeds[label_id].append(label_embed.squeeze())
        #print(torch.stack(label_embeds[label_id]).size())
        label_embeds[label_id] = torch.stack(label_embeds[label_id]).sum(0) / weight


    return label_ids, label_embeds



def oversample_data(data, question_training_data, sample_times):
    oversampled_data = []
    oversampled_question_data = []
    for i in range(sample_times):
        copied_data = [[label, ([], []), list(token_list), length] for label, neg_label, token_list, length in
                       data]
        copied_question_data = [(id, text) for id, text in question_training_data]
        oversampled_data.extend(copied_data)
        oversampled_question_data.extend(copied_question_data)

    return oversampled_data, oversampled_question_data


def get_vocabulary(sentence_array):
    ''' Compute vocabulary

        :param sentence_array: a list of sentences
        :returns: a list of tokens
    '''
    data_vocabulary = {}
    total = 0

    for sentence in sentence_array:
        for token in sentence.strip().split():
            if token not in data_vocabulary:
                data_vocabulary[token] = 1  # /len(line.strip().split())
            else:
                data_vocabulary[token] += 1  # /len(line.strip().split())
            total += 1

    return total, data_vocabulary


def compute_ttr(sentences):
    ''' Computes the type token ratio

        :param sentences: the sentences
        :returns: The type token ratio (float)
    '''

    total, vocabulary = get_vocabulary(sentences)
    return len(vocabulary) / total


def distinct(seqs, n):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        if n == 1:
            unigrams = Counter(seq)
            intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
            unigrams_all.update(unigrams)
        else:
            bigrams = Counter(zip(*(seq[i:] for i in range(n))))
            #print(bigrams)
            intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))
            bigrams_all.update(bigrams)
    if n == 1:
        inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
        intra_dist1 = np.average(intra_dist1)
        return intra_dist1, inter_dist1
    else:
        inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
        intra_dist2 = np.average(intra_dist2)
        return intra_dist2, inter_dist2


def word_mover_evaluate(nlp_pipeline, questions, label_scores, bleu_scores, reverse_scores,
                        distinct1_sequence_scores,
                    distinct2_sequence_scores ,
                    distinct3_sequence_scores ,
                    ttr_sequence_scores):

    question_dict = {}
    for label_id, question in questions:
        if label_id in question_dict:
            question_dict[label_id].append(question)
        else:
            question_dict[label_id] = []
            question_dict[label_id].append(question)

    for label_id, question_list in question_dict.items():
        score = 0
        bleu_score = 0
        reverse_score = 0
        seqs = [source_text.split(' ') for source_text in question_list]
        distinct1_score = distinct(seqs, 1)
        distinct2_score = distinct(seqs, 2)
        distinct3_score = distinct(seqs, 3)
        distinct1_sequence_scores[label_id] = distinct1_score[0]
        distinct2_sequence_scores[label_id] = distinct2_score[0]
        distinct3_sequence_scores[label_id] = distinct3_score[0]
        ttr_score = compute_ttr(question_list)
        ttr_sequence_scores[label_id] = ttr_score
        for source_text in question_list:
            for target_text in question_list:
                doc1 = nlp_pipeline(source_text)
                doc2 = nlp_pipeline(target_text)
                word_mover_distance = doc1.similarity(doc2)
                if source_text is None or source_text == "":
                    source_text = "ERROR"
                if target_text is None or target_text == "":
                    target_text = "ERROR"
                bleu_score += sacrebleu.corpus_bleu([source_text], [[target_text]]).score
                score += word_mover_distance
                reverse_score += (1 - word_mover_distance)
        label_scores[label_id] = score/(len(question_list)*len(question_list))
        bleu_scores[label_id] = bleu_score/(len(question_list)*len(question_list))
        reverse_scores[label_id] = reverse_score/(len(question_list)*len(question_list))
    return label_scores, bleu_scores, reverse_scores, distinct1_sequence_scores, distinct2_sequence_scores, distinct3_sequence_scores, ttr_sequence_scores

def print_function(printer, avg_acc, whole_acc,round_tasks, p_mode, config):
    printer.output()

    log.info(p_mode + "avg_acc:")
    for i in avg_acc:
        log.info(i)
    log.info(p_mode + "whole_acc:")
    for i in whole_acc:
        log.info(i)
    log.info(p_mode + "task_setting:")
    for i in round_tasks:
        log.info(i)

    avg_total = [np.mean(i[-1]) for i in avg_acc]
    whole_total = np.array([i[-1] for i in whole_acc])
    avg_mean, avg_interval = interval(avg_total)
    log.info(p_mode + "avg_mean: " + str(avg_mean) + p_mode + " avg_interval: " + str(avg_interval))
    whole_mean, whole_interval = interval(whole_total)
    log.info(p_mode + "whole_mean: " + str(whole_mean) + p_mode + " whole_interval: " + str(whole_interval))

    # 记载数据
    avg_acc = [[i.tolist() for i in j] for j in avg_acc]
    final_result = {p_mode + "avg_mean": str(avg_mean), p_mode + "avg_interval": str(avg_interval), p_mode + "whole_mean": str(whole_mean), p_mode + "whole_interval":str(whole_interval), p_mode + "avg_acc": avg_acc, p_mode + "whole_acc": whole_acc}
    with open(os.path.join(emar_prefix, "log_fewrel/{0}_{1}_{2}_{3}_{4}_{5}_{6}.json".format(p_mode, config['aug'], config['data_type'], config['few_shot_num'], config['prefix_model_name'], params.prior, config['var'])),
              "w") as file_in:
        json.dump(final_result, file_in)

    #record_file.close()


if __name__ == '__main__':

    #record_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    #'dataset/record_results_new.txt')
    #record_file = open(record_file_path, 'w')
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    print(torch.cuda.is_available())
    print(config['device'])
    config['n_gpu'] = torch.cuda.device_count()
    #config['aug_iter'] = int(config['aug_iter']/config['n_gpu'])
    aug = None
    pretrained_tok = None
    pretrained_model = None
    if config['aug'] == 'prompt':
        pretrained_tok = AutoTokenizer.from_pretrained('facebook/bart-base', cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '.cache/huggingface'))  # Initialize tokenizer
        pretrained_model = AutoModelForMaskedLM.from_pretrained('facebook/bart-base', cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.cache/huggingface'))
        pretrained_model.to(config['device'])
    elif config['aug'] == 'few_shot':
        t5_config = T5Config.from_pretrained('t5-small')
        pre_plmm_tokenizer = T5Tokenizer.from_pretrained('t5-small', cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '.cache/huggingface'))  # Initialize tokenizer
        pre_plmm_model = T5ForConditionalGeneration.from_pretrained('t5-small', config=t5_config, cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.cache/huggingface'))

        pre_plmm_model.to(config['device'])
    elif config['aug'] == 'lambda':
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        gpt2_config.use_prefix = False
        gpt2_config.preseqlen = -1
        gpt2_config._my_arg_task_mode = "underspecified"
        gpt2_config._my_arg_tune_mode = "finetune"
        gpt2_config._objective_mode = 0
        pre_plmm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '.cache/huggingface'))  # Initialize tokenizer
        pre_plmm_model = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config, cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.cache/huggingface'))
        pre_plmm_model.to(config['device'])
    elif config['aug'] == 'glove_insert':
        aug = naw.WordEmbsAug(
            model_type='glove', model_path=config['glove_file'],
            action="insert")
    elif config['aug'] == 'eda':
        #a = naw.SynonymAug(aug_src='wordnet')
        aug = []
        aug.append(naw.SynonymAug(aug_src='wordnet'))
        aug.append(naw.RandomWordAug(action="swap"))
        aug.append(naw.RandomWordAug())
        aug.append(naw.WordEmbsAug(
            model_type='glove', model_path=config['glove_file'],
            action="insert"))
    elif config['aug'] == 'gpt2_sentence_insert':
        aug = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2')
    elif config['aug'] == 'cbert':
        aug = []
        aug.append(naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action="insert"))
        aug.append(naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action="substitute"))
    elif config['aug'] == 'back_translate':
        aug = {}
        forward_config = MarianConfig.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        forward_config.use_prefix = False
        forward_config.preseqlen = -1
        aug['forward_tokenizer'] = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        aug['forward_model'] = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de', config=forward_config)

        backward_config = MarianConfig.from_pretrained('Helsinki-NLP/opus-mt-de-en')
        backward_config.use_prefix = False
        backward_config.preseqlen = -1
        aug['backward_tokenizer'] = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        aug['backward_model'] = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en', config=backward_config)

    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2",cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) , '.cache/huggingface'))
    # special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'sep_token': '<SEP>'}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # autoencoder_model = GPT2LMHeadModel.from_pretrained("gpt2",cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) , '.cache/huggingface'), pad_token_id=tokenizer.eos_token_id)
    # autoencoder_model.resize_token_embeddings(len(tokenizer))
    # load_autoencoder()
    print(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       '.cache/huggingface'))

    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])

    root_path = '.'
    word2id = json.load(open(os.path.join(dir_prefix, 'glove/word2id.txt'), encoding='utf-8'))
    word2vec = np.load(os.path.join(dir_prefix, 'glove/word2vec.npy'))

    word2vec_back = word2vec.copy()

    printer = outputer()

    p5_printer = outputer()

    p10_printer = outputer()

    se_printer = outputer()

    avg_acc = []
    whole_acc = []

    p5_avg_acc = []
    p5_whole_acc = []

    p10_avg_acc = []
    p10_whole_acc = []

    se_avg_acc = []
    se_whole_acc = []

    reverse_label_score = []
    label_score = []
    bleu_score = []

    distinct1_score = []
    distinct2_score = []
    distinct3_score = []
    ttr_score = []


    dup_score = []

    nlp = spacy.load('en_core_web_md')
    nlp.pipe(wmd.WMD.SpacySimilarityHook(nlp))

    round_tasks = []


    cache_examples = {}
    
    prefix_model = None
    gpt2 = None
    gpt2_tokenizer = None
    adv_model = None
    for i in range(5):
        start = time.time()
        print("Start to train turn %i" % i)
        print("----------------------------------------")
        # 设置全局的随机种子
        set_seed(config, config['random_seed'] + 100 * i)

        encoder = lstm_encoder(
            token2id=word2id,
            word2vec=word2vec,
            word_size=len(word2vec[0]),
            max_length=128,
            pos_size=None,
            hidden_size=config['hidden_size'],
            dropout=0,
            bidirectional=True,
            num_layers=1,
            config=config)

        if config["fixed"]:
            sampler = topic_data_sampler(config, None, encoder.tokenizer, fixed=config["fixed"],
                                         seed=config['random_seed'] + 100 * i)
        else:
            sampler = topic_data_sampler(config, None, encoder.tokenizer, seed=config['random_seed'] + 100 * i)

        model = proto_softmax_layer(
            encoder,
            num_class=len(sampler.id2rel),
            id2rel=sampler.id2rel,
            drop=0,
            config=config)
        model = model.to(config["device"])

        sentence_tokenizer = AutoTokenizer.from_pretrained(config['bert'], cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '.cache/huggingface'))
        if config['aug'] == 'casual_lens':
            hidden_size = 768
            print('adapting the size of the model embedding to include [PAD]')
            print('len(tokenizer) = ', len(sentence_tokenizer))
            num_added_tokens = sentence_tokenizer.add_special_tokens(
                {'pad_token': '[PAD]'})

        else:
            hidden_size = 512
        # print(sentence_tokenizer.pad_token_id)
        encoder_config = {'hidden_size':hidden_size, 'id2rel': sampler.id2rel, 'config': config, 'sentence_tokenizer_vocab_size': sentence_tokenizer.vocab_size,'sentence_tokenizer':sentence_tokenizer}
        #topic_model = topic_lstm_layer(512, id2rel=sampler.id2rel, config=config,
        #                               bert_vocab_size=sentence_tokenizer.vocab_size,
        #                               sentence_tokenizer=sentence_tokenizer)
        #topic_model = topic_model

        #label_model = label_lstm_layer(config=config, hidden_size=512)

        # AutoModel.from_pretrained('prajjwal1/bert-small', cache_dir=os.path.join(
        # os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        # '.cache/huggingface'))
        #label_model = label_model


        # for param in topic_model.parameters():
        #    nn.init.xavier_normal_(param.data)
        reverse_label_sequence_scores = {}
        label_sequence_scores = {}
        bleu_sequence_scores = {}

        distinct1_sequence_scores = {}
        distinct2_sequence_scores = {}
        distinct3_sequence_scores = {}
        ttr_sequence_scores = {}


        seed_indx = i
        sequence_results = []
        result_whole_test = []

        p5_sequence_results = []
        p5_result_whole_test = []

        p10_sequence_results = []
        p10_result_whole_test = []

        se_sequence_results = []
        se_result_whole_test = []

        mem_data = []
        raw_mem_set = []
        template_set = []
        proto_memory = []
        relation_per_task = []
        last_seen_relations = 0
        num_class = len(sampler.id2rel)
        label_proto_list = []
        for j in range(len(sampler.id2rel)):
            proto_memory.append([sampler.id2rel_pattern[j]])

        generation_args_path = os.path.join(config['prefix_dir'],
                                            'transformers/examples/control/args/generation_args.pickle')
        with open(generation_args_path, 'rb') as handle:
            generation_args = pickle.load(handle)
        vector_dicts = {}


        first_length = 0
        data_length = 0
        sentence_list = []

        for steps, (
                training_data, valid_data, test_data, test_all_data, seen_relations, current_relations,
                question_training_data, question_valid_data, question_test_data) in enumerate(sampler):
            current_time = time.time()

            ori_length = len(training_data)
            print("training ======================", file=sys.stderr)
            if steps > 0 and config['aug'] == 'prefix':
                training_data, question_training_data = oversample_data(training_data, question_training_data, 5)


            if steps > 0 and (config['aug'] == 'prefix'):

                prefix_model.sentence_encoder.eval()
                prefix_model.topic_encoder.eval()

                if params.prior == 'prior':
                    label_ids, label_embeds = get_label_embeds_prior(current_relations, prefix_model.sentence_encoder, sentence_tokenizer,
                                                                   sampler.id2rel, device=config['device'],
                                                                   training_data=question_training_data)
                    if config["topic_num"] == 1 and config['vae'] == 'ivae':
                        topic_embeds = compress_topic_embedding_1(sentence_tokenizer, prefix_model.topic_encoder.train_questions,200, prefix_model.topic_encoder, config["device"])
                    elif config['vae'] in ['ivae','ivae_nocond','ivae_nocond_fix']:
                        topic_embeds = prefix_model.topic_encoder.get_topic_embedding()
                    elif config['vae'] == 'vae':
                        cluster_centers = np.random.normal(size=(
                        15, 512))  # np.concatenate(kmeans.cluster_centers_).reshape(num_clusters, -1)
                        topic_embeds = torch.from_numpy(cluster_centers).float().to(config["device"])
                    elif config['vae'] == 'vq_vae' or config['vae'] == 'c_vae':
                        topic_embed = prefix_model.topic_encoder.topic_embedding.weight
                        perm = torch.randperm(topic_embed.size(0))
                        idx = perm[:15]
                        topic_embeds = topic_embed[idx]
                    elif config['vae'] == 'vamp_vae':
                        topic_embed_list = {}
                        for label_id, in_label_embeds in label_embeds.items():

                            in_topic_embed_list = []
                            for in_label_embed in in_label_embeds:
                                print(in_label_embed.shape)
                                z1_p_mean = prefix_model.topic_encoder.z1_p_mean_layer(in_label_embed.reshape(1,512))
                                z1_p_logvar = prefix_model.topic_encoder.z1_p_logvar_layer(in_label_embed.reshape(1,512))
                                std = torch.exp(0.5 * z1_p_logvar)
                                topic_embeds = torch.randn(z1_p_mean.shape[0], prefix_model.topic_encoder.hidden_size,
                                    device=prefix_model.topic_encoder.config['device']) * std + z1_p_mean
                                print(topic_embeds.shape)
                                for i in range(topic_embeds.size(0)):
                                    # print(i)
                                    # print(compress_topic_embed[i].squeeze())
                                    in_topic_embed_list.append(topic_embeds[i].squeeze())

                            topic_embed_list[label_id] = in_topic_embed_list
                elif params.prior == 'posterior':
                    label_ids, label_embeds = get_label_embeds_posterior(current_relations,
                                                                     prefix_model.sentence_encoder,
                                                                     sentence_tokenizer,
                                                                     sampler.id2rel, device=config['device'],
                                                                     training_data=question_training_data)
                    if config["topic_num"] == 1:
                        config["topic_num"] = 800
                    if config['vae'] == 'vamp_vae':
                        topic_embed_list = {}
                        for label_id, label_embed in label_embeds.items():
                            topic_embeds = initilize_topic_embedding(sentence_tokenizer,
                                                                     prefix_model.topic_encoder.train_questions,
                                                                     config["topic_num"], prefix_model.topic_encoder,
                                                                     config["device"], label_embed= label_embed.unsqueeze(0))
                            topic_embed_list[label_id] = [topic_embeds[i].squeeze() for i in range(topic_embeds.size(0))]

                    else:
                        topic_embeds = initilize_topic_embedding(sentence_tokenizer, prefix_model.topic_encoder.train_questions, config["topic_num"], prefix_model.topic_encoder, config["device"])
                    print(topic_embeds.size())
                #compress_topic_embed = compress_topic_embedding(config['compress_topic_num'], topic_model.topic_embedding.weight, config["device"])

                # compress_topic_embeds(config['compress_topic_num'], topic_model.topic_embedding)
                # print(compress_topic_embed.size(0))
                if not config['vae'] == 'vamp_vae':
                    topic_embed_list = []
                    for i in range(topic_embeds.size(0)):
                        #print(i)
                        # print(compress_topic_embed[i].squeeze())
                        topic_embed_list.append(topic_embeds[i].squeeze())
                # list(torch.split(topic_model.topic_embedding.weight, topic_model.topic_embedding.weight.size(0), dim=0))
                whole_pseduo_data, pseduo_raw_data = prefix_resample_data(generation_args, prefix_model.gpt2_model, prefix_model,
                                                                          label_ids, sampler.id2rel, label_embeds,
                                                                          topic_embed_list, tokenizer=encoder.tokenizer,
                                                                          max_length=128,
                                                                          bert_tokenizer=sentence_tokenizer, aug_num=config['aug_num_per_class'], vae=config['vae'], dup_score = dup_score)
                word_mover_evaluate(nlp_pipeline=nlp, questions=pseduo_raw_data, label_scores=label_sequence_scores,
                                    bleu_scores=bleu_sequence_scores, reverse_scores=reverse_label_sequence_scores,
                                    distinct1_sequence_scores=distinct1_sequence_scores,
                                    distinct2_sequence_scores=distinct2_sequence_scores,
                                    distinct3_sequence_scores=distinct3_sequence_scores,
                                    ttr_sequence_scores=ttr_sequence_scores
                                    )
                training_data = training_data + whole_pseduo_data
                question_training_data = question_training_data + pseduo_raw_data


                if steps == 0:
                    first_length = len(question_training_data)
                data_length = len(question_training_data)


                print(time.time()-current_time)
                current_time = time.time()

                mem_data_back = mem_data.copy()
                proto_memory_back = []
                for i in range(len(sampler.id2rel)):
                    proto_memory_back.append((proto_memory[i]).copy())
                # aug_memory(mem_data, template_set,pretrained_tok,pretrained_model, encoder.tokenizer, encoder.max_length, proto_memory)
                model = train_simple_model(config, model, mem_data + training_data, 1)

                print(time.time()-current_time)
                current_time = time.time()
                # if config['aug'] == 'prefix' and steps > 0:
                #    mem_data, proto_memory = prefix_resample_data(generation_args, label_proto_list, config, gpt2,
                #                                                  prefix_models, proto_memory,
                #                                                  encoder.tokenizer, encoder.max_length, sampler.id2rel)
                #    select_data(mem_data, proto_memory, config, model, training_data, config['task_memory_size'],
                #                raw_mem_set=raw_mem_set, raw_train_set=question_training_data)
                # else:
                if steps > 0 and (config['aug'] == 'lambda'):
                    whole_pseduo_data, pseduo_raw_data = pretrain_nlaug(current_relations, max_length=128, classifier_tok=encoder.tokenizer, plmm=updated_plmm_model, plmm_tokenizer=updated_plmm_tokenizer,
                                     aug_num=config['aug_num_per_class'], sampler=sampler, class_model = model, config=config)

                    word_mover_evaluate(nlp_pipeline=nlp, questions=pseduo_raw_data, label_scores=label_sequence_scores,
                                        bleu_scores=bleu_sequence_scores, reverse_scores=reverse_label_sequence_scores,
                                        distinct1_sequence_scores=distinct1_sequence_scores,
                                        distinct2_sequence_scores=distinct2_sequence_scores,
                                        distinct3_sequence_scores=distinct3_sequence_scores,
                                        ttr_sequence_scores=ttr_sequence_scores
                                        )
                    training_data = training_data + whole_pseduo_data
                    question_training_data = question_training_data + pseduo_raw_data


                if config['task_memory_size'] >= len(training_data):
                    select_whole_data(mem_data, proto_memory, training_data, raw_mem_set=raw_mem_set,
                                      raw_train_set=question_training_data)
                else:
                    select_data(mem_data, proto_memory, config, model, training_data, config['task_memory_size'],
                                raw_mem_set=raw_mem_set, raw_train_set=question_training_data)
                print(time.time()-current_time)
                current_time = time.time()
                for i in range(2):
                    # if len(template_set)>0:
                    #    copied_mem_data, copied_proto_set = aug_memory(mem_data, template_set, pretrained_tok, pretrained_model,
                    #                                                   encoder.tokenizer,
                    #                                                   encoder.max_length, proto_memory,
                    #                                                   [sentence for (label, sentence) in raw_mem_set], aug, config['aug'])
                    #    current_proto = get_memory(config, model, copied_proto_set)
                    # model.set_memorized_prototypes(current_proto)
                    #    model = train_simple_model(config, model, copied_mem_data + training_data, 1)
                    #    model = train_model(config, model, copied_mem_data, 1, current_proto)
                    # else:
                    current_proto = get_memory(config, model, proto_memory)
                    # model.set_memorized_prototypes(current_proto)
                    model = train_simple_model(config, model, mem_data + training_data, 1)
                    model = train_model(config, model, mem_data, 1, current_proto)

                if steps == 0:
                    model_args_path = os.path.join(config['prefix_dir'],
                                                   'transformers/examples/control/args/model_args.pickle')
                    with open(model_args_path, 'rb') as handle:
                        model_args = pickle.load(handle)
                    print(model_args)
                    data_args_path = os.path.join(config['prefix_dir'],
                                                  'transformers/examples/control/args/data_args.pickle')
                    with open(data_args_path, 'rb') as handle:
                        data_args = pickle.load(handle)
                    print(data_args)
                    training_args_path = os.path.join(config['prefix_dir'],
                                                      'transformers/examples/control/args/training_args.pickle')
                    with open(training_args_path, 'rb') as handle:
                        training_args = pickle.load(handle)
                    print(training_args)
                    label_id_List = [instance[0] for instance in question_training_data]
                    sentence_list = [instance[1] for instance in question_training_data]
                    label_list = [sampler.id2rel[label_id] for label_id in label_id_List]

                    #topic_model.train_questions = sentence_list
                    print (encoder_config)
                    encoder_config['train_questions'] = sentence_list
                    if gpt2 is None:
                        gpt2, gpt2_tokenizer = initilize_gpt2(model_args, data_args)
                        gpt2.to(config['device'])

                    prefix_path = params.saved_model_prefix_dir + config['prefix_model_name']
                    #topic_path = os.path.join(prefix_path, 'topic_model.pt')

                    #label_path = os.path.join(prefix_path, 'label_model.pt')
                    if prefix_model is None:
                        if os.path.isdir(prefix_path):
                            #topic_model = torch.load(topic_path)

                            #label_model = torch.load(label_path)

                            prefix_model = load_prefix_model(generation_args, prefix_path, gpt2, config=config, encoder_config=encoder_config)

                            # get_proto_stored_instance(instance_rep, label_proto_list, config['task_memory_size'])
                        else:
                            # initilize_topic_embedding(sentence_tokenizer, sentence_list, config["topic_num"], topic_model, config["device"])
                            generate_dir(prefix_path)
                            # topic_path = os.path.join(prefix_path, 'topic_model.pt')
                            # store_sentence_rep(rep_path, config, training_data, label_id_List, label_list, sentence_list, model)
                            prefix_model = train_prefix(label_id_List, label_list, sentence_list,
                                                        sentence_tokenizer=sentence_tokenizer, gpt2=gpt2,
                                                        tokenizer=gpt2_tokenizer, model_args=model_args,
                                                        data_args=data_args,
                                                        training_args=training_args, prefix_path=prefix_path,
                                                        ratio=first_length / data_length,
                                                        task_mode=config['prefix_task_mode'], aug_epoch=config['aug_epoch'],
                                                        aug_iter=config['aug_iter'],
                                                        examples_per_class=config['sample_per_class'],
                                                        classes_per_episode=config['classes_per_episode'],
                                                        num_support=config["n_support"],
                                                        max_length = config["max_length"],
                                                        warm_epoch = config["warm_epoch"],
                                                        tuning_mode=config["tuning_mode"],
                                                        device=config['device'],
                                                        encoder_config=encoder_config)


                print(time.time() - current_time)
                current_time = time.time()

            current_proto = get_memory(config, model, proto_memory)
            model.set_memorized_prototypes(current_proto)

            if len(label_sequence_scores) > 0:
                print("WMD Diversity Score:")
                print((np.array(list(label_sequence_scores.values()))).mean())
                
                print("BLEU Diversity Score:")
                print((np.array(list(bleu_sequence_scores.values()))).mean())
                
                print("Reverse WMD Diversity Score:")
                print((np.array(list(reverse_label_sequence_scores.values()))).mean())

                print("distinct1 score")
                print((np.array(list(distinct1_sequence_scores.values()))).mean())

                print("distinct2 score")
                print((np.array(list(distinct2_sequence_scores.values()))).mean())

                print("distinct3 score")
                print((np.array(list(distinct3_sequence_scores.values()))).mean())

                print("ttr score")
                print((np.array(list(ttr_sequence_scores.values()))).mean())

                print("label number")
                print(len(label_sequence_scores))




            # print("evaluate model ==========================================")
            results = []
            p5_results = []
            p10_results = []
            se_results = []
            for item in test_data:
                top_acc, p5_acc, p_10_acc, se_acc = evaluate_model_p5(config, model, item, num_class)
                results.append(top_acc)
                p5_results.append(p5_acc)
                p10_results.append(p_10_acc)
                se_results.append(se_acc)
            # cluster 平均准确率
            print("Top Accuracy:")
            print((np.array(results)).mean())
            # 训练过程中打印
            printer.print_list(results)

            print("P5 Accuracy:")
            print((np.array(p5_results)).mean())
            # 训练过程中打印
            p5_printer.print_list(p5_results)

            print("P10 Accuracy:")
            print((np.array(p10_results)).mean())
            # 训练过程中打印
            p10_printer.print_list(p10_results)

            print("SE Accuracy:")
            print((np.array(se_results)).mean())
            # 训练过程中打印
            se_printer.print_list(se_results)

            # 序列任务内
            # sequence_results restore the sequential results of clusters.
            # triangle data
            #print("whole accuracy ==========================================")
            sequence_results.append(np.array(results))
            p5_sequence_results.append(np.array(p5_results))
            p10_sequence_results.append(np.array(p10_results))
            se_sequence_results.append(np.array(se_results))
            #print(len(test_all_data))
            whole_top_acc, whole_p5_acc, whole_p10_acc, whole_se_acc = evaluate_model_p5(config, model, test_all_data, num_class)
            result_whole_test.append(whole_top_acc)
            p5_result_whole_test.append(whole_p5_acc)
            p10_result_whole_test.append(whole_p10_acc)
            se_result_whole_test.append(whole_se_acc)

            #print("ending evaluate whole accuracy ==========================================")
            relations_num_in_current_task = last_seen_relations - len(seen_relations)
            relations_in_current_task = seen_relations[relations_num_in_current_task:]
            relation_per_task.append(relations_in_current_task)
            last_seen_relations = len(seen_relations)
            #print("ending tasks================================================================================")

        # store the result
        avg_acc.append(sequence_results)
        whole_acc.append(result_whole_test)

        p5_avg_acc.append(p5_sequence_results)
        p5_whole_acc.append(p5_result_whole_test)

        p10_avg_acc.append(p10_sequence_results)
        p10_whole_acc.append(p10_result_whole_test)

        se_avg_acc.append(se_sequence_results)
        se_whole_acc.append(se_result_whole_test)

        round_tasks.append(relation_per_task)
        end = time.time()
        print("training time: " + str((end - start) / 60))

        printer.append(sequence_results, result_whole_test)

        p5_printer.append(p5_sequence_results, p5_result_whole_test)

        p10_printer.append(p10_sequence_results, p10_result_whole_test)

        se_printer.append(se_sequence_results, se_result_whole_test)
        if len(label_sequence_scores) > 0:
            label_score.append(float((np.array(list(label_sequence_scores.values()))).mean()))
            bleu_score.append(float((np.array(list(bleu_sequence_scores.values()))).mean()))
            reverse_label_score.append(float((np.array(list(reverse_label_sequence_scores.values()))).mean()))
            distinct1_score.append(float((np.array(list(distinct1_sequence_scores.values()))).mean()))
            distinct2_score.append(float((np.array(list(distinct2_sequence_scores.values()))).mean()))
            distinct3_score.append(float((np.array(list(distinct3_sequence_scores.values()))).mean()))
            ttr_score.append(float((np.array(list(ttr_sequence_scores.values()))).mean()))
        # initialize the models
        model = model.to('cpu')
        del model
        #del topic_model
        #del label_model
        del encoder
        #del prefix_model

        torch.cuda.empty_cache()
        encoder = lstm_encoder(
            token2id=word2id,
            word2vec=word2vec_back.copy(),
            word_size=len(word2vec[0]),
            max_length=128,
            pos_size=None,
            hidden_size=config['hidden_size'],
            dropout=0,
            bidirectional=True,
            num_layers=1,
            config=config)
        model = proto_softmax_layer(
            sentence_encoder=encoder,
            num_class=len(sampler.id2rel),
            id2rel=sampler.id2rel,
            drop=0,
            config=config)
        model.to(config["device"])
    # output the final avg result
    print_function(printer, avg_acc, whole_acc, round_tasks, "p1_", config)
    print_function(p5_printer, p5_avg_acc, p5_whole_acc, round_tasks, "p5_", config)
    print_function(p10_printer, p10_avg_acc, p10_whole_acc, round_tasks, "p10_", config)
    print_function(se_printer, se_avg_acc, se_whole_acc, round_tasks, "se_", config)
    if len(label_score) > 0:
        #==============write diversity score
        avg_diversity_score = sum(label_score)/len(label_score)
        avg_bleu_score = sum(bleu_score)/len(bleu_score)
        avg_reverse_diversity_score = sum(reverse_label_score)/len(reverse_label_score)

        avg_distinct1_score = sum(distinct1_score)/len(distinct1_score)
        avg_distinct2_score = sum(distinct2_score)/len(distinct2_score)
        avg_distinct3_score = sum(distinct3_score)/len(distinct3_score)
        avg_ttr_score = sum(ttr_score)/len(ttr_score)

        avg_dup_score = 0
        if len(dup_score) > 0:
            avg_dup_score = sum(dup_score)/len(dup_score)
        print("task round : ", len(label_score))
        final_result = {"avg_diversity_score": avg_diversity_score, "avg_bleu_score": avg_bleu_score,"avg_reverse_diversity_score":avg_reverse_diversity_score, "avg_dup_score":avg_dup_score,
                        "avg_distinct1_score":avg_distinct1_score,
                        "avg_distinct2_score":avg_distinct2_score,
                        "avg_distinct3_score":avg_distinct3_score,
                        "avg_ttr_score":avg_ttr_score}

        with open(os.path.join(emar_prefix, "log_fewrel/{0}_{1}_{2}_{3}_{4}_{5}_{6}.json".format("diversity", config['aug'], config['data_type'], config['few_shot_num'], config['prefix_model_name'], params.prior, config['var'])),
                  "w") as file_in:
            json.dump(final_result, file_in)