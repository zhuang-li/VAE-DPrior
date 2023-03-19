import argparse
import copy
import csv
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as st
import sys
import random
import json
import os
from tqdm import tqdm
from sklearn.cluster import KMeans

from transformers import AutoTokenizer

sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(sys_path)
print(sys.path)
from PrefixTuning.transformers.examples.control.run_generation_clean import load_prefix_model, \
    generate_topic_instances, generate_optimus_instances
from PrefixTuning.transformers.examples.control.run_language_modeling_clean import train_prefix, initilize_gpt2
from autoencoders.utils import sample_k
from style_transfer_config.config import CONFIG as config

from lifelong.model.module import topic_lstm_layer, label_lstm_layer
from data_loader import get_data_loader
# -------------------------------------------------
import os
import logging

emar_prefix = os.path.dirname(os.path.abspath(__file__))

# f = open(emar_prefix + "config/config.json", "r")
# config = json.loads(f.read())

# print (torch.cuda.is_available())
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_CACHE'] = os.path.join(sys_path, '.cache/huggingface')

print(os.environ['TRANSFORMERS_CACHE'])

dir_prefix = config['dir_prefix']

log = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

log.info(config)


def convert_text_to_list(classifier_tok, sample, max_length):
    tokens = classifier_tok.tokenize(sample)
    length = min(len(tokens), max_length)
    tokens = classifier_tok.convert_tokens_to_ids(tokens, unk_id=classifier_tok.vocab['[UNK]'])

    if (len(tokens) > max_length):
        tokens = tokens[:max_length]
    return tokens, length



def random_sample_list(sample_list):
    random.shuffle(sample_list)
    return sample_list[0]




def get_content_embedding(sentence_tokenizer, train_questions, topic_model, device):
    features = []
    for question in train_questions:
        question_ids = sentence_tokenizer(question, add_special_tokens=True, truncation=True,
                                          max_length=100,
                                          is_split_into_words=False, return_tensors='pt')['input_ids'].to(device)

        if topic_model.vae:
            logits, mu, z_var = topic_model(question_ids)
        else:
            logits = topic_model(question_ids)

        features.append(logits)
    return features


def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_label_embeds(current_relations, label_model, sentence_tokenizer, id2rel, device, training_data):
    label_instance_dict = {}

    for instance in training_data:
        label = instance[0]
        if label in label_instance_dict:
            label_instance_dict[label].append(instance)
        else:
            label_instance_dict[label] = []
            label_instance_dict[label].append(instance)

    label_embeds = {}
    label_ids = list(set(current_relations))
    label_tokens = [id2rel[id] for id in label_ids]
    for indx, label_token in enumerate(label_tokens):
        label_id = label_ids[indx]
        label_bpe = sentence_tokenizer(label_token, add_special_tokens=True, truncation=True,
                                       max_length=100,
                                       is_split_into_words=False, return_tensors='pt')['input_ids'].to(device)

        # (self, input_ids, attn=None, temp=-1,
        label_embeds[label_id] = []
        label_embeds[label_id].append(label_model(input_ids=label_bpe, src_input_ids= label_bpe, temp=0.5).squeeze())
        for instance in label_instance_dict[label_id]:
            instance_bpe = sentence_tokenizer(instance[1], add_special_tokens=True, truncation=True,
                                              max_length=100,
                                              is_split_into_words=False, return_tensors='pt')['input_ids'].to(device)
            label_embeds[label_id].append(label_model(input_ids=instance_bpe, src_input_ids= label_bpe, temp=0.5).squeeze())
        label_embeds[label_id] = torch.stack(label_embeds[label_id]).mean(0)

    return label_ids, label_embeds


def oversample_data(data, question_training_data, sample_times):
    oversampled_data = []
    oversampled_question_data = []
    for i in range(sample_times):
        copied_data = [[label, list(neg_label), list(token_list), length] for label, neg_label, token_list, length in
                       data]
        copied_question_data = [(id, text) for id, text in question_training_data]
        oversampled_data.extend(copied_data)
        oversampled_question_data.extend(copied_question_data)

    return oversampled_data, oversampled_question_data

def write_result(label_ids, id2rel, sentence_list, results, file_path):
    ft = open(file=file_path, mode='w')
    cnt = 0
    for label_iter_index, source_label_id in enumerate(label_ids):

        source_label_name = id2rel[source_label_id]

        for sentence_iter_index, source_sentence in enumerate(sentence_list):

            res = results[cnt]
            target_label_id = res[0]
            target_sentence = res[1]

            target_label_name = id2rel[target_label_id]

            ft.write(str(source_label_id) + '\t' + str(
                target_label_id) + '\t' + source_label_name + '\t' + target_label_name + '\t' + source_sentence + '\t' + target_sentence + '\n')
            ft.close()

            cnt += 1


DEFAULT_CONFIG = "autoencoders/config/default.json"
LOG_DIR_NAME = "logs/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/empathetic/back_translation/back_translation_test.json",
                        help="The config file specifying all params.")
    parser.add_argument("--shuffle", type=int, default=0,
                        help="The shuffle index")
    parser.add_argument("--shot", type=int, default=1,
                        help="The shot number")

    params = parser.parse_args()
    with open(DEFAULT_CONFIG) as f:
        config = json.load(f)
    with open(params.config) as f:
        config.update(json.load(f))

    print(config)

    n = argparse.Namespace()
    n.__dict__.update(config)
    return n, params



if __name__ == '__main__':

    config, params = parse_args()

    print(json.dumps(config.__dict__, indent=4))

    print(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       '.cache/huggingface'))
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    print(torch.cuda.is_available())
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])

    root_path = '..'

    label_names = pd.read_csv(config.label_name, header=None)

    label_names = label_names[0].tolist()

    id2rel = {}

    for index, label_name in enumerate(label_names):
        id2rel[index] = label_name


    sentence_tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small', cache_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        '.cache/huggingface'))


    num_class = len(id2rel)

    generation_args_path = os.path.join(config['prefix_dir'],
                                        'transformers/examples/control/args/generation_args.pickle')
    with open(generation_args_path, 'rb') as handle:
        generation_args = pickle.load(handle)

    prefix_model = None
    gpt2 = None
    gpt2_tokenizer = None

    model_args_path = os.path.join(config['prefix_dir'],
                                   'transformers/examples/control/args/model_args.pickle')
    with open(model_args_path, 'rb') as handle:
        model_args = pickle.load(handle)
    data_args_path = os.path.join(config['prefix_dir'],
                                  'transformers/examples/control/args/data_args.pickle')
    with open(data_args_path, 'rb') as handle:
        data_args = pickle.load(handle)
    training_args_path = os.path.join(config['prefix_dir'],
                                      'transformers/examples/control/args/training_args.pickle')
    with open(training_args_path, 'rb') as handle:
        training_args = pickle.load(handle)

    prefix_path = '/data/saved_models/prefix_model/' + config['prefix_model_name']
    topic_path = os.path.join(prefix_path, 'topic_model.pt')

    support_set_path = os.path.join(config.data_dir, 'support/0_0/support.csv')

    seen_test_path = os.path.join(config.data_dir, 'seen_test/seen_test.csv')

    unseen_test_path = os.path.join(config.data_dir, 'unseen_test/unseen_test.csv')

    topic_model = torch.load(topic_path)

    support_set_frame = pd.read_csv(filepath_or_buffer=support_set_path, sep='\t', header=None, lineterminator='\n',
                             quoting=csv.QUOTE_NONE, encoding='utf-8')

    seen_data_frame = pd.read_csv(filepath_or_buffer=seen_test_path, sep='\t', header=None, lineterminator='\n',
                             quoting=csv.QUOTE_NONE, encoding='utf-8')

    style_transfer_seen_res_file = os.path.join(config.seen_result_file,"{0}_{1}".format(params.shuffle, params.shot))

    style_transfer_unseen_res_file = os.path.join(config.unseen_result_file,"{0}_{1}".format(params.shuffle, params.shot))

    label_ids = list(set(support_set_frame[0].tolist()))

    label_tokens = [id2rel[id] for id in label_ids]


    if gpt2 is None:
        gpt2, gpt2_tokenizer = initilize_gpt2(model_args, data_args)
        gpt2.to(config['device'])

    # initilize_topic_embedding(sentence_tokenizer, sentence_list, config["topic_num"], topic_model, config["device"])
    generate_dir(prefix_path)
    # topic_path = os.path.join(prefix_path, 'topic_model.pt')
    # store_sentence_rep(rep_path, config, training_data, label_id_List, label_list, sentence_list, model)

    seen_label_id_List = seen_data_frame[0].tolist()
    seen_label_list = seen_data_frame[1].tolist()
    seen_sentence_list = seen_data_frame[2].tolist()

    seen_topic_embeds = get_content_embedding(sentence_tokenizer, seen_sentence_list, topic_model, config['device'])

    seen_test_results = generate_optimus_instances(generation_args, prefix_model, label_ids, label_tokens, gpt2, seen_topic_embeds)



    unseen_data_frame = pd.read_csv(filepath_or_buffer=unseen_test_path, sep='\t', header=None, lineterminator='\n',
                             quoting=csv.QUOTE_NONE, encoding='utf-8')

    unseen_label_id_List = unseen_data_frame[0].tolist()
    unseen_label_list = unseen_data_frame[1].tolist()
    unseen_sentence_list = unseen_data_frame[2].tolist()

    unseen_topic_embeds = get_content_embedding(sentence_tokenizer, unseen_sentence_list, topic_model, config['device'])

    unseen_test_results = generate_optimus_instances(generation_args, prefix_model, label_ids, label_tokens, gpt2, unseen_topic_embeds)
    
    write_result(label_ids, id2rel, seen_sentence_list, seen_test_results, style_transfer_seen_res_file)
    
    write_result(label_ids, id2rel, unseen_sentence_list, unseen_test_results, style_transfer_unseen_res_file)
