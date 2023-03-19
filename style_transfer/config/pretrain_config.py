import os

import torch
import json
#print ("config dir")
#print (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
dir_prefix = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'dataset')
tarced_dir = os.path.join(dir_prefix , 'tacred/tacred_emar/')
fewrel_dir = os.path.join(dir_prefix , 'fewrel/fewrel_emar/')
webred_dir = os.path.join(dir_prefix , 'webred/')

clinc_dir = os.path.join(dir_prefix , 'clinc150/CLINC150_emar/')

person_dir = os.path.join(dir_prefix , 'personality_caption/random_split/')

glove_path = os.path.join(dir_prefix , 'glove/glove.6B.300d.txt')

prefix_dir_prefix = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'PrefixTuning')


CONFIG = {
    'data_type': 'tarced',
    'learning_rate': 0.001, # 0.001
    'embedding_dim': 300,
    'hidden_size': 200,
    'batch_size': 50,
    'cache_path': 'lambda_model/',
    'gradient_accumulation_steps':1,
    'num_clusters': 10,
    'encoder': 'bilstm', # bilstm bert
    'epoch': 2,
    'random_seed': 100,
    'aug': 'lambda',
    'few_shot_sample_number':10,
    'few_shot_num': 1,
    'task_memory_size': 200,
    'loss_margin': 0.5,
    'sequence_times': 5,
    'initial_task': 5,
    'lambda': 100,
    'num_cands': 10,
    'num_steps': 1,
    'dir_prefix' : dir_prefix,
    'dir':tarced_dir,
    'vector_size': 60,
    'vector_learning_rate': 0.001,
    'epoch_base_vector': 1,
    'num_constrain': 10,
    'data_per_constrain': 5,
    'lr_alignment_model': 0.0001,
    'epoch_alignment_model': 20,
    'checkpoint_path': 'checkpoint',
    'use_gpu': True,
    'relation_file': tarced_dir + 'relation_name.txt',
    'training_file': tarced_dir + 'train.txt',
    'test_file': tarced_dir + 'test.txt',
    'valid_file': tarced_dir + 'valid.txt',
    'glove_file': glove_path,
    'task_name': 'FewRel',
    'num_workers':4,
    'max_grad_norm':1,
    'fixed': False,
    'aug_num_per_class':50,
    'prefix_dir': prefix_dir_prefix,
    'prefix_model_name': "optimus_pretrain",
    'prefix_task_mode' : 'proto_reg',
    'vae': False,
    'topic_prior': False,
    're_cluster': 1000,
    'reg_cluster': 100,
    'aug_epoch':160,
    'aug_iter': 600,
    'topic_num': 800,
    'compress_topic_num': 200,
    'debug': True,
    'topic_coeff':0.1,
    'mse_coeff':1,
    'decay':0.99,
    'epsilon':1e-5,
    'pretrain_coeff':4,
    'sample_per_class': 8,
    'classes_per_episode': 4,
    "n_support": 4,
    "max_length": 128,
    "warm_epoch": 40,
    "attention": 'mhd'
}