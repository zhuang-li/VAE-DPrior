import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn, optim
import math
import torch.nn.functional as F
from ..base_model import base_model
from transformers import AutoModel
from .attention_layers import MultiHeadAttention, SingleHeadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F



class label_lstm_layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, config, hidden_size):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(label_lstm_layer, self).__init__()

        self.config = config
        self.label_encoder = AutoModel.from_pretrained(config['bert'], cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '.cache/huggingface'))

        for param in self.label_encoder.parameters():
            param.requires_grad = False

        #self.topic_encoder = topic_encoder

        #self.weight_linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        
        self.vae = config['vae']
        if self.vae in ['ivae','vae', 'ivae_nocond', 'ivae_nocond_fix', 'vamp_vae']:
            self.hidden2mean = nn.Linear(hidden_size, hidden_size)
            self.hidden2logv = nn.Linear(hidden_size, hidden_size)


        if self.vae in ['vq_vae','c_vae', 'vamp_vae']:
            self.topic_embedding = nn.Embedding(config['topic_num'], self.hidden_size)


        self.gaussian = config['gaussian']
        if self.gaussian:
            self.gaussian_hidden2mean = nn.Linear(hidden_size, hidden_size)
            self.gaussian_hidden2logv = nn.Linear(hidden_size, hidden_size)
        self.starting_flag = False
        self.re_cluster = int(config['re_cluster'])
        self.iter = 0
        self.tau0 = 1
        self.np_temp = self.tau0
        self.ANNEAL_RATE = 0.00003
        self.MIN_TEMP = 0.5
        if config['attention'] == 'dot':
            self.attention = SingleHeadAttention(d_model=hidden_size)
        elif config['attention'] == 'mhd':
            self.attention = MultiHeadAttention(self.label_encoder.config.num_attention_heads, d_model = hidden_size, d_k = hidden_size//self.label_encoder.config.num_attention_heads, d_v=hidden_size//self.label_encoder.config.num_attention_heads)

            
    def __euclidean_dist__(self, x, y):
        # x: B x D
        # y: T x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)
            
    def forward(self, input_ids, src_input_ids, attn=None, src_attn=None, vae='ivae'):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        # print(input_ids, file=sys.stderr)

        #if self.iter % self.re_cluster == 1:
        #    self.np_temp = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * self.iter), self.MIN_TEMP)


        logits = self.label_encoder(input_ids, attention_mask=attn, return_dict=True,
                                    output_hidden_states=True).last_hidden_state

        label_logits = self.label_encoder(src_input_ids, attention_mask=src_attn, return_dict=True,
                                    output_hidden_states=True).last_hidden_state



        #topic_embed = self.topic_encoder(input_ids, attention_mask=attn, return_dict=True,
        #                            output_hidden_states=True).last_hidden_state

        # B * len * hidden
        #if temp == -1:
        #    current_temp = self.np_temp
        #else:
        #    current_temp = temp

        #weights = torch.sigmoid(self.weight_linear(logits) / current_temp)

        #print(weights, file=sys.stderr)
        if src_attn is not None:
            if self.config['attention'] == 'mhd':
                attn = attn.unsqueeze(-2)
            #print(self.attention(label_logits, logits, logits, mask=attn)[0].size())
            #print(src_attn.size())
            label_embed = self.attention(label_logits, logits, logits, mask=attn)[0]
            label_embed = (label_embed * src_attn.unsqueeze(2).expand(src_attn.size(0), src_attn.size(1), label_embed.size(2))).mean(1)
        else:
            label_embed = (self.attention(label_logits, logits, logits, mask=attn)[0]).mean(1)

        
    
        if vae in ['ivae', 'vae', 'ivae_nocond', 'ivae_nocond_fix', 'vamp_vae']:
            vae_mean = self.hidden2mean(label_embed)
            vae_logv = self.hidden2logv(label_embed)
            vae_std = torch.exp(0.5 * vae_logv)
            label_embed = torch.randn(input_ids.shape[0], self.hidden_size,
                                device=self.config['device']) * vae_std + vae_mean
            if self.training:
                return label_embed, vae_mean, vae_logv
            else:
                return vae_mean
        elif vae == 'auto':
            return label_embed
        elif vae == 'vq_vae':
            inputs, quantized = self.get_vq_vae_rep(label_embed)
            if self.training:
                return inputs, quantized
            else:
                return quantized
        elif vae == 'c_vae':
            return self.c_vae_forward(label_embed)
                

    def get_vamp_memory_and_var(self):
        vae_mean = self.hidden2mean(self.topic_embedding.weight)
        vae_logv = self.hidden2logv(self.topic_embedding.weight)
        vae_std = torch.exp(0.5 * vae_logv)
        latent_embed = torch.randn(vae_mean.shape[0], self.hidden_size,
                                  device=self.config['device']) * vae_std + vae_mean
        if self.training:
            return latent_embed, vae_mean, vae_logv
        else:
            return latent_embed
        

    def get_vq_vae_rep(self, inputs):
        input_shape = inputs.shape
        topic_embedding = self.topic_embedding.weight

        # Flatten input
        flat_input = inputs.view(-1, self.hidden_size)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(topic_embedding ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, topic_embedding.t()))


        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.config['topic_num'], device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, topic_embedding).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()
        return inputs, quantized

        
    def vq_forward(self, inputs, quantized):
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        loss = q_latent_loss + self.config['topic_coeff'] * e_latent_loss

        return loss
       
    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand_like(shape)
        return -torch.log(-torch.log(U + eps) + eps)
        
    def c_vae_forward(self, rep):
        topic_embedding = self.topic_embedding.weight

        distance = self.__euclidean_dist__(rep, topic_embedding)

        logits = -distance

        topic_prior = torch.softmax((logits + self.sample_gumbel(logits))/0.5, dim=1)


        prob_topic = torch.softmax(logits, dim=1)

        topic_smooth = math.log(1/self.config['topic_num'])

        log_prob_topic = torch.log_softmax(logits, dim=1) - topic_smooth


        topic_loss = (prob_topic * log_prob_topic).sum(dim=1).mean()


        topic_prior_embed = topic_prior.matmul(topic_embedding.detach())

        if self.training:
            return self.config['topic_coeff'] * topic_loss, topic_prior_embed
        else:
            return topic_prior_embed


    def ivae_forward(self, inputs, proto, log_var):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        # b * topic_num
        #if self.config['pretrain_coeff'] * self.config['aug_iter'] == self.iter:
        #    self.starting_flag = True

        #self.iter += 1

        #print(self.starting_flag)
        if not self.starting_flag:
            return 0, inputs
        else:

            if not self.gaussian:
                quantized = proto
            else:
                quantized = self.gaussian_hidden2mean(proto)
                #logv = self.gaussian_hidden2logv(proto)
                #std = torch.exp(0.5 * logv)
                #logits = torch.randn(proto.shape[0], self.hidden_size,
                #                    device=self.config['device']) * std + mean
                #print(logits.device)
                #quantized = logits



            # Loss
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            if self.gaussian:
                loss = q_latent_loss + self.config['topic_coeff'] * e_latent_loss - self.config[
                    'gas_coeff'] * torch.mean(log_var)
                #print(loss)
            else:
                loss = q_latent_loss + self.config['topic_coeff'] * e_latent_loss


            return loss, quantized
