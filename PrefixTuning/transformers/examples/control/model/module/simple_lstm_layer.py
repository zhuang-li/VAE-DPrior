import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from transformers import GPT2Model
from ..base_model import base_model


class simple_lstm_layer(base_model):

    def __init__(self, max_length=128, embed_num=-1, input_size=256, output_size=256, dropout=0, bidirectional=True,
                 num_layers=1, config=None):
        """
        Args:
            input_size: dimention of input embedding
            hidden_size: hidden size
            dropout: dropout layer on the outputs of each RNN layer except the last layer
            bidirectional: if it is a bidirectional RNN
            num_layers: number of recurrent layers
            activation_function: the activation function of RNN, tanh/relu
        """
        super(simple_lstm_layer, self).__init__()
        self.device = config['device']
        print("========================== embed size ======================")
        print(embed_num)
        self.max_length = max_length
        self.output_size = output_size
        self.input_size = input_size

        self.gpt2_classifier = GPT2Model.from_pretrained('distilgpt2', cache_dir=config['cache_path'])
        self.gpt2_classifier.resize_token_embeddings(embed_num)
        self.embed = self.gpt2_classifier.wte
        self.config = config
        if bidirectional:
            self.hidden_size = output_size // 2
        else:
            self.hidden_size = output_size
        # self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional = bidirectional, num_layers = num_layers, dropout = dropout)

        self.iter = 0

        self.tau0 = 1
        self.np_temp = self.tau0
        self.ANNEAL_RATE = 0.00003
        self.MIN_TEMP = 0.5

    def gumbel_softmax(self, logits):

        self.np_temp = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * self.iter), self.MIN_TEMP)
        self.iter = self.iter + 1

        topic_embedding = self.embed.weight

        topic_prior = torch.softmax((logits + self.sample_gumbel(logits)) / self.np_temp, dim=-1)

        topic_prior_embed = topic_prior.matmul(topic_embedding)

        return topic_prior_embed

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand_like(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def init_hidden(self, batch_size=1, device='cpu'):
        self.hidden = (torch.zeros(2, batch_size, self.hidden_size).to(device),
                       torch.zeros(2, batch_size, self.hidden_size).to(device))

    def forward(self, input_ids, attn=None):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        # print(input_ids, file=sys.stderr)
        logits = self.gpt2_classifier(input_ids, attention_mask=attn, return_dict=True,
                                    output_hidden_states=True).last_hidden_state

        if attn is not None:
            attn = attn.unsqueeze(2).expand(attn.size(0), attn.size(1), logits.size(2))
            logits = (logits * attn).mean(1)
        else:
            logits = logits.mean(1)

        return logits

    def gumbel_forward(self, logits, attn=None):
        embed = self.gumbel_softmax(logits)
        #breakpoint()
        # print (embed.size())
        # print (lengths)
        self.init_hidden(embed.shape[0], logits.device)
        # packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu(), batch_first = True, enforce_sorted=False)
        # lstm_out, hidden = self.lstm(packed_embeds, self.hidden)
        # permuted_hidden = hidden[0].permute([1,0,2]).contiguous()
        # output_embedding = permuted_hidden.view(-1, self.hidden_size * 2)
        logits = self.gpt2_classifier(inputs_embeds=embed, attention_mask=attn, return_dict=True,
                                    output_hidden_states=True).last_hidden_state

        if attn is not None:
            attn = attn.unsqueeze(2).expand(attn.size(0), attn.size(1), logits.size(2))
            logits = (logits * attn).mean(1)
        else:
            logits = logits.mean(1)

        return logits
