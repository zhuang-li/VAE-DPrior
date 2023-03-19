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
from ...prototype_loss import loss_variational


class AdapterModule(nn.Module):
    def __init__(self, hidden_size, ):
        super(AdapterModule, self).__init__()
        #self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size, int(self.hidden_size/2), bias=False)
        self.linear2 = nn.Linear(int(self.hidden_size / 2), self.hidden_size, bias=False)
        #self.linear2 = nn.Linear(self.action_embed_size, self.action_embed_size, bias=False)

    def forward(self, input):
        return torch.tanh(self.linear2(torch.relu(self.linear(input))))

class topic_lstm_layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

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

    def __distance__(self, rep, rel):
        rep_ = rep.view(rep.shape[0], 1, rep.shape[-1])
        rel_ = rel.view(1, -1, rel.shape[-1])
        dis = (rep_ * rel_).sum(-1)
        return dis

    def __init__(self, hidden_size, id2rel, config=None, bert_vocab_size = None, sentence_tokenizer = None):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(topic_lstm_layer, self).__init__()



        self.config = config
        #self.sentence_encoder = sentence_encoder
        #self.num_class = num_class
        self.vae = config['vae']
        self.topic_prior = config['topic_prior']

        self.hidden_size = hidden_size
        #self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.topic_embedding = nn.Embedding(config['topic_num'], self.hidden_size)

        critic_layers = [nn.Linear(self.hidden_size,
                                   self.hidden_size),
                         nn.ReLU(),
                         nn.Linear(self.hidden_size, self.hidden_size)]
        self.topic_readout = nn.Sequential(*critic_layers)

        #self.topic_embedding.weight.requires_grad = False



        #self.topic_readout = torch.nn.Linear(self.hidden_size, config['topic_num'])

        self.topic_query_embedding = nn.Embedding(config['topic_num'], self.hidden_size)

        nn.init.xavier_normal_(self.topic_embedding.weight.data)
        #nn.init.xavier_normal_(self.topic_query_embedding.weight.data)

        #self.topic_embedding.weight.requires_grad = False
        #'prajjwal1/bert-small'

        self.topic_encoder = AutoModel.from_pretrained(config['bert'], cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '.cache/huggingface'))
        #"tuning_mode": "bothtune",
        if not config['style_transfer_model'] == "optimus":
            for n, p in self.topic_encoder.named_parameters():
                #print(n, file=sys.stderr)
                if n.startswith('embeddings') or n.startswith('encoder.layer.0') or n.startswith('encoder.layer.1') or n.startswith('encoder.layer.2'):
                        #or n.startswith('transformer.layer.0') or n.startswith('transformer.layer.1') or n.startswith('transformer.layer.2')\
                        #or n.startswith('transformer.layer.3') or n.startswith('transformer.layer.4') or n.startswith('transformer.layer.5') or n.startswith('transformer.layer.6') \
                        #or n.startswith('transformer.layer.7'):
                    print(n, file=sys.stderr)
                    p.requires_grad = False

        if config['style_transfer_model'] == 'casual_lens':
            self.topic_encoder.resize_token_embeddings(len(sentence_tokenizer))


        self.id2rel = id2rel
        self.rel2id = {}
        for id, rel in id2rel.items():
            self.rel2id[rel] = id

        if self.vae in ['ivae', 'vae', 'vamp_vae', 'ivae_nocond', 'ivae_nocond_fix']:
            self.hidden2mean = nn.Linear(self.hidden_size, self.hidden_size)
            self.hidden2logv = nn.Linear(self.hidden_size, self.hidden_size)
        #self.hidden2mean = nn.Linear(self.hidden_size, self.hidden_size)

        if self.vae == 'vamp_vae':
            self.z1_p_mean_layer = nn.Linear(hidden_size, hidden_size)
            self.z1_p_logvar_layer = nn.Linear(hidden_size, hidden_size)

        self.gaussian = config['gaussian']
        self.is_hard_ivae = config['is_hard_ivae']

        if self.gaussian and self.vae in ['ivae','ivae_nocond', 'ivae_nocond_fix']:
            self.gaussian_hidden2mean = nn.Linear(self.hidden_size, self.hidden_size)
            nn.init.eye_(self.gaussian_hidden2mean.weight)
            self.gaussian_hidden2logv = nn.Linear(self.hidden_size, self.hidden_size)

        if self.vae == 'vamp_vae':
            self.joint_layer = nn.Linear(2*hidden_size, hidden_size)
            self.hidden2mean = nn.Linear(hidden_size, hidden_size)
            self.hidden2logv = nn.Linear(hidden_size, hidden_size)

        if self.topic_prior:
            self.bert_vocab_size = bert_vocab_size
            self.topic_token_linear = nn.Linear(config['topic_num'], bert_vocab_size)
            self.topic_token_criterion = torch.nn.KLDivLoss(reduction='batchmean')

        self.register_buffer('_ema_cluster_size', torch.zeros(config['topic_num']))

        self._ema_w = nn.Parameter(torch.Tensor(config['topic_num'], self.hidden_size))
        self._ema_w.data.normal_()

        self._decay = self.config['decay']
        self._epsilon = self.config['epsilon']

        self.reg_mse_criterion = torch.nn.MSELoss()

        self.dropout = torch.nn.Dropout(p=0.5)

        self.re_cluster = int(config['re_cluster'])
        self.kmeans_prior_prob = None
        if self.re_cluster:
            self.iter = 0
            self.sentence_tokenizer = sentence_tokenizer
            self.train_questions = None
            self.tau0 = 1
            self.np_temp = self.tau0
            self.ANNEAL_RATE = 0.00003
            self.MIN_TEMP = 0.5
            self.topic_coff = 0
            total_iter = self.config['aug_epoch'] * self.config['aug_iter']

            self.period_interval = int(total_iter / 16)
            self.period_index = 0
            self.mid_period_index = self.period_index + int(self.period_interval*0.5)
            self.end_period_index = self.period_index + int(self.period_interval*0.75)
            self.flag = True
            self.starting_flag = False


    def initilize_topic_embedding(self, sentence_tokenizer, train_questions, num_clusters, device):
        features = []
        for question in train_questions:
            if isinstance(question, tuple):
                question = question[0]
            question_ids = sentence_tokenizer(question, add_special_tokens=True, truncation=True,
                                              max_length=100,
                                              is_split_into_words=False, return_tensors='pt')['input_ids'].to(device)

            if self.vae in ['ivae', 'vae', 'ivae_nocond', 'ivae_nocond_fix']:
                logits, mu, z_var = self(question_ids)
                features.append(mu.cpu().detach().numpy())
            else:
                logits = self(question_ids)
                features.append(logits.cpu().detach().numpy())


        features = np.concatenate(features)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
        label_values, label_counts = np.unique(kmeans.labels_, return_counts=True)
        self.kmeans_prior_prob = torch.Tensor(label_counts/label_counts.sum()).to(device)

        print(self.kmeans_prior_prob.sum())

        cluster_centers = np.concatenate(kmeans.cluster_centers_).reshape(num_clusters, -1)
        self.topic_embedding.weight.data.copy_(torch.from_numpy(cluster_centers).to(device))
        self.topic_embedding.weight.requires_grad = False

    def set_memorized_prototypes(self, protos):
        self.prototypes = protos.detach().to(self.config['device'])

    def get_feature(self, sentences, length=None):
        rep = self.sentence_encoder(sentences, length)
        return rep.cpu().data.numpy()

    def get_prefix_feature(self, sentences, length=None):
        rep = self.sentence_encoder(sentences, length)
        return rep

    def get_mem_feature(self, rep):
        dis = self.mem_forward(rep)
        return dis.cpu().data.numpy()

    def forward(self, input_ids, attn=None):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        #print(input_ids, file=sys.stderr)
        logits = self.topic_encoder(input_ids, attention_mask=attn, return_dict=True, output_hidden_states=True).last_hidden_state

        if attn is not None:
            attn = attn.unsqueeze(2).expand(attn.size(0), attn.size(1), logits.size(2))
            logits = (logits*attn).mean(1)
        else:
            logits = logits.mean(1)

        
        if self.vae in ['ivae', 'vae', 'ivae_nocond', 'ivae_nocond_fix']:
            mean = self.hidden2mean(logits)
            logv = self.hidden2logv(logits)
            std = torch.exp(0.5 * logv)
            vae_logits = torch.randn(input_ids.shape[0], self.hidden_size,
                                device=self.config['device']) * std + mean
            if self.training:
                return vae_logits, mean, logv
            else:
                return mean
        elif self.vae == 'auto':
            return logits
        elif self.vae == 'vq_vae':
            inputs, quantized = self.get_vq_vae_rep(logits)
            if self.training:
                return inputs, quantized
            else:
                return quantized
        elif self.vae == 'c_vae':
            return self.c_vae_forward(logits)

    def get_topic_embedding(self):
        if self.vae in ['ivae','ivae_nocond', 'ivae_nocond_fix']:
            if self.gaussian:
                if self.training:
                    logits = self.gaussian_hidden2mean(self.topic_embedding.weight)
                    #logv = self.gaussian_hidden2logv(self.topic_embedding.weight)
                    #std = torch.exp(0.5 * logv)
                    #logits = torch.randn(self.topic_embedding.weight.shape[0], self.hidden_size,
                    #                    device=self.config['device']) * std + mean
                    return logits #, mean, logv
                else:
                    #print("random embed ==================")
                    if self.config['topic_num'] == 1:
                        return self.gaussian_hidden2mean(self.topic_embedding.weight) + self.config['var']*torch.randn_like(self.topic_embedding.weight)
                    else:
                        return self.gaussian_hidden2mean(self.topic_embedding.weight) + self.config['var']*torch.randn_like(self.topic_embedding.weight)
            else:
                return self.topic_embedding.weight
        elif self.vae in ['vq_vae','c_vae']:
            return self.topic_embedding.weight
        

    def log_Normal_diag(self, x, mean, log_var, average=False, dim=None):
        log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
        if average:
            return torch.mean(log_normal, dim)
        else:
            return torch.sum(log_normal, dim)

    def log_p_z2(self, z2, memory, memory_var):
        # z2 - MB x M
        C = self.config['topic_num']

        # calculate params for given data
        z2_p_mean = memory
        z2_p_logvar = memory_var


        # expand z
        z_expand = z2.unsqueeze(1)
        means = z2_p_mean.unsqueeze(0)
        logvars = z2_p_logvar.unsqueeze(0)

        a = self.log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB
        # calculte log-sum-exp
        log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1)))  # MB

        return log_prior


    def vamp_vae_forward(self, input_ids, z2_logits, attn=None):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        # print(input_ids, file=sys.stderr)

        # if self.iter % self.re_cluster == 1:
        #    self.np_temp = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * self.iter), self.MIN_TEMP)

        logits = self.topic_encoder(input_ids, attention_mask=attn, return_dict=True,
                                    output_hidden_states=True).last_hidden_state

        if attn is not None:
            attn = attn.unsqueeze(2).expand(attn.size(0), attn.size(1), logits.size(2))
            logits = (logits*attn).mean(1)
        else:
            logits = logits.mean(1)


        label_embed = self.joint_layer(torch.cat((logits, z2_logits), dim=1))



        vae_mean = self.hidden2mean(label_embed)
        vae_logv = self.hidden2logv(label_embed)
        vae_std = torch.exp(0.5 * vae_logv)
        label_embed = torch.randn(input_ids.shape[0], self.hidden_size,
                                  device=self.config['device']) * vae_std + vae_mean
        if self.training:
            return label_embed, vae_mean, vae_logv
        else:
            return vae_mean



    def vamp_vae_loss(self, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, memory, memory_var):
        # first topic
        # second label
        z1_p_mean = self.z1_p_mean_layer(z2_q)
        z1_p_logvar = self.z1_p_logvar_layer(z2_q)

        log_p_z1 = self.log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
        log_q_z1 = self.log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
        log_p_z2 = self.log_p_z2(z2_q, memory, memory_var)
        log_q_z2 = self.log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
        KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2).mean()

        #print(KL.size())

        return KL


    def EMA_VQA_forward(self, inputs, prototype=True):
        # convert inputs from BCHW -> BHWC
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.hidden_size)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.topic_embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.topic_embedding.weight.t()))

        print(torch.argmin(distances, dim=1), file=sys.stderr)
        print(-distances, file=sys.stderr)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.config['topic_num'], device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.topic_embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self.hidden_size * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self.topic_embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.config['topic_coeff'] * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings

    
    def get_vq_vae_rep(self,inputs):
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

        
    def vq_forward(self,inputs, quantized):
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        loss = q_latent_loss + self.config['topic_coeff'] * e_latent_loss

        return loss





    def ivae_forward(self, inputs, log_var):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        # b * topic_num

        if not self.starting_flag:
            return 0, inputs, 0, 0
        else:
            input_shape = inputs.shape
            if self.gaussian and self.training:
                topic_embedding = self.get_topic_embedding()
            else:
                topic_embedding = self.get_topic_embedding()


            if self.is_hard_ivae:
                # Flatten input
                flat_input = inputs.view(-1, self.hidden_size)

                # Calculate distances
                distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                             + torch.sum(topic_embedding ** 2, dim=1)
                             - 2 * torch.matmul(flat_input, topic_embedding.t()))

                #print(torch.argmin(distances, dim=1), file=sys.stderr)
                #print(-distances, file=sys.stderr)

                # Encoding
                encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
                encodings = torch.zeros(encoding_indices.shape[0], self.config['topic_num'], device=inputs.device)
                encodings.scatter_(1, encoding_indices, 1)

                # Quantize and unflatten
                quantized = torch.matmul(encodings, topic_embedding).view(input_shape)

                # Loss
                e_latent_loss = F.mse_loss(quantized.detach(), inputs)
                q_latent_loss = F.mse_loss(quantized, inputs.detach())
                if self.gaussian:
                    if self.training:
                        loss = q_latent_loss + self.config['topic_coeff'] * e_latent_loss - self.config['gas_coeff'] * torch.mean(log_var)
                    else:
                        loss = q_latent_loss + self.config['topic_coeff'] * e_latent_loss
                else:
                    loss = q_latent_loss + self.config['topic_coeff'] * e_latent_loss

                quantized = inputs + (quantized - inputs).detach()
                avg_probs = torch.mean(encodings, dim=0)
                perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

                # convert quantized from BHWC -> BCHW
                return loss, quantized, perplexity, encodings
            else:

                if self.iter%self.re_cluster == 1:
                    self.np_temp = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * self.iter), self.MIN_TEMP)

                self.iter = self.iter + 1

                distance = self.__euclidean_dist__(inputs, topic_embedding)

                logits = - distance

                topic_prior = torch.softmax((logits + self.sample_gumbel(logits)) / self.np_temp, dim=1)

                batch_topic_prior = topic_prior.unsqueeze(1).view(topic_prior.size(0),1, topic_prior.size(1))

                prob_topic = torch.softmax(logits, dim=1)

                topic_smooth = torch.log(self.kmeans_prior_prob)

                log_prob_topic = torch.log_softmax(logits, dim=1) - topic_smooth

                expand_inputs = inputs.unsqueeze(1).expand(inputs.size(0),topic_embedding.size(0),topic_embedding.size(1))
                expand_topic_embedding = topic_embedding.unsqueeze(0).expand(inputs.size(0),topic_embedding.size(0),topic_embedding.size(1))

                e_latent_loss = torch.mean(F.mse_loss(expand_topic_embedding.detach(), expand_inputs, reduction='none'), dim=-1).view(inputs.size(0), topic_embedding.size(0), 1)
                q_latent_loss = torch.mean(F.mse_loss(expand_topic_embedding, expand_inputs.detach(), reduction='none'), dim=-1).view(inputs.size(0), topic_embedding.size(0), 1)

                #print(batch_topic_prior.matmul(q_latent_loss + self.config['topic_coeff'] * e_latent_loss).squeeze().size())

                topic_loss = self.config['mse_coeff'] * (prob_topic * log_prob_topic).sum(dim=1).mean() + batch_topic_prior.matmul(q_latent_loss + self.config['topic_coeff'] * e_latent_loss).squeeze().mean() - self.config['gas_coeff'] * torch.mean(log_var)

                #topic_prior_embed = topic_prior.matmul(topic_embedding)

                #if self.training:
                return topic_loss, 0, 0, 0



    def c_vae_forward(self, rep):
        self.np_temp = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * self.iter), self.MIN_TEMP)
        self.iter = self.iter + 1
        topic_embedding = self.topic_embedding.weight

        distance = self.__euclidean_dist__(rep, topic_embedding)

        logits = -distance

        topic_prior = torch.softmax((logits + self.sample_gumbel(logits))/self.np_temp, dim=1)


        prob_topic = torch.softmax(logits, dim=1)

        topic_smooth = math.log(1/self.config['topic_num'])

        log_prob_topic = torch.log_softmax(logits, dim=1) - topic_smooth


        topic_loss = (prob_topic * log_prob_topic).sum(dim=1).mean()


        topic_prior_embed = topic_prior.matmul(topic_embedding.detach())

        if self.training:
            return self.config['topic_coeff'] * topic_loss, topic_prior_embed
        else:
            return topic_prior_embed

        

    def topic_forward(self, rep, prototype=True):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        # b * topic_num
        #if self.iter%self.re_cluster == 1:
        #    self.np_temp = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * self.iter), self.MIN_TEMP)
            #print(self.np_temp)
            #print("Re-clustering...")
            #self.initilize_topic_embedding(self.sentence_tokenizer, self.train_questions, self.config['topic_num'], self.config['device'])

        #if 5 * self.period_interval == self.iter:
        #    self.initilize_topic_embedding(self.sentence_tokenizer, self.train_questions, self.config['topic_num'],
        #                                   self.config['device'])
        #    self.starting_flag = True
        #if self.iter == self.period_index:
        #    self.topic_coff = 0
        #    self.flag = True
        #    self.starting_flag = True
        #    self.period_index = self.period_index + self.period_interval
            #for param in self.topic_encoder.parameters():
            #    param.requires_grad = True
        #elif self.iter == self.mid_period_index:
        #    self.flag = False
        #    self.starting_flag = False
        #    self.mid_period_index = self.mid_period_index + self.period_interval
        #    self.topic_coff = 1
            #for param in self.topic_encoder.parameters():
            #    param.requires_grad = False
        #elif self.iter == self.end_period_index:
        #    self.flag = False
        #    self.starting_flag = False
        #    self.topic_coff = 1
        #    self.end_period_index = self.end_period_index + self.period_interval
            #for param in self.topic_encoder.parameters():
            #    param.requires_grad = False

        if not prototype:
            if self.flag:
                annel_interval = int(self.period_interval) - int(self.period_interval*0.5)
                self.topic_coff = self.topic_coff + 1/annel_interval

            self.iter += 1

        if not self.starting_flag:
            return 0, rep, 0, 0
        else:
            topic_embedding = self.topic_embedding.weight

            distance = self.__euclidean_dist__(rep, topic_embedding)
            #print(distance.size(), file=sys.stderr)
            #print(rep.size(), file=sys.stderr)

            logits = -distance

            #logits = self.topic_readout(rep)

            #print(self.topic_embedding.weight, file=sys.stderr)
            #print(logits, file=sys.stderr)
            #print(self.starting_flag, file=sys.stderr)
            #print(self.topic_coff, file=sys.stderr)
            #print(torch.argmax(logits, dim=1), file=sys.stderr)
            #logits = rep.matmul(self.topic_embedding.weight.T)

            #logits = self.topic_readout(rep)

            #print(torch.softmax(alpha * -distance, dim=1))
            #print(distance.size())

            topic_prior = torch.softmax((logits + self.sample_gumbel(logits))/self.np_temp, dim=1)

            #print(self.topic_coff, file=sys.stderr)
            #print(torch.nonzero(topic_prior), file=sys.stderr)
            #print(self.starting_flag, file=sys.stderr)

            prob_topic = torch.softmax(logits, dim=1)

            if self.topic_prior:
                topic_readout = self.topic_readout(rep)
                log_topic_prior_distribution = torch.log_softmax(topic_readout, dim=1)

                topic_prior_distribution = torch.softmax(topic_readout, dim=1)

                log_prob_topic = torch.log_softmax(logits, dim=1) - log_topic_prior_distribution.detach()
            else:
                topic_smooth = math.log(1/self.config['topic_num'])

                log_prob_topic = torch.log_softmax(logits, dim=1) - topic_smooth



            # topic_loss = (topic_prior * distance).sum(dim=1).mean()

            compare_constants = torch.Tensor([0.2]).squeeze().to(self.config['device'])

            topic_loss = torch.max(compare_constants, (prob_topic * log_prob_topic).sum(dim=1).mean())

            #print(prob_topic * log_prob_topic, file=sys.stderr)

            #print(self.topic_coff, file=sys.stderr)
            #print(topic_loss, file=sys.stderr)
            #print(topic_loss)


            #print(topic_prior)

            topic_prior_embed = topic_prior.matmul(topic_embedding.detach())

            #topic_prior_embed = self.dropout(topic_prior_embed)

            mse_reg_loss = self.reg_mse_criterion(topic_prior_embed, rep.detach())

            #print(self.config['topic_coeff'] * topic_loss, file=sys.stderr)


            if self.topic_prior:
                return self.config['topic_coeff'] * topic_loss, topic_prior_embed, self.config['mse_coeff'] * mse_reg_loss, topic_prior_distribution
            else:
                return self.config['topic_coeff'] * topic_loss, topic_prior_embed, self.config['mse_coeff'] * mse_reg_loss, prob_topic


    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand_like(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def topic_word_forward(self, topic_prior, label):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        # b * topic_num


        topic_word_loss = self.topic_token_criterion(torch.log_softmax(self.topic_token_linear(topic_prior), dim=1), label)

        #print(topic_loss)
        return topic_word_loss

