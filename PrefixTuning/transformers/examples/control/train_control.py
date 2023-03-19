# from transformers import Trainer
import sys

import torch
from PrefixTuning.transformers.examples.control.model.module import simple_lstm_layer
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Tokenizer
from torch import  nn
import torch.nn.functional as F
from PrefixTuning.transformers.examples.control.prototype_loss import HLoss, prototypical_loss, loss_variational, hsic, MMD
from PrefixTuning.transformers.examples.control.model.module.topic_lstm_layer import topic_lstm_layer
from PrefixTuning.transformers.examples.control.model.module.label_lstm_layer import label_lstm_layer

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


class PrefixTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False, deep_param=False, encoder_config = None):
        super().__init__(config)
        print('under the PrefixTuning model')

        #self.topic_encoder = topic_encoder
        #self.sentence_encoder = sentence_encoder
        
        self.gpt2_model = model_gpt2
        #print(encoder_config)
        self.topic_encoder = topic_lstm_layer(encoder_config['hidden_size'], id2rel=encoder_config['id2rel'], config=encoder_config['config'],
                                       bert_vocab_size=encoder_config['sentence_tokenizer_vocab_size'],
                                       sentence_tokenizer=encoder_config['sentence_tokenizer'])
        self.topic_encoder.train_questions = encoder_config['train_questions']
        
        self.class_num = len(encoder_config['id2rel'])
        self.sentence_encoder = label_lstm_layer(config=encoder_config['config'], hidden_size=encoder_config['hidden_size'])
        self.encoder_config = encoder_config['config']
        # AutoModel.from_pretrained('prajjwal1/bert-small', cache_dir=os.path.join(
        # os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        # '.cache/huggingface'))
        #if self.encoder_config['disentangle_loss'] == 4:
        #    self.adv_label_classifier = nn.Linear(encoder_config['hidden_size'], len(encoder_config['id2rel']))
        #if self.encoder_config['vae'] == 'ivae_nocond':
        #    self.label_embed_ivae_nocond = nn.Embedding(len(encoder_config['id2rel']), encoder_config['hidden_size'])

        self.intervention = self.encoder_config['intervention']
        if self.intervention == 1:
            self.label_embed = nn.Embedding(len(encoder_config['id2rel']), 70)
            self.intervention_lstm = simple_lstm_layer(max_length = 128, embed_num = self.gpt2_model.transformer.config.vocab_size, input_size = 256, output_size = encoder_config['hidden_size'], dropout = 0, bidirectional = True, num_layers = 1, config = self.encoder_config)
            self.intervention_linear = nn.Linear(encoder_config['hidden_size'], len(encoder_config['id2rel']),)
        
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.encoder_embed = config.encoder_embed

        self.proto_type_embed = None
        self.task_mode = None
        self.n_support = config.n_support
        self.classes_per_it = config.classes_per_it

        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
            if self.task_mode == 'no_hidden' or self.task_mode == 'casual_lens':
                self.sentence_encoder = None

        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'


        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(config, 'lowdata'):
            self.lowdata = config.lowdata
        else:
            self.lowdata = False

        if hasattr(config, 'lowdata_token'):
            self.lowdata_token = config.lowdata_token
        else:
            self.lowdata_token = None





        if self.task_mode == 'dataless':
            self.mode_para = 1
        elif self.task_mode == 'data2text' or self.task_mode == 'triples' or self.task_mode == 'webnlg' or \
                self.task_mode == 'writingPrompts':
            # with src and input based encoding.
            self.mode_para = 2
            # self.mode_para=0 and optim_prefix == True for Instruction based.
        else:
            self.mode_para = 4

        if not self.optim_prefix:
            if self.train_weights:
                self.wte = model_gpt2.transformer.wte
                for p in self.wte.parameters():
                    p.requires_grad = True
            else:
                if not self.init_random:
                    self.wte = None
                else:
                    print('the is just for baseline checking!!! We reinitialize the LM embeddings and try cat '
                          'and peek.')
                    print('BASELINE'*100)
                    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
                    print(self.wte)



            if self.mode_para == 1:
                print('mode_para=1, for dataless.')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p4_infix
                else:
                    self.get_prompt = self.get_prompt_p4
            elif self.mode_para == 2 or self.mode_para == 4:
                print('mode_para=2 or 4, for (2)data2text having a variable length input prefix parametrization. or for (4) topic/keyword/attributes...')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p3_infix
                else:
                    self.get_prompt = self.get_prompt_p3


            elif self.mode_para == 3:
                print('mode_para=3, OLD VERSION: many parameters.')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.preseqlen * config.n_layer * 2 * config.n_embd), nn.Tanh())
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p1_infix
                else:
                    self.get_prompt = self.get_prompt_p1
        else:
            self.mode_para = 0
            print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
            print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))


            if self.lowdata and self.lowdata_token is not None:
                low_data_init = 3
                if low_data_init == 1:
                    print('IN THE LOW DATA SETTING, EXPLORE INITIALIZATION FOR DIRECT OPTIM...')
                    # self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p22
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                    sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    src, tgt = sample_text.split('||')
                    sample_input = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)
                    self.control_trans = self.lowdata_init_train1(gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input)
                    print(self.control_trans.shape)
                elif low_data_init == 2:
                    print('IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, need to train first')
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p5

                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                    # sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    src, tgt = sample_text.split('||')
                    sample_input = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)

                elif low_data_init == 3:
                    # use a single prepended token.
                    assert self.lowdata_token is not None
                    self.preseqlen = len(self.lowdata_token[0])
                    print('IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, low_data_init=3, '
                          'preseqlen = {} Unifying with FINETUNE'.format(self.preseqlen))
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd + self.encoder_embed, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p5






            # DIFFERENT PARAMETRIZATION:
            elif not deep_param:
                low_data_init = 0
                print('UNDER PARAMETRIZATION 1')
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                #self.test_gpu = nn.Embedding(self.preseqlen, config.n_embd)
                if self.task_mode == 'no_hidden':
                    self.control_trans = nn.Sequential(
                        nn.Linear(int(self.encoder_embed/2), self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                elif self.task_mode == 'casual_lens':
                    self.control_trans = nn.Sequential(
                        nn.Linear(int(self.encoder_embed/2) + 70, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                else:
                    if self.encoder_config['activation'] == 'tanh':
                        self.control_trans = nn.Sequential(
                            nn.Linear(config.n_embd + self.encoder_embed, self.mid_dim),
                            nn.Tanh(),
                            nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                    elif self.encoder_config['activation'] == 'relu':
                        print("========================================= using ReLU ==========================================")
                        self.control_trans = nn.Sequential(
                            nn.Linear(config.n_embd + self.encoder_embed, self.mid_dim),
                            nn.ReLU(),
                            nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))

                self.expand_topic = nn.Sequential(
                    nn.Linear(int(self.encoder_embed/2), int(self.encoder_embed/4)),
                    nn.ReLU(),
                    nn.Linear(int(self.encoder_embed/4), self.preseqlen*int(self.encoder_embed/2)),
                    nn.Tanh())

                if self.use_infix:
                    self.wte2 = nn.Embedding(self.preseqlen, config.n_embd)
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

            else:
                low_data_init = 0
                print('UNDER PARAMETRIZATION DEEP 1')
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd + self.encoder_embed, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5


            # DIFFERENT PARAMETRIZATION 2.
            # elif True:
            #     print('UNDER PARAMETRIZATION 2')
            #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            #     input_word_lst = [['name', 'Type', 'price', 'customer rating', 'near', 'area', 'family friendly']]
            #     input_word_ids = tokenizer(input_word_lst, add_special_tokens=True, is_split_into_words=True, return_tensors='pt')['input_ids']
            #     self.input_embs = model_gpt2.transformer.wte(input_word_ids.to(model_gpt2.device))
            #     print(self.input_embs.shape)
            #     self.control_trans = nn.Sequential(
            #         nn.Linear(config.n_embd, self.mid_dim),
            #         nn.Tanh(),
            #         nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
            #     if self.use_infix:
            #         self.get_prompt = self.get_prompt_p6_infix
            #     else:
            #         self.get_prompt = self.get_prompt_p6



            # OLD CODE.
            # self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
            # if self.use_infix:
            #     assert False, "just optimizing a set of parameter is not really related to infix position."
            #     self.get_prompt = self.get_prompt_p2_infix
            # else:
            #     self.get_prompt = self.get_prompt_p2

        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.forward = self.forward_infix

        ###### just trying #########
        total_param = 0
        total_trainable_param = 0
        for name, param in self.named_parameters():
            #print(param.shape)

            if param.requires_grad == True:
                total_trainable_param += param.numel()

            total_param += param.numel()
        print('total param is {}'.format(total_param))
        print('total trainable param is {}'.format(total_trainable_param))


        if low_data_init == 2:
            self.lowdata_init_train2(gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input)
        elif low_data_init == 3:
            print('use pt for this tensor', torch.LongTensor(self.lowdata_token))
            self.lowdata_init_train3(gpt2=model_gpt2, sample_input=torch.LongTensor(self.lowdata_token))



    def lowdata_init_train1(self, gpt2, tokenizer, sample_input):
        input = tokenizer(sample_input, return_tensors='pt')
        output = gpt2(input['input_ids'].to(gpt2.device), return_dict=True, use_cache=True)
        output = output.past_key_values
        print(len(output), output[0].shape)
        output = torch.cat(output, dim=0).detach()
        return torch.nn.Parameter(output)

    def get_prompt_p22(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        past_key_values = self.control_trans.expand(-1, bsz, -1, -1, -1).split(2, dim=0)
        return past_key_values

    def lowdata_init_train2(self, gpt2, tokenizer, sample_input, epochs=500): # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            input = tokenizer(sample_input, return_tensors='pt')
            output = gpt2(input['input_ids'].to(gpt2.device), return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)

        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = torch.cat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            print(loss)
            loss.backward()
            optimizer_temp.step()
            self.control_trans.zero_grad()

        return


    def lowdata_init_train3(self, gpt2, sample_input, epochs=500): # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            output = gpt2(sample_input.to(gpt2.device), return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)

        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = torch.cat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            print(loss)
            loss.backward()
            optimizer_temp.step()
            self.control_trans.zero_grad()
        return

    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        temp_control = self.control_trans.view(1, self.preseqlen,  self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd).expand(bsz, -1, -1, -1, -1)
        temp_control = self.dropout(temp_control)
        past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def get_prompt_p3_infix(self, src, control_code=None, gpt2=None, bsz=None):
        # temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        # print('infix')
        src_out = gpt2(input_ids=src, use_cache=True, return_dict=True, output_hidden_states=True)
        src_repr = src_out.hidden_states[-1] #bsz, seqlen, hidden
        src_past_key_vals = src_out.past_key_values
        past_key_values = self.control_trans(src_repr) #bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        # print(past_key_values.shape)
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        full_lst = []
        for i in range(len(src_past_key_vals)):
            full_lst.append(torch.cat([src_past_key_vals[i], past_key_values[i]], dim=3))

        return full_lst

    def get_prompt_p3(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values.shape
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
        return past_key_values


    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None, sentences_embed=None, topic_embed=None, wte=None,control_trans=None):
        if self.task_mode == 'no_hidden':
            expand_topic_embed = topic_embed.expand(bsz, self.preseqlen, topic_embed.size(2))

            temp_control = expand_topic_embed
            if wte is None:
                wte = self.wte
            if control_trans is None:
                control_trans = self.control_trans
        elif self.task_mode == 'casual_lens':
            expand_topic_embed = topic_embed.expand(bsz, self.preseqlen, topic_embed.size(2))

            temp_control = expand_topic_embed
            if wte is None:
                wte = self.wte
            if control_trans is None:
                control_trans = self.control_trans
            if sentences_embed is not None:
                expand_sentences_embed = sentences_embed.expand(bsz, temp_control.size(1), sentences_embed.size(2))
                temp_control = torch.cat([temp_control, expand_sentences_embed], dim=-1)
        else:
            #print("topic device")
            #print(topic_embed.get_device())
            
            if wte is None:
                wte = self.wte
            if control_trans is None:
                control_trans = self.control_trans
            
            input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).cuda()
            #print("===========")
            #print(input_tokens.get_device())
            #print("wte device")
            #print(wte.weight.device)
            #print("test gPUS device")
            #print(self.test_gpu.weight.device)
            temp_control = wte(input_tokens)
            #temp_control = torch.zeros_like(temp_control)
            #print(sentences_embed.size())
            if sentences_embed is not None:
                expand_sentences_embed = sentences_embed.expand(bsz, temp_control.size(1), sentences_embed.size(2))
                temp_control = torch.cat([temp_control, expand_sentences_embed], dim=-1)

            if topic_embed is not None:
                expand_topic_embed = topic_embed.expand(bsz, temp_control.size(1), topic_embed.size(2))
                #expand_topic_embed = self.expand_topic(topic_embed).reshape(bsz, temp_control.size(1), topic_embed.size(2))
                temp_control = torch.cat([temp_control, expand_topic_embed], dim=-1)
        #print(temp_control, file=sys.stderr)
        #print(temp_control.size(), file=sys.stderr)
        #print(temp_control.size())
        #breakpoint()
        past_key_values = control_trans(temp_control) #bsz, seqlen, layer*emb
        #print(past_key_values.size())
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        #print(past_key_values[0].size())
        #print(past_key_values[1].size())
        return past_key_values

    def get_prompt_p5_infix(self, src, control_code=None, gpt2=None, bsz=None, attn_mask=None):
        # VERSION1. infixing by taking in the last layer of the hidden states as input.

        # VERSION2. infixing by pretending some input to first get the history, then add upon them.
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])


        temp_emb = self.wte2(input_tokens)
        src_emb = gpt2.transformer.wte(src)
        total_emb = torch.cat([src_emb, temp_emb], dim=1) #bsz, seqlen, dim
        src_out = gpt2(inputs_embeds=total_emb, attention_mask=attn_mask ,use_cache=True, return_dict=True)
        src_past_key_vals = src_out.past_key_values
        src_past_key_vals = torch.cat(src_past_key_vals, dim=0)
        # print(src_past_key_vals.shape, past_key_values.shape) # the src should be longer than past.
        # get a zero mask.
        _, src_len = src.shape
        nl, nb, nh, _, ndim = past_key_values.shape
        zero_mask = torch.zeros(nl, nb, nh, src_len, ndim).to(self.device)
        # print(zero_mask.shape, past_key_values.shape)
        past_key_values = torch.cat([zero_mask, past_key_values], dim=3)
        # print(past_key_values.shape)
        past_key_values = past_key_values + src_past_key_vals

        # add them together.
        past_key_values = past_key_values.split(2)

        return past_key_values

    def get_prompt_p6(self, control_code=None, gpt2=None, bsz=None):
        input_embs = self.input_embs.to(self.device)
        past_key_values = self.control_trans(input_embs).expand(bsz, -1, -1) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def get_prompt_p4(self, control_code, gpt2=None, bsz=None):
        # print(control_code, control_code.shape)
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = self.control_trans(temp_control).mean(1).unsqueeze(1) #bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values.shape
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def get_prompt_p1(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:

            if type(control_code) is tuple :
                assert False, 'Tuples'
                control_embs, control_word = control_code
                past_key_values = self.control_trans(control_embs)
                past_key_values = past_key_values.mean(1).unsqueeze(1)
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(bsz, seq_pastlen * self.preseqlen, self.match_n_layer * 2,
                                                       self.match_n_head,
                                                       self.match_n_embd)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
                print(control_word, control_embs.shape)
            else:
                # print('running with control code')
                # use the control code to generate the first 5 activation layers.
                if not self.embMatch:
                    if self.wte:
                        temp_control = self.wte(control_code)
                    else:
                        assert gpt2 is not None
                        temp_control = gpt2.transformer.wte(control_code)
                    temp_control = temp_control.sum(1).unsqueeze(1)
                else:
                    temp_control = control_code
                    # print(control_code.shape)
                past_key_values = self.control_trans(temp_control)
                # print(past_key_values.shape) #bsz, controlCodeLen, long... 5 * config.n_layer * 2 * config.n_embd
                past_key_values = past_key_values.sum(1).unsqueeze(1)
                # print(past_key_values.shape)  # bsz, 1, long...
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(bsz, seq_pastlen*self.preseqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def forward(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        src=None,
        tgt=None,
        input_attn=None,
        src_attn=None,
        tgt_attn=None,
        dis_labels=None,
        dis_src=None,
        dis_tgt=None,
        dis_src_attn=None,
        dis_tgt_attn=None,
        dis_distributions=None,
        alpha=1.0,
        **kwargs,
        ):
        #print(dis_labels)
        #sentence_encoder = self.sentence_encoder
        #topic_encoder = self.topic_encoder
        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}
        #print(dis_labels.size())
        #print(input_ids.size())
        #print("input devices")
        #print(input_ids.get_device())
        gpt2_model = self.gpt2_model
        #dis_labels = dis_labels.tolist()
        #print(input_ids.size())
        
        bsz = input_ids.shape[0] 
        #- self.classes_per_it * self.n_support
        #

        #print(input_ids)
        #print(labels)

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        else:
            #print ("past key values prompt")
            # print(dis_tgt)
            #if self.task_mode == 'understand_front':
            #    sentences_embed_list = []
            #    for dis_label in dis_labels:
            #        sentences_embed_list.append(self.proto_type_embed[dis_label])
            #    sentences_embed = torch.stack(sentences_embed_list).unsqueeze(1)
                #print(sentences_embed.size())
            #else:
            #print(dis_tgt)
            #dis_tgt = dis_tgt.to(self.device)
            #print (type(sentence_encoder))
            #print (sentence_encoder(dis_tgt))
            if self.task_mode == 'proto':
                target_cpu = torch.LongTensor(dis_labels).squeeze()

                input_cpu = dis_tgt.detach().clone()
                input_cpu_attn = dis_tgt_attn.detach().clone()

                label_input_cpu = dis_src.detach().clone()
                label_input_cpu_attn = dis_src_attn.detach().clone()
                #print(dis_src)
                label_dict = {}
                for indx, dis_label in enumerate(dis_labels):
                    label_dict[dis_label] = dis_src[indx].unsqueeze(0)

                def supp_idxs(c):
                    # FIXME when torch will support where as np
                    return target_cpu.eq(c).nonzero()[:self.n_support].squeeze(1)

                # FIXME when torch.unique will be available on cuda too
                classes = torch.unique(target_cpu)
                reverse_classes = {}
                for indx, label_id in enumerate(classes.tolist()):
                    reverse_classes[label_id] = indx
                #n_classes = len(classes)
                # FIXME when torch will support where as np
                # assuming n_query, n_target constants
                #n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support

                support_idxs = list(map(supp_idxs, classes))
                #print(support_idxs)
                #print(input_cpu[support_idxs[0]])
                if self.encoder_config['vae'] == 'ivae' and self.sentence_encoder.starting_flag:
                    prototypes = torch.stack([self.sentence_encoder(input_ids=input_cpu[idx_list], attn=input_cpu_attn[idx_list], src_input_ids = label_input_cpu[idx_list], src_attn = label_input_cpu_attn[idx_list], vae='auto').mean(0) for indx, idx_list in enumerate(support_idxs)])

                # FIXME when torch will support where as np
                #print(target_cpu)
                query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[self.n_support:], classes))).view(-1)

                n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support


                #print(query_idxs)
                input_ids = input_ids[query_idxs]
                labels = labels[query_idxs]
                #src = src[query_idxs]
                #tgt = tgt[query_idxs]
                #print(input_ids[0], file=sys.stderr)
                #print(labels[0], file=sys.stderr)
                src_attn = src_attn[query_idxs]
                tgt_attn = tgt_attn[query_idxs]
                dis_labels = [dis_labels[i] for i in query_idxs.tolist()]
                #dis_src = dis_src[query_idxs]
                dis_tgt = dis_tgt[query_idxs]
                dis_tgt_attn = dis_tgt_attn[query_idxs]

                dis_src = dis_src[query_idxs]
                dis_src_attn = dis_src_attn[query_idxs]

                dis_distributions = dis_distributions[query_idxs]

                #print(dis_tgt, file=sys.stderr)

                #if self.task_mode == 'proto_reg':
                    #topic_prototypes = []
                    #for indx, idx_list in enumerate(support_idxs):
                    #    if topic_encoder.vae:
                    #        logits, proto_mu, proto_z_var = topic_encoder(input_cpu[idx_list], attn=input_cpu_attn[idx_list])
                    #        topic_prototypes.append(logits.mean(0))
                    #    else:
                    #        logits = topic_encoder(input_cpu[idx_list], attn=input_cpu_attn[idx_list])
                            #logits = sentence_encoder(input_cpu[idx_list], attention_mask=input_cpu_attn[idx_list], return_dict=True).pooler_output
                            #if topic_encoder.topic_prior:
                            #proto_topic_loss, proto_topic_prior_embed, proto_mse_loss, proto_prob_topic = topic_encoder.topic_forward(logits, prototype=True)
                            #logits = proto_topic_prior_embed

                    #        topic_prototypes.append(logits.mean(0))
                    #topic_prototypes = torch.stack(topic_prototypes)

                #query_samples = input[query_idxs]




            if self.task_mode == 'no_hidden':
                if self.topic_encoder.vae == 'vae':
                    topic_embed, mu, z_var = self.topic_encoder(torch.cat([dis_src, dis_tgt], dim=1), attn=torch.cat([dis_src_attn, dis_tgt_attn], dim=1))
                    topic_embed = topic_embed.unsqueeze(1)
                else:
                    topic_embed = self.topic_encoder(dis_tgt, attn=dis_tgt_attn).unsqueeze(1)

                past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz,
                                                         topic_embed=topic_embed, wte=self.wte,control_trans=self.control_trans)
                loss_vae = 0
                if self.topic_encoder.vae  == 'vae':
                    loss_vae = loss_variational(mu, z_var, lambda_kl=1, reduction="sum")
            elif self.task_mode == 'casual_lens':
                if self.topic_encoder.vae == 'vae':
                    topic_embed, mu, z_var = self.topic_encoder(tgt, attn=tgt_attn)
                    topic_embed = topic_embed.unsqueeze(1)
                else:
                    topic_embed = self.topic_encoder(tgt, attn=tgt_attn).unsqueeze(1)

                style_embed = self.label_embed(dis_labels)
                if style_embed.dim() == 1:
                    style_embed = style_embed.unsqueeze(0).unsqueeze(1)
                else:
                    style_embed = style_embed.unsqueeze(1)

                past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz,
                                                         sentences_embed=style_embed,topic_embed=topic_embed, wte=self.wte,control_trans=self.control_trans)

            else:


                if self.task_mode in ['proto_reg', 'privacy']:
                    # label_mean_embed = torch.stack([prototypes[reverse_classes[label_id]] for label_id in dis_labels]).unsqueeze(1)
                    # sentences_embed = torch.zeros_like(sentences_embed)
                    label_loss = 0
                    if self.encoder_config['vae'] in ['ivae', 'vae', 'ivae_nocond', 'ivae_nocond_fix', 'vamp_vae']:
                        sentences_embed, label_mu, label_z_var = self.sentence_encoder(input_ids=dis_tgt,
                                                                                       attn=dis_tgt_attn,
                                                                                       src_input_ids=dis_src,
                                                                                       src_attn=dis_src_attn,
                                                                                       vae=self.encoder_config['vae'])
                    elif self.encoder_config['vae'] == 'auto':
                        sentences_embed = self.sentence_encoder(input_ids=dis_tgt, attn=dis_tgt_attn,
                                                                src_input_ids=dis_src,
                                                                src_attn=dis_src_attn, vae=self.encoder_config['vae'])
                    elif self.encoder_config['vae'] == 'vq_vae':
                        label_inputs, sentences_embed = self.sentence_encoder(input_ids=dis_tgt, attn=dis_tgt_attn,
                                                                              src_input_ids=dis_src,
                                                                              src_attn=dis_src_attn,
                                                                              vae=self.encoder_config['vae'])
                    elif self.encoder_config['vae'] == 'c_vae':
                        label_loss, sentences_embed = self.sentence_encoder(input_ids=dis_tgt, attn=dis_tgt_attn,
                                                                            src_input_ids=dis_src,
                                                                            src_attn=dis_src_attn,
                                                                            vae=self.encoder_config['vae'])
                    #elif self.encoder_config['vae'] == 'vamp_vae':
                    #    sentences_embed, label_mu, label_z_var = self.sentence_encoder.vamp_vae_forward(input_ids=dis_tgt, z2_logits = topic_embed.squeeze(), attn=dis_tgt_attn)

                    sentences_embed = sentences_embed.unsqueeze(1)

                    if self.encoder_config['vae'] in ['ivae', 'ivae_nocond', 'ivae_nocond_fix'] and self.sentence_encoder.starting_flag:
                        # label_phrase_embed = torch.stack([prototypes[reverse_classes[label_id]] for label_id in dis_labels]).unsqueeze(1)
                        #if self.encoder_config['vae'] == 'ivae':
                        label_phrase_embed = self.sentence_encoder(input_ids=dis_src, attn=dis_src_attn,
                                                                       src_input_ids=dis_src,
                                                                       src_attn=dis_src_attn,
                                                                       vae='auto').unsqueeze(1)
                        #elif self.encoder_config['vae'] == 'ivae_nocond':
                        #    label_phrase_embed = self.label_embed_ivae_nocond(dis_labels)

                        label_loss, _ = self.sentence_encoder.ivae_forward(label_mu.squeeze(),
                                                                           label_phrase_embed.squeeze(),
                                                                           log_var=label_z_var)
                    elif self.encoder_config['vae'] == 'vae':
                        label_loss = loss_variational(label_mu, label_z_var, lambda_kl=1, reduction="sum")
                    elif self.encoder_config['vae'] == 'vq_vae':
                        label_loss = self.sentence_encoder.vq_forward(label_inputs.squeeze(), sentences_embed.squeeze())
                    #elif self.encoder_config['vae'] == 'vamp_vae':
                    #    label_loss = self.topic_encoder.vamp_vae_loss(sentences_embed.squeeze(), label_mu, label_z_var, topic_embed.squeeze(), topic_mu, topic_z_var)

                    # sentences_embed = z_label_embed
                else:
                    sentences_embed = self.sentence_encoder(input_ids=dis_tgt, attn=dis_tgt_attn, src_input_ids=dis_src,
                                                            src_attn=dis_src_attn).unsqueeze(1)

                #print(dis_src)
                #print(sentence_encoder.iter)
                #sentence_encoder.iter = sentence_encoder.iter + 1
                    #print(dis_tgt, file=sys.stderr)
                    #sentences_embed = torch.zeros_like(sentences_embed)
                topic_loss = 0
                if self.encoder_config['vae'] in ['ivae', 'vae', 'ivae_nocond', 'ivae_nocond_fix']:
                    topic_embed, topic_mu, topic_z_var = self.topic_encoder(dis_tgt, attn=dis_tgt_attn)
                elif self.encoder_config['vae'] == 'auto':
                    topic_embed = self.topic_encoder(dis_tgt, attn=dis_tgt_attn)
                elif self.encoder_config['vae'] == 'vq_vae':
                    topic_input, topic_embed = self.topic_encoder(dis_tgt, attn=dis_tgt_attn)
                elif self.encoder_config['vae'] == 'c_vae':
                    topic_loss, topic_embed = self.topic_encoder(dis_tgt, attn=dis_tgt_attn)
                elif self.encoder_config['vae'] == 'vamp_vae':
                    topic_embed, topic_mu, topic_z_var = self.topic_encoder.vamp_vae_forward(input_ids=dis_tgt,
                                                                                                    z2_logits=sentences_embed.squeeze(),
                                                                                                    attn=dis_tgt_attn)

                #print(dis_tgt)
                # topic embedding
                topic_embed = topic_embed.squeeze().unsqueeze(1)
                #loss_vae = 0
                #if topic_encoder.vae:
                    #loss_vae = loss_variational(mu, z_var, lambda_kl=0.05, reduction="mean")


                # print(topic_encoder.topic_prior)
                #loss_topic_prior = 0
                #if self.topic_encoder.topic_prior:
                #    topic_loss, topic_prior_embed, mse_reg_loss, topic_prior = self.topic_encoder.VQ_forward(topic_embed, prototype=False)
                #    gumbel_topic_embed = topic_prior_embed.unsqueeze(1)
                    #print(torch.argmax(topic_prior,dim=1), file=sys.stderr)
                    #print("test")
                #    loss_topic_prior = self.topic_encoder.topic_word_forward(topic_prior, dis_distributions)
                    #print(torch.argmax(dis_distributions,dim=1), file=sys.stderr)
                    #print(torch.nonzero(dis_distributions), file=sys.stderr)
                    # print(topic_prior.size())
                    # print(dis_distributions.size())
                #else:
                if self.encoder_config['vae'] in ['ivae','ivae_nocond', 'ivae_nocond_fix'] and self.topic_encoder.starting_flag:
                    topic_loss, _, _, _ = self.topic_encoder.ivae_forward(topic_mu,log_var=topic_z_var)
                elif self.encoder_config['vae'] == 'vae':
                    topic_loss = loss_variational(topic_mu, topic_z_var, lambda_kl=1, reduction="sum")
                elif self.encoder_config['vae'] == 'vq_vae':
                    topic_loss = self.topic_encoder.vq_forward(topic_input.squeeze(), topic_embed.squeeze())
                elif self.encoder_config['vae'] == 'vamp_vae':
                    _, memory_mean, memory_logv = self.sentence_encoder.get_vamp_memory_and_var()
                    topic_loss = self.topic_encoder.vamp_vae_loss(topic_embed.squeeze(), topic_mu, topic_z_var, sentences_embed.squeeze(), label_mu, label_z_var, memory_mean, memory_logv)

                    #gumbel_topic_embed = topic_embed.unsqueeze(1)
                    #print(torch.argmax(topic_prior, dim=1), file=sys.stderr)
                ########



                #print(sentences_embed.size())
                #print(topic_embed.size())
                past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz, sentences_embed=sentences_embed, topic_embed=topic_embed, wte=self.wte,control_trans=self.control_trans)
                #topic_embed = gumbel_topic_embed.squeeze()


        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)
        else:
            prefix_len = past_key_values[0].size(3)
            prefix_mask = torch.ones(input_attn.size(0), prefix_len, dtype=torch.bool).to(input_attn.device)
            #print(self.device)
            #print(tgt_attn.device)
            attention_mask = torch.cat([prefix_mask, input_attn], dim=1)
            #print(attention_mask)
        #print(past_key_values_prompt.size())
        #print (input_ids.size())
        #print(input_ids, file=sys.stderr)
        #print(labels, file=sys.stderr)
        #print(input_ids.size())
        #print(attention_mask.size())
        #print(past_key_values[0].size())
        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, dis_labels=dis_labels, **kwargs)
        cofonder_loss = 0
        intervention_loss = 0
        if self.intervention == 1 and self.topic_encoder.starting_flag:

            indexes = torch.randperm(tgt.shape[0])
            permute_src = src[indexes]
            permute_tgt = tgt[indexes]
            permute_tgt_attn = tgt_attn[indexes]
            if dis_labels.dim() == 0:
                dis_labels = dis_labels.unsqueeze(0)
            permute_dis_labels = dis_labels[indexes]

            if self.topic_encoder.vae == 'vae':
                permute_topic_embed, permute_mu, permute_z_var = self.topic_encoder(permute_tgt, attn=permute_tgt_attn)
                permute_topic_embed = permute_topic_embed.unsqueeze(1)
            else:
                permute_topic_embed = self.topic_encoder(permute_tgt, attn=permute_tgt_attn).unsqueeze(1)

            permute_style_embed = self.label_embed(permute_dis_labels).unsqueeze(1)

            past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz,
                                                     sentences_embed=permute_style_embed, topic_embed=permute_topic_embed, wte=self.wte,
                                                     control_trans=self.control_trans)
            gpt2_model.eval()
            #breakpoint()
            output_sequences, logits = gpt2_model.generate(
                input_ids=permute_src,
                emb_match=None,
                control_code=None,
                past_key_values=past_key_values_prompt,
                max_length=128,
                min_length=10,
                top_p=0.9,  # top_p=0.5,
                do_sample=False,
                num_beams=1,
                bad_words_ids=[[628], [198]] if True else None,
                num_return_sequences=1,
                return_logits= True
            )
            agg_logits = torch.stack(logits, dim=1)
            gpt2_model.train()
            intervention_state = self.intervention_lstm.gumbel_forward(agg_logits)
            intervention_style_logits = self.intervention_linear(intervention_state)
            intervention_loss = F.cross_entropy(intervention_style_logits, permute_dis_labels)

            original_z = topic_embed.mean(dim=0)
            normalized_z = (original_z - original_z.min())/(original_z.max() - original_z.min())
            permute_z = permute_topic_embed.mean(dim=0)
            cofonder_loss = (normalized_z* torch.log(torch.clamp(torch.sigmoid(permute_z), min=1e-45)) + (1 - normalized_z)*torch.log(torch.clamp(1-torch.sigmoid(permute_z), min=1e-45))).mean()
            #breakpoint()
        if self.task_mode == 'no_hidden':
            return output, loss_vae
        elif self.task_mode == 'casual_lens':
            return output, intervention_loss + cofonder_loss
        else:
            #cos = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
            #batch_labels = torch.LongTensor(bsz).fill_(-1).to(self.device)

            #sentences_embed = sentences_embed.squeeze()
            #classes = classes.to(self.device)
            #sentences_embed = sentences_embed.squeeze()
            #print(cos(sentences_embed, topic_embed, batch_labels))
            #print(topic_encoder.topic_forward(topic_embed, alpha=alpha))

            entropy_loss = HLoss()
            #print(loss_topic_prior, file=sys.stderr)
            #print(topic_loss, file=sys.stderr)
            # loss_variational(mu, z_var, lambda_kl=1, reduction="mean")
            if self.task_mode in ['proto_reg', 'privacy']:
                #print(3 * prototypical_loss(topic_embed, topic_prototypes, classes, n_query, self.device, entropy_loss = entropy_loss))
                # prototypical_loss(torch.cat((sentences_embed, topic_embed), dim=0), torch.cat((prototypes, topic_prototypes), dim=0), classes, n_query, self.device, entropy_loss = None) +
                if self.encoder_config['disentangle_loss'] == 0:
                    disentangle_loss = prototypical_loss(topic_embed.squeeze(), sentences_embed.squeeze(), None, 0, self.device, entropy_loss = entropy_loss)
                elif self.encoder_config['disentangle_loss'] == 1:
                    squeeze_topic_embed = topic_embed.squeeze()
                    squeeze_sentences_embed = sentences_embed.squeeze()
                    #h = (torch.eye(bsz)-1/(bsz)).to(squeeze_sentences_embed.device)
                    kc = squeeze_topic_embed.matmul(squeeze_topic_embed.T)
                    lc = squeeze_sentences_embed.matmul(squeeze_sentences_embed.T)
                    disentangle_loss =  -0.0000001 *  hsic(kc,lc)
                elif self.encoder_config['disentangle_loss'] == 2:
                    disentangle_loss = 0
                elif self.encoder_config['disentangle_loss'] == 3:
                    squeeze_topic_embed = topic_embed.squeeze()
                    squeeze_sentences_embed = sentences_embed.squeeze()
                    disentangle_loss = -30 * MMD(squeeze_topic_embed, squeeze_sentences_embed)
                    #h = (torch.eye(bsz)-1/(bsz)).to(squeeze_sentences_embed.device)
                    #kc = squeeze_topic_embed.matmul(squeeze_topic_embed.T)
                    #lc = squeeze_sentences_embed.matmul(squeeze_sentences_embed.T)
                    #klc = squeeze_sentences_embed.matmul(squeeze_topic_embed.T)
                    #disentangle_loss = 0.05*(- kc - lc + 2*klc).mean()
                    #breakpoint()
                elif self.encoder_config['disentangle_loss'] == 5:
                    cos = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
                    batch_labels = torch.LongTensor(bsz).fill_(-1).to(topic_embed.device)
                    
                    squeeze_topic_embed = topic_embed.squeeze()
                    squeeze_sentences_embed = sentences_embed.squeeze()
                    disentangle_loss = cos(squeeze_sentences_embed, squeeze_topic_embed, batch_labels)
            #classes = classes.to(self.device)
            #sentences_embed = sentences_embed.squeeze()
            #print(cos(sentences_embed, topic_embed, batch_labels))
                #print("topic loss",file=sys.stderr)
                #print(topic_loss, file=sys.stderr)
                #print("proto loss",file=sys.stderr)
                #print(proto_loss, file=sys.stderr)
                if self.encoder_config['disentangle_loss'] == 4:
                    #print(type(dis_labels))
                    return_embed = None
                    if self.encoder_config['vae'] in ['ivae', 'vae', 'ivae_nocond', 'ivae_nocond_fix']:
                        return_embed = topic_mu
                    elif self.encoder_config['vae'] == 'auto':
                        return_embed = topic_embed
                    elif self.encoder_config['vae'] == 'vq_vae':
                        return_embed = topic_input
                    elif self.encoder_config['vae'] == 'c_vae':
                        return_embed = topic_embed
                        
                    return output, topic_loss + label_loss, return_embed.squeeze(), dis_labels
                else:
                    return output, topic_loss + disentangle_loss + label_loss #+ prototypical_loss(gumbel_topic_embed.squeeze(), prototypes, classes, n_query, self.device, entropy_loss = entropy_loss) #topic_loss + prototypical_loss(gumbel_topic_embed.squeeze(), topic_prototypes, classes, n_query, self.device, entropy_loss = entropy_loss) + loss_topic_prior
            else:
                #topic_embed = topic_embed.squeeze()
                return output, topic_loss



    def forward_infix(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1) # bsz, seqlen
        else:
            infix_attn = torch.ones(bsz, self.preseqlen).bool().to(self.device)
            attention_mask = torch.cat([src_attn, infix_attn, tgt_attn], dim=1)  # bsz, seqlen
            partial_attn_mask = torch.cat([src_attn, infix_attn], dim=1)  # bsz, seqlen
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz, attn_mask=partial_attn_mask)
            # print(src_attn)
            # print()
            # print(infix_attn)
            # infix_attn = torch.ones(bsz, self.preseqlen).to(self.device)
            # attention_mask = torch.cat([src_attn, infix_attn, tgt_attn], dim=1)  # bsz, seqlen

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"


        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output



class PrefixEmbTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False):
        super().__init__(config)

        print('under the PrefixEmbTuning model')

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0


        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512


        if hasattr(config, 'parametrize_emb'):
            self.parametrize_emb = config.parametrize_emb
        else:
            self.parametrize_emb = 'MLP'



        # if hasattr(config, 'mid_layers'):
        #     self.mid_layers = config.mid_layers
        # else:
        #     self.mid_layers = 1

        if self.task_mode == 'dataless':
            self.mode_para = 1
        elif self.task_mode == 'data2text' or self.task_mode == 'triples' or self.task_mode == 'webnlg' or \
                self.task_mode == 'writingPrompts' or self.task_mode == 'summarization':
            # with src and input based encoding.
            self.mode_para = 2
            # self.mode_para=0 and optim_prefix == True for Instruction based.
        else:
            self.mode_para = 4


        if not self.optim_prefix:
            if self.train_weights:
                self.wte = model_gpt2.transformer.wte
                for p in self.wte.parameters():
                    p.requires_grad = True
            else:
                if not self.init_random:
                    self.wte = None
                else:
                    print('the is just for baseline checking!!! We reinitialize the LM embeddings and try cat '
                          'and peek.')
                    print('BASELINE'*100)
                    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
                    print(self.wte)



            if self.mode_para == 1:
                print('mode_para=1, for dataless.')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p4_infix
                else:
                    self.get_prompt = self.get_prompt_p4
            elif self.mode_para == 2 or self.mode_para == 4:
                print('mode_para=2 or 4, for (2)data2text having a variable length input prefix parametrization. or for (4) topic/keyword/attributes...')

                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p3_infix
                else:
                    self.get_prompt = self.get_prompt_p3

        else:
            self.mode_para = 0
            print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
            print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))

            # DIFFERENT PARAMETRIZATION:
            if True:
                if self.parametrize_emb == 'MLP':
                    print('MLP: UNDER PARAMETRIZATION 1 FOR embeddings. With the mid_dim = {}'.format(self.mid_dim))
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_embd))
                    if self.use_infix:
                        self.get_prompt = self.get_prompt_p5_infix
                    else:
                        self.get_prompt = self.get_prompt_p5
                elif self.parametrize_emb == 'Emb':
                    print('Emb: UNDER PARAMETRIZATION 2 FOR embeddings.')
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)

                    if self.use_infix:
                        self.get_prompt = self.get_prompt_p7_infix
                    else:
                        self.get_prompt = self.get_prompt_p7


            # DIFFERENT PARAMETRIZATION 2.
            elif True:
                print('UNDER PARAMETRIZATION 2')
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                input_word_lst = [['name', 'Type', 'price', 'customer rating', 'near', 'area', 'family friendly']]
                input_word_ids = tokenizer(input_word_lst, add_special_tokens=True, is_split_into_words=True, return_tensors='pt')['input_ids']
                self.input_embs = model_gpt2.transformer.wte(input_word_ids.to(model_gpt2.device))
                print(self.input_embs.shape)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p6_infix
                else:
                    self.get_prompt = self.get_prompt_p6



            # OLD CODE.
            # self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
            # if self.use_infix:
            #     assert False, "just optimizing a set of parameter is not really related to infix position."
            #     self.get_prompt = self.get_prompt_p2_infix
            # else:
            #     self.get_prompt = self.get_prompt_p2

        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.forward = self.forward_infix

        ###### just trying #########
        total_param = 0
        trainable_param = 0
        for name, param in self.named_parameters():
            #print(param.shape)
            if param.requires_grad == True:
                trainable_param += param.numel()

            total_param += param.numel()
        print('total param is {}'.format(total_param))
        print('total trainable param is {}'.format(trainable_param))


        ############################################################################



    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        '''
        Directly specifying/optimizing the input embeddings.
        :param control_code:
        :param gpt2:
        :param bsz:
        :return:
        '''
        assert bsz is not None
        temp_control = self.control_trans.unsqueeze(0).expand(bsz, -1, -1) #bsz, seqlen, emb
        temp_control = self.dropout(temp_control)
        temp_result = gpt2(inputs_embeds=temp_control, use_cache=True)
        past_key_values = temp_result.past_key_values
        return past_key_values

    def get_prompt_p2_infix(self, src_x, control_code=None, gpt2=None, bsz=None):
        '''
        Directly specifying/optimizing the input embeddings.
        :param control_code:
        :param gpt2:
        :param bsz:
        :return:
        '''
        assert bsz is not None
        temp_control = self.control_trans.unsqueeze(0).expand(bsz, -1, -1) #bsz, seqlen, emb
        temp_control = self.dropout(temp_control)
        src_embs = gpt2.wte(src_x)
        print(temp_control.shape, src_embs.shape)
        temp_control = torch.cat([src_embs, temp_control], dim=1)
        print(temp_control.shape)
        temp_result = gpt2(inputs_embeds=temp_control, use_cache=True)
        past_key_values = temp_result.past_key_values
        return past_key_values


    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        input_embs = self.control_trans(temp_control) #bsz, seqlen, emb_dim
        bsz, seqlen, _ = input_embs.shape
        input_embs = self.dropout(input_embs)
        temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        past_key_values = temp_result.past_key_values
        return past_key_values


    def get_prompt_p7(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        input_embs = self.wte(input_tokens)
        bsz, seqlen, _ = input_embs.shape
        input_embs = self.dropout(input_embs)
        temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        past_key_values = temp_result.past_key_values
        return past_key_values



    def get_prompt_p3_infix(self, src_x, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb

            src_embs = gpt2.transformer.wte(src_x)
            input_embs = self.control_trans(temp_control) #bsz, seqlen, emb
            input_embs = self.dropout(input_embs)
            input_embs = torch.cat([src_embs, input_embs], dim=1)
            # print(input_embs.shape)
            bsz, seqlen, _ = input_embs.shape
            # print(past_key_values.shape)
            temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
            past_key_values = temp_result.past_key_values
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values


    def get_prompt_p3(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            input_embs = self.control_trans(temp_control) #bsz, seqlen, emb
            input_embs = self.dropout(input_embs)
            bsz, seqlen, _ = input_embs.shape
            # print(past_key_values.shape)
            temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
            past_key_values = temp_result.past_key_values
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values


    def get_prompt_p4(self, control_code, gpt2=None, bsz=None):
        # print(control_code, control_code.shape)
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            input_embs = self.control_trans(temp_control)  # bsz, seqlen, emb
            input_embs = self.dropout(input_embs)
            bsz, seqlen, _ = input_embs.shape
            # print(past_key_values.shape)
            temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
            past_key_values = temp_result.past_key_values
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def forward_infix(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]
        # TODO-LISA
        self.format_mode = 'cat'
        if self.mode_para == 2:
            if self.format_mode == 'cat':
                past_key_values_prompt = self.get_prompt(src, cate_batch, gpt2=gpt2_model, bsz=bsz)
                attention_mask = torch.cat([src_attn, cate_attn, tgt_attn], dim=1)
            else:
                past_key_values_prompt = self.get_prompt(src, src, gpt2=gpt2_model, bsz=bsz)
                attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1)
        else:

            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            bsz, seqlen = src.shape
            temp_attn = torch.ones(bsz, self.preseqlen).bool()
            attention_mask = torch.cat([src_attn, temp_attn, tgt_attn], dim=1)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        # if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
        #     attention_mask = torch.cat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output

    def forward(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        else:
            past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)

        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output





