import os
from typing import List, Dict, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

from .base_encoder import base_encoder
from ..module import embedding_layer, lstm_layer
from ..base_model import base_model

class bert_encoder(base_model):

    def __init__(self, max_length = 128, hidden_size = 768, config = None):
        super(bert_encoder, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_size = hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '.cache/huggingface'))
        self.encoder_layer = AutoModel.from_pretrained('bert-base-uncased', cache_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            '.cache/huggingface'))

    def forward(self, inputs, lengths = None):
        inputs = self._tensorize_batch(inputs, self.tokenizer)
        inputs = inputs.to(self.config['device'])
        x = self.encoder_layer(inputs, return_dict=True).pooler_output
        return x

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]], tokenizer
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
