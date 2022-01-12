# encoding=utf-8
import torch
from torch import nn
import os, sys
sys.path.append('bert_model')
from embedding import TokenEmbedding
from config import *
from bert_layer import BertLayer
from LayerNorm import clones, Classify


class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.all_layers = []
        self.token_embedding = TokenEmbedding()
        self.bert_layer = BertLayer()
        self.classify = Classify(args.hidden_size, args.classify)

    def get_src_mask(self, seq, pad_idx):
        src_mask = (seq != pad_idx).unsqueeze(1)
        return src_mask.int()

    def get_trg_mask(self, trg, pad_idx):
        batch, trg_len = trg.size()
        trg_mask = (trg != pad_idx).unsqueeze(-2)
        trg_mask = trg_mask & (1 - torch.triu(torch.ones(1, trg_len, trg_len), diagonal=1))
        return trg_mask

    def load_pre_model(self, model_path, is_pre_model=True):
        if is_pre_model:
            state_dict = torch.load(model_path)
            new_state_dic = {}
            model_state_dic = self.state_dict()
            for old_key, new_key in zip(args.old_key, args.new_key):
                if '##' in old_key:
                    for i in range(args.num_layers):
                        new_key1 = str(new_key).replace('##', str(i))
                        old_key1 = str(old_key).replace('##', str(i))
                        new_state_dic[new_key1] = state_dict[old_key1]
                else:
                    new_state_dic[new_key] = state_dict[old_key]
            model_state_dic.update(new_state_dic)
            self.load_state_dict(model_state_dic)
            print('pretrain model load successful')
        else:
            self.load_state_dict(torch.load(model_path))
            print('finetune model load successful')

    def forward(self, input_ids, attention_mask=None, return_all_layers=True):
        input_embedding = self.token_embedding(input_ids)
        if attention_mask is None:
            attention_mask = self.get_src_mask(input_ids, args.pad_idx)

        for i in range(args.num_layers):
            layer = self.bert_layer
            input_embedding = layer(input_embedding, attention_mask)

            if return_all_layers:
                self.all_layers.append(input_embedding)
        if not return_all_layers:
            self.all_layers.append(input_embedding)

        return self.classify(input_embedding), self.all_layers


if __name__ == '__main__':
    input_ids = torch.arange(10).view(2, 5)
    pp, _ = BertModel()(input_ids)
    print(pp.size(), pp.type())












