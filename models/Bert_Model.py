from .bert_modules.bert import BERT
from .bert_modules.utils.post_pre_process import  Post_Processing
import torch.nn as nn

class BERTModel(nn.Module):
    def __init__(self, metadata, inference = False):
        super().__init__()
        self.inference = inference

        if inference:
            self.post_process = Post_Processing(metadata.top_k)

        self.bert = BERT(metadata.model_init_seed,metadata.max_len, metadata.word_size,metadata.bert_num_blocks, metadata.bert_num_heads, metadata.bert_hidden_units, metadata.bert_dropout)
        
        self.out = nn.Linear(self.bert.hidden, metadata.word_size)

    def forward(self, x):
        

        x = self.bert(x)
        x = self.out(x)

        if self.inference:
            x = self.post_process(x)
        
        return x


    def change_top_k(self, new_top_k):
        self.post_process.top_k = new_top_k
