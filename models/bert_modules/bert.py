from torch import nn as nn

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from common_utils.general import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, model_init_seed, max_len, word_size, nb_layers, nb_heads, nb_hidden, bert_dropout):
        super().__init__()

        fix_random_seed_as(model_init_seed) #Need to Fix This too 

        self.max_len = max_len
        self.num_items = word_size
                
        self.heads = nb_heads
        self.vocab_size = self.num_items + 2 #This takes into account the CLOZE Variable being the word_size + 1 Value. Ex: if Vocab is 10 (from 0 to 9) if CLOZE is 11 we need from 0 to 11, hence 12 (10 + 2)
         

        self.hidden = nb_hidden

        self.dropout = bert_dropout

        # embedding for BERT, sum of positional, token embeddings
        self.embedding = BERTEmbedding(vocab_size=self.vocab_size, embed_size=self.hidden, max_len=self.max_len, dropout=self.dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout) for _ in range(nb_layers)])

    def forward(self, x):

        #Need to check this too
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
    
    """
    def init_weights(self):
        print("IS THIS USED?")
        pass
    """