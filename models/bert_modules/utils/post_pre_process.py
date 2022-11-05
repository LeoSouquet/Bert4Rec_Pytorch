import torch.nn as nn
import torch

class Post_Processing(nn.Module):
    """
    BERT Post_Processing
    """

    def __init__(self, top_k):

        super().__init__()

        self.top_k = top_k

    def forward(self, y_pred):
        
        focused_pred = y_pred[:, -1]
        res = torch.argsort(focused_pred,dim=1)
        reversed_res = torch.flip(res,[0,1])

        return reversed_res[:,:self.top_k]