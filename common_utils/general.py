import json
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn

def load_file_moviesids_to_title(path_to_file):

                # Opening JSON file
        f = open(path_to_file)

        # returns JSON object as 
        # a dictionary
        data = json.load(f)


        f.close()
        return data
        
def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
