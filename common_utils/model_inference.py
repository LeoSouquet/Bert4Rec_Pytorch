from common_utils.torch_utils import select_device
import logging
from models.Bert_Model import BERTModel
import torch
import numpy as np
import json

from common_utils.general import load_file_moviesids_to_title

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s >_ %(message)s')
logger = logging.getLogger('Pytorch Bert4Rec Model Inference')


class Bert4Rec_inference():
    def __init__(self, metadata):

        self.device = select_device(metadata.device)
        self.max_len = metadata.max_len


        self.model = BERTModel(metadata, inference = True)
        self.model = self.model.to(self.device)

        self.close_mark = metadata.CLOZE_MARK

        if metadata.init_weights != "NONE": 
            self.model.load_state_dict(torch.load(metadata.init_weights,map_location=self.device))
        self.model.eval()
        
        logger.info(f"Loading model at path : {metadata.init_weights}")

        self.moviesidx_to_title = load_file_moviesids_to_title(metadata.file_idmovies_to_title)
        logger.info(f"There is {len(self.moviesidx_to_title)} listed")

        self.warmup()


    def warmup(self):
        """ 
        Function used to warmup the model (perform 10 Inferences) with a generated fake sequence.
        """
        #Perform 10 Inferences on fake images to allow to speed up the model.
        for i in range(0,10):
            seq = torch.zeros((1, self.max_len),dtype = torch.int64, device=self.device)  # init img
            _ = self.model(seq)
            
        logger.info(f"Model Warmed Up")


    def pre_processing(self,id_movies_batch):
        ##Need to integrate the fact that the 
        """ 
        Function pre-process the image_input to the proper format required by the model.
        """

        if not isinstance(id_movies_batch[0], list):
            id_movies_batch = [id_movies_batch]

        batch = []
        #Create batch
        for sequence in id_movies_batch:
            #Get the max-1 latest in history (in case history is larger then max_len)
            new_id = sequence[-self.max_len+1:]

            #Compute pad len
            pad_len = (self.max_len-1) - len(new_id)

            #Create Sequence input
            input = [0] * pad_len + new_id + [self.close_mark]

            input = torch.tensor(input, dtype=torch.long)
            batch.append(input)

        return torch.stack(batch)

    def inference(self,input_sequence, top_k = 5):
        """ 
        Function to perform the inference of the model using the sequence as input
        """
        with torch.no_grad():
            self.model.change_top_k(top_k)
            pred = self.model(input_sequence)
    
        return pred

    def post_processing(self, y_pred):
        """ 
        Function to perform the post_processing:
        Sends back the top_k best predictions
        """


        """
        This has been replaced by the Post_Processing Layer :) 
        focused_pred = y_pred[0, -1].numpy()
        res = np.argsort(focused_pred).tolist()[::-1]
        """

        res_final = []
        for item in y_pred:
            res = [self.moviesidx_to_title[str(movie_title)] for movie_title in item.numpy() if str(movie_title) in self.moviesidx_to_title.keys()]
            res_final.append(res)

        return res_final

    def detect(self,seq_input, top_k = 5):
        """ 
        Function to perform the full inference
        """

        seq_input = self.pre_processing(seq_input)
        predictions = self.inference(seq_input, top_k)
        return self.post_processing(predictions)