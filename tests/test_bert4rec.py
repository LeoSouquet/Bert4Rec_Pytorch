from datasets.custum_datasets import My_DataSet
import common_utils.metadata as metadata
from common_utils.model_inference import Bert4Rec_inference


import unittest
import logging

#Format the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s >_ %(message)s')
logger = logging.getLogger('Test API')



class Test_Bert4Rec(unittest.TestCase):
    """ 
    Class used to perform API Tests.
    """
        
    def test_dataset(self):
        """
        Test the DataSet
        """
        
        data = My_DataSet(metadata)
        seq , labels = data[0]

        self.assertEqual(seq.shape, labels.shape)
        self.assertEqual(len(seq), metadata.max_len)

        logger.info("Test the DataSet Completed")
        logger.info("------")


    def test_inference(self):

        metadata.top_k = 15
        model = Bert4Rec_inference(metadata) 

        seg_1 = list(range(1, 2))
        seg_2 = list(range(1, 3))

        res = model.detect([seg_1,seg_2], metadata.top_k)
        self.assertEqual(len(res), 2)

        self.assertEqual(len(res[0]), metadata.top_k)

        new_top_k = 18
        res = model.detect([seg_1,seg_2], new_top_k)

        self.assertEqual(len(res[0]), new_top_k)

        logger.info("Test the Inference Completed")
        logger.info("------")
        
        



if __name__ == '__main__':

    unittest.main()