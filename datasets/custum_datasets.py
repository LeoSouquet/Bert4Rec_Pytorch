from torch.utils.data import Dataset

import random
import torch

import numpy as np

import glob, os

import pickle

class My_DataSet(Dataset):

    def __init__(self, metadata):

        print(metadata.path_to_record)

        
        files_paths = glob.glob(os.path.join(metadata.path_to_record,"*.tfrecord"))

        self.full_sorted_dataset = {}

        self.uidx_to_uid = {}


        self.mask_token = 40857 + 1 
        self.max_len = 100
        self.mask_prob = 0.15


        #For the random selection of MASK
        seed = 1337
        self.rng = random.Random(seed)

        self.min_nb_movies = 2

        self.all_movies = []
        path_to_cache = os.path.join(metadata.path_to_record, "data.cache")
        if os.path.exists(path_to_cache):

            print(f"Reading data from cache file : {path_to_cache}")
            
            cache_file = open(path_to_cache, "rb")
            cache_data = pickle.load(cache_file)
            cache_file.close()

            self.full_sorted_dataset = cache_data["all_data"]
            self.uidx_to_uid = cache_data["uidx_to_uid"]
            self.all_movies = cache_data["all_movies"]

            print(f"{len(self.full_sorted_dataset)} users have been found from cache data")
            print(f"{len(self.all_movies)} movies have been found from cache data")

        else:
            
            #Only imports tensorflow if data need to be read
            import tensorflow as tf
            
            # Read the data back out.
            #I decided to put this function here because it is very specific to this data set.
            def decode_fn(record_bytes):

                schema = {
                    "userIndex": tf.io.FixedLenFeature([], tf.int64),
                    "movieIndices": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
                    "timestamps": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64)
                }

                return tf.io.parse_single_example(record_bytes,schema)

            #Collect all Data Set from different file
            data_set = {} 
            for file in files_paths:
                print(f"Processing {file}")

                dataset = tf.data.TFRecordDataset(file)
                for _ , batch in enumerate(dataset.map(decode_fn)):

                    #Refactor with variables :) 

                    

                    #Check if a a session is only composed of one movie
                    if len(batch["movieIndices"].numpy()) >= self.min_nb_movies: 

                        usr_id = batch["userIndex"].numpy()

                        if usr_id not in data_set.keys():
                            data_set[usr_id] = {"movieIndices": batch["movieIndices"].numpy(), "timestamps":batch["timestamps"].numpy()}
                        else:
            
                            data_set[usr_id]["movieIndices"] = np.concatenate((data_set[usr_id]["movieIndices"] , batch["movieIndices"].numpy()),axis=0)
                            data_set[usr_id]["timestamps"] = np.concatenate((data_set[usr_id]["timestamps"] ,  batch["timestamps"].numpy()),axis=0)

            

            
            #Now for all users collected, the objetive is to sort the sequence by timetamps and also compute the number of unique movies collected
            for index, user_id in enumerate(data_set.keys()):

                self.uidx_to_uid[index] = user_id

                sorted_timestamps_ids = np.argsort(data_set[user_id]["timestamps"])

                new_movies = [a for a in np.unique(data_set[user_id]["movieIndices"]) if a not in self.all_movies]
                if len(new_movies) > 0:
                    self.all_movies = np.concatenate((new_movies,self.all_movies))
                
                self.full_sorted_dataset[user_id] = data_set[user_id]["movieIndices"][sorted_timestamps_ids]

            print("-----")
            print(f"There is {len(self.full_sorted_dataset)} number of users")
            print(f"There is {len(self.all_movies)} movies after")

            #Cache the Data for future-reuse
            
            cache = {"all_data" : self.full_sorted_dataset, "uidx_to_uid" : self.uidx_to_uid , "all_movies": self.all_movies}

            cache_file = open(path_to_cache, 'wb')
            pickle.dump(cache, cache_file)
            cache_file.close()

            print(f"Data has been chached in {path_to_cache}")

        
    
    def __len__(self):
        return len(self.full_sorted_dataset)

    def __getitem__(self, idx):

        udix_user = self.uidx_to_uid[idx] 

        seq = self.full_sorted_dataset[udix_user]

        tokens = []
        labels = []

        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                tokens.append(self.mask_token)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        

        return torch.LongTensor(tokens), torch.LongTensor(labels)

