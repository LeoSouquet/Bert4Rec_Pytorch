import tensorflow as tf
import pandas as pd 
import numpy as np

path_to_record = "C:\\Users\\Leo Souquet\\Downloads\\recommendation\\recommendation\\dataset\\data-1.tfrecord"

dataset = tf.data.TFRecordDataset(path_to_record)

users_id = []
# Read the data back out.
def decode_fn(record_bytes):

  schema = {
    "userIndex": tf.io.FixedLenFeature([], tf.int64),
    "movieIndices": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
    "timestamps": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64)
  }

  return tf.io.parse_single_example(record_bytes,schema)

def foo_bar(x):
    return x.numpy()
"""
#Better Solution :) 
for batch in dataset.map(decode_fn):
  print(batch["userIndex"].numpy())
  print(batch["movieIndices"].numpy())
  print(batch["timestamps"].numpy())
"""

data_frame = pd.DataFrame(dataset.map(decode_fn))
data_frame = data_frame.applymap(foo_bar)
#Not that Good :) 
print(len(data_frame))
print(data_frame.head(5))