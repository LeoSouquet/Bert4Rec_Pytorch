
import os

#General Parameters
model_init_seed = int(os.environ.get('MODEL_INIT_SEED',100))
init_weights =  os.environ.get('INIT_WEIGHT','model.pth')

#Data Preparation
path_to_record = os.environ.get('PATH_TO_RECORD','data\\') 
file_idmovies_to_title = os.environ.get('FILE_IDMOVIES_TO_TITLE','data\\movie_title_by_index.json')
word_size = int(os.environ.get('WORD_SIZE',40857))  #To be Determined Automatically
CLOZE_MARK = word_size + 1
max_len = int(os.environ.get('MAX_LEN',100))


#Training 
batch_size = int(os.environ.get('BATCH_SIZE',128))
device = os.environ.get('DEVICE','cpu')  
epochs = int(os.environ.get('EPOCHS',300))
lr = float(os.environ.get('LR',0.001))   
decay_step = int(os.environ.get('DECAY_STEPS',25))   
weight_decay = float(os.environ.get('WEIGHT_DECAY',0.001))  
gamma = float(os.environ.get('GAMMA',1.0))  


#Inference
top_k = int(os.environ.get('TOP_K',5))


#Bert Architecture
bert_num_blocks = int(os.environ.get('BERT_NUM_BLOCK',2))
bert_num_heads = int(os.environ.get('BERT_NUM_HEADS',4)) 
bert_hidden_units = int(os.environ.get('BERT_NUM_HIDDEN_UNITS',256))  
bert_dropout = float(os.environ.get('DROP_OUT',0.1))   


