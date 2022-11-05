import torch
from models.Bert_Model import BERTModel
from datasets.custum_datasets import My_DataSet
import torch.utils.data as data_utils
import torch.optim as optim
from tqdm import tqdm
import numpy as np


from common_utils.loss import ComputeLoss
from common_utils.torch_utils import select_device


def calculate_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted sequence

    Args:
        y_pred (T.tensor): Prediction from the model
        y_true (T.tensor): Ground Truth

    Returns:
        T.tensor: Prediction Accuracy
    """
    
    mask = y_true > 0
    _, prediction = y_pred.max(2)
    prediction = prediction[mask]
    y_true = y_true[mask]

    #Alternative to the lines above :) 
    #More optimized if done on GPU.

    #y_true = torch.masked_select(y_true, mask)
    #prediction = torch.masked_select(prediction, mask)

    return (y_true == prediction).double().mean()


def train(metadata):
   
    dataset = My_DataSet(metadata)

    #Get Data Loader
    dataloader_training = data_utils.DataLoader(dataset, batch_size=metadata.batch_size,shuffle=True, pin_memory=True)

    device = select_device(metadata.device)

    #Create Model
    model = BERTModel(metadata)
    model = model.to(device)

    if metadata.init_weights != "NONE":
        model.load_state_dict(torch.load(metadata.init_weights,map_location=device))

    #Create Loss Function
    compute_loss = ComputeLoss(model)


    #Create Optimiser
    optimizer = optim.Adam(model.parameters(), lr=metadata.lr, weight_decay=metadata.weight_decay)

    #Create Scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=metadata.decay_step, gamma=metadata.gamma)

    for epoch in range(metadata.epochs):  # epoch ----
        
        model.train()

        tqdm_dataloader = tqdm(dataloader_training)

        losses = torch.zeros(1, dtype=torch.float64)
        accuracies = torch.zeros(1, dtype=torch.float64)

        
        for iteration , batch in enumerate(tqdm_dataloader):

            batch = [x.to(device) for x in batch]

            optimizer.zero_grad()

            seqs, labels = batch
            predictions = model(seqs)  

            loss = compute_loss(predictions, labels)
            loss.backward()
            optimizer.step()

            accuracy = calculate_accuracy(predictions, labels)

            losses += loss #Check because not sure about the detach
            accuracies += accuracy

            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f}, Acc: {:.3f} '.format(epoch+1, (losses/(iteration+1)).item(), (accuracies / (iteration+1)).item()))

        #Move on step into the scheduler function
        lr_scheduler.step()

        #Save Model
        torch.save(model.state_dict(), f"model_{epoch}.pth")


if __name__ == '__main__':
    import common_utils.metadata as metadata
    train(metadata)     