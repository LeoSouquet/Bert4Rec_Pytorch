import torch.nn as nn



class ComputeLoss:

    # Compute losses
    def __init__(self, model):

        self.device = next(model.parameters()).device  # get model device

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)


    def __call__(self, predictions, labels):  # predictions, targets

        #Flatten predictions and labels
        predictions = predictions.view(-1, predictions.size(-1))  
        labels = labels.view(-1)  

        #Only Compute the loss on the modified labels
        mask = labels > 0
        labels = labels[mask]
        predictions = predictions[mask]
        
        loss = self.loss_function(predictions, labels)

        return loss
    