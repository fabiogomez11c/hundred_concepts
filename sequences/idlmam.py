import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tqdm.autonotebook import tqdm


import os
from uuid import uuid4
from typing import Dict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

import time

def visualize2DSoftmax(X, y, model):
    """Function to visualize the classification boundary of a learned model on a 2-D dataset

    Arguments:
    X -- a numpy array of shape (2, N), where N is the number of data points. 
    y -- a numpy array of shape (N,), which contains values of either "0" or "1" for two different classes
    model -- a PyTorch Module object that represents a classifer to visualize. s
    """
    x_min = np.min(X[:,0])-0.5
    x_max = np.max(X[:,0])+0.5
    y_min = np.min(X[:,1])-0.5
    y_max = np.max(X[:,1])+0.5
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=20), np.linspace(y_min, y_max, num=20), indexing='ij')
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    with torch.no_grad():
        preds = model(torch.tensor(xy_v, dtype=torch.float32))
        preds = F.softmax(preds, dim=1).numpy()

    cs = plt.contourf(xv, yv, preds[:,0].reshape(20,20), levels=np.linspace(0,1,num=20), cmap=plt.cm.RdYlBu)
    ax = plt.gca()
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=ax)

def run_epoch(model,
    optimizer,
    data_loader,
    loss_func,
    device,
    results,
    score_funcs,
    prefix="",
    problem_type='classification'
):
    # init empty variables
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    # loop over the batchs
    for inputs, labels in data_loader:
        # move x, y to correct device
        inputs = moveTo(inputs, device=device)
        labels = moveTo(labels, device=device)

        # estimate
        y_hat = model(inputs)

        # compute loss, backpropagate and apply gradients if needed
        loss = loss_func(y_hat, labels)
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # store the results
        running_loss.append(loss.item())
        if score_funcs is not None and len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
        
    end = time.time()

    # clean results to compute metrics under sklearn standards
    y_pred = np.asarray(y_pred)
    if problem_type == 'classification':
        if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        elif y_pred.shape[1] == 1:
            y_pred = np.where(y_pred > 0.5, 1, 0)
        
    
    results[prefix + ' loss'].append(np.mean(running_loss))
    if score_funcs is not None:
        for name, score_func in score_funcs.items():
            try:
                results[prefix + " " + name].append( score_func(y_true, y_pred) )
            except:
                results[prefix + " " + name].append(float("NaN"))
    return end-start #time spent on epoc

def save_model(
    model: nn.Module,
    checkpoint_file: str,
    optimizer: nn.Module,
    true_epoch: int,
    results: Dict,
    model_file_name: str = None
):
    # check if the folder for models exists
    folder_path = "checkpoints"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # save the model checkpoint
    file_name = uuid4().hex + '_' + checkpoint_file
    if model_file_name is not None:
        file_name = model_file_name
    torch.save({
        'epoch': true_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results' : results
        }, os.path.join(folder_path, file_name))

def train_simple_network(
        model, 
        loss_func,
        train_loader,
        test_loader=None,
        score_funcs=None,
        device='cpu',
        epochs=50,
        checkpoint_file=None,
        prev_results=None,
        checkpoint_dict=None,
        checkpoint_every_x=1,
        problem_type='classification'
    ):
    # init tracking metrics
    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")
    if score_funcs is not None:
        for eval_score in score_funcs:
            to_track.append("train " + eval_score )
            if test_loader is not None:
                to_track.append("test " + eval_score )
    total_train_time = 0
    if prev_results is None:
        results = {}
        for item in to_track:
            results[item] = []
    else:
        results = prev_results

    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    if checkpoint_dict is not None and 'optimizer_state_dict' in checkpoint_dict.keys():
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # start training loop
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch"):
        # pass the model throught a epoch
        model = model.train()
        total_train_time += run_epoch(
            model, 
            optimizer, 
            train_loader, 
            loss_func, 
            device, 
            results, 
            score_funcs, 
            prefix="train",
            problem_type=problem_type
        )
    
        # grab more results from the last epoch
        results["total time"].append( total_train_time )
        results["epoch"].append( epoch )

        # test the model in validation
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test")

        # save the model checkpoint after the epoch
        true_epoch = len(results['epoch']) - 1 # useful when model and optimizer loaded
        if checkpoint_file is not None and (true_epoch % checkpoint_every_x) == 0:
            save_model(
                model=model,
                checkpoint_file=checkpoint_file,
                optimizer=optimizer,
                true_epoch=true_epoch,
                results=results
            )
    
    # save the last model
    if checkpoint_file is not None:
        save_model(
                model=model,
                checkpoint_file=checkpoint_file,
                optimizer=optimizer,
                true_epoch=true_epoch,
                results=results,
                model_file_name='last.pt'
            )

    return pd.DataFrame.from_dict(results)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape) 
    
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return self.lambd(x)
    
class DebugShape(nn.Module):
    """
    Module that is useful to help debug your neural network architecture. 
    Insert this module between layers and it will print out the shape of 
    that layer. 
    """
    def forward(self, input):
        print(input.shape)
        return input
    
def weight_reset(m):
    """
    Go through a PyTorch module m and reset all the weights to an initial random state
    """
    if "reset_parameters" in dir(m):
        m.reset_parameters()
    return

def moveTo(obj, device):
    """
    obj: the python object ot move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if isinstance(obj, list):
        return [moveTo(o, device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(o, device) for o in obj)
    elif isinstance(obj, set):
        return set(moveTo(o, device) for o in obj)
    elif isinstance(obj, dict):
        to_ret = dict()
        for k, v in obj.items():
            to_ret[moveTo(k, device)] = moveTo(v, device)
        return to_ret
    elif hasattr(obj, "to"):
        return obj.to(device)
    else:
        return obj
        
def train_network(model, loss_func, train_loader, val_loader=None, test_loader=None,score_funcs=None, 
                         epochs=50, device="cpu", checkpoint_file=None, 
                         lr_schedule=None, optimizer=None, disable_tqdm=False
                        ):
    """Train simple neural networks
    
    Keyword arguments:
    model -- the PyTorch model / "Module" to train
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs. 
    val_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model
    epochs -- the number of training epochs to perform
    device -- the compute lodation to perform training
    lr_schedule -- the learning rate schedule used to alter \eta as the model trains. If this is not None than the user must also provide the optimizer to use. 
    optimizer -- the method used to alter the gradients for learning. 
    
    """
    if score_funcs == None:
        score_funcs = {}#Empty set 
    
    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score )
        if val_loader is not None:
            to_track.append("val " + eval_score )
        if test_loader is not None:
            to_track.append("test "+ eval_score )
        
    total_train_time = 0 #How long have we spent in the training loop? 
    results = {}
    #Initialize every item with an empty list
    for item in to_track:
        results[item] = []

        
    if optimizer == None:
        #The AdamW optimizer is a good default optimizer
        optimizer = torch.optim.AdamW(model.parameters())
        del_opt = True
    else:
        del_opt = False

    #Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch", disable=disable_tqdm):
        model = model.train()#Put our model in training mode

        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")
        
        results["epoch"].append( epoch )
        results["total time"].append( total_train_time )
        
      
        if val_loader is not None:
            model = model.eval() #Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimizer, val_loader, loss_func, device, results, score_funcs, prefix="val", desc="Validating")
                
        #In PyTorch, the convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()
                
        if test_loader is not None:
            model = model.eval() #Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")
        
        
        if checkpoint_file is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results' : results
                }, checkpoint_file)
    if del_opt:
        del optimizer

    return pd.DataFrame.from_dict(results)

### RNN utility Classes 

class LastTimeStep(nn.Module):
    """
    A class for extracting the hidden activations of the last time step following 
    the output of a PyTorch RNN module. 
    """
    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        if bidirectional:
            self.num_driections = 2
        else:
            self.num_driections = 1    
    
    def forward(self, input):
        #Result is either a tupe (out, h_t)
        #or a tuple (out, (h_t, c_t))
        rnn_output = input[0]

        last_step = input[1]
        if(type(last_step) == tuple):
            last_step = last_step[0]
        batch_size = last_step.shape[1] #per docs, shape is: '(num_layers * num_directions, batch, hidden_size)'
        
        last_step = last_step.view(self.rnn_layers, self.num_driections, batch_size, -1)
        #We want the last layer's results
        last_step = last_step[self.rnn_layers-1] 
        #Re order so batch comes first
        last_step = last_step.permute(1, 0, 2)
        #Finally, flatten the last two dimensions into one
        return last_step.reshape(batch_size, -1)
    
class EmbeddingPackable(nn.Module):
    """
    The embedding layer in PyTorch does not support Packed Sequence objects. 
    This wrapper class will fix that. If a normal input comes in, it will 
    use the regular Embedding layer. Otherwise, it will work on the packed 
    sequence to return a new Packed sequence of the appropriate result. 
    """
    def __init__(self, embd_layer):
        super(EmbeddingPackable, self).__init__()
        self.embd_layer = embd_layer 
    
    def forward(self, input):
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            # We need to unpack the input, 
            sequences, lengths = torch.nn.utils.rnn.pad_packed_sequence(input.cpu(), batch_first=True)
            #Embed it
            sequences = self.embd_layer(sequences.to(input.data.device))
            #And pack it into a new sequence
            return torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.cpu(), 
                                                           batch_first=True, enforce_sorted=False)
        else:#apply to normal data
            return self.embd_layer(input)



### Attention Mechanism Layers

class ApplyAttention(nn.Module):
    """
    This helper module is used to apply the results of an attention mechanism toa set of inputs. 
    """

    def __init__(self):
        super(ApplyAttention, self).__init__()
        
    def forward(self, states, attention_scores, mask=None):
        """
        states: (B, T, H) shape giving the T different possible inputs
        attention_scores: (B, T, 1) score for each item at each context
        mask: None if all items are present. Else a boolean tensor of shape 
            (B, T), with `True` indicating which items are present / valid. 
            
        returns: a tuple with two tensors. The first tensor is the final context
        from applying the attention to the states (B, H) shape. The second tensor
        is the weights for each state with shape (B, T, 1). 
        """
        
        if mask is not None:
            #set everything not present to a large negative value that will cause vanishing gradients 
            attention_scores[~mask] = -1000.0
        #compute the weight for each score
        weights = F.softmax(attention_scores, dim=1) #(B, T, 1) still, but sum(T) = 1
    
        final_context = (states*weights).sum(dim=1) #(B, T, D) * (B, T, 1) -> (B, D)
        return final_context, weights

class AttentionAvg(nn.Module):

    def __init__(self, attnScore):
        super(AttentionAvg, self).__init__()
        self.score = attnScore
    
    def forward(self, states, context, mask=None):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, D), a weighted av
        
        """
        
        B = states.size(0)
        T = states.size(1)
        D = states.size(2)
        
        scores = self.score(states, context) #(B, T, 1)
        
        if mask is not None:
            scores[~mask] = float(-10000)
        weights = F.softmax(scores, dim=1) #(B, T, 1) still, but sum(T) = 1
        
        context = (states*weights).sum(dim=1) #(B, T, D) * (B, T, 1) -> (B, D, 1)
        
        
        return context.view(B, D) #Flatten this out to (B, D)


class AdditiveAttentionScore(nn.Module):

    def __init__(self, D):
        super(AdditiveAttentionScore, self).__init__()
        self.v = nn.Linear(D, 1)
        self.w = nn.Linear(2*D, D)
    
    def forward(self, states, context):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, T, 1), giving a score to each of the T items based on the context D
        
        """
        T = states.size(1)
        #Repeating the values T times 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, D) -> (B, T, D)
        state_context_combined = torch.cat((states, context), dim=2) #(B, T, D) + (B, T, D)  -> (B, T, 2*D)
        scores = self.v(torch.tanh(self.w(state_context_combined)))
        return scores

class GeneralScore(nn.Module):

    def __init__(self, D):
        super(GeneralScore, self).__init__()
        self.w = nn.Bilinear(D, D, 1)
    
    def forward(self, states, context):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, T, 1), giving a score to each of the T items based on the context D
        
        """
        T = states.size(1)
        D = states.size(2)
        #Repeating the values T times 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, D) -> (B, T, D)
        scores = self.w(states, context) #(B, T, D) -> (B, T, 1)
        return scores

class DotScore(nn.Module):

    def __init__(self, D):
        super(DotScore, self).__init__()
    
    def forward(self, states, context):
        """
        states: (B, T, D) shape
        context: (B, D) shape
        output: (B, T, 1), giving a score to each of the T items based on the context D
        
        """
        T = states.size(1)
        D = states.size(2)
        
        scores = torch.bmm(states,context.unsqueeze(2)) / np.sqrt(D) #(B, T, D) -> (B, T, 1)
        return scores
    
def getMaskByFill(x, time_dimension=1, fill=0):
    """
    x: the original input with three or more dimensions, (B, ..., T, ...)
        which may have unsued items in the tensor. B is the batch size, 
        and T is the time dimension. 
    time_dimension: the axis in the tensor `x` that denotes the time dimension
    fill: the constant used to denote that an item in the tensor is not in use,
        and should be masked out (`False` in the mask). 
    
    return: A boolean tensor of shape (B, T), where `True` indicates the value
        at that time is good to use, and `False` that it is not. 
    """
    to_sum_over = list(range(1,len(x.shape))) #skip the first dimension 0 because that is the batch dimension
    
    if time_dimension in to_sum_over:
        to_sum_over.remove(time_dimension)
       
    with torch.no_grad():
        #Special case is when shape is (B, D), then it is an embedding layer. We just return the values that are good
        if len(to_sum_over) == 0:
            return (x != fill)
        #(x!=fill) determines locations that might be unused, beause they are 
        #missing the fill value we are looking for to indicate lack of use. 
        #We then count the number of non-fill values over everything in that
        #time slot (reducing changes the shape to (B, T)). If any one entry 
        #is non equal to this value, the item represent must be in use - 
        #so return a value of true. 
        mask = torch.sum((x != fill), dim=to_sum_over) > 0
    return mask

class LanguageNameDataset(Dataset):
    
    def __init__(self, lang_name_dict, vocabulary):
        self.label_names = [x for x in lang_name_dict.keys()]
        self.data = []
        self.labels = []
        self.vocabulary = vocabulary
        for y, language in enumerate(self.label_names):
            for sample in lang_name_dict[language]:
                self.data.append(sample)
                self.labels.append(y)
        
    def __len__(self):
        return len(self.data)
    
    def string2InputVec(self, input_string):
        """
        This method will convert any input string into a vector of long values, according to the vocabulary used by this object. 
        input_string: the string to convert to a tensor
        """
        T = len(input_string) #How many characters long is the string?
        
        #Create a new tensor to store the result in
        name_vec = torch.zeros((T), dtype=torch.long)
        #iterate through the string and place the appropriate values into the tensor
        for pos, character in enumerate(input_string):
            name_vec[pos] = self.vocabulary[character]
            
        return name_vec
    
    def __getitem__(self, idx):
        name = self.data[idx]
        label = self.labels[idx]
        
        #Conver the correct class label into a tensor for PyTorch
        label_vec = torch.tensor([label], dtype=torch.long)
        
        return self.string2InputVec(name), label
    
def pad_and_pack(batch):
    #1, 2, & 3: organize the batch input lengths, inputs, and outputs as seperate lists
    input_tensors = []
    labels = []
    lengths = []
    for x, y in batch:
        input_tensors.append(x)
        labels.append(y)
        lengths.append(x.shape[0]) #Assume shape is (T, *)
    #4: create the padded version of the input
    x_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
    #5: create the packed version from the padded & lengths
    x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)
    #Convert the lengths into a tensor
    y_batched = torch.as_tensor(labels, dtype=torch.long)
    #6: return a tuple of the packed inputs and their labels
    return x_packed, y_batched
