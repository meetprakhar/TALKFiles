# %env CUDA_LAUNCH_BLOCKING=1
import torch
from torch import nn
import torch.nn.functional as f
from torch.optim.optimizer import Optimizer
import numpy as np
import pandas as pd
from torchsummary import summary
import random
import os
import glob
from scipy.signal import savgol_filter as sgolayfilt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Deterministic Behaviour
seed_everything(420)


def calc_delta(arr):
    del_arr = np.zeros_like(arr)
    del_arr[1:] = np.array(
        list([arr[i] - arr[i-1] for i in range(1, len(arr))]))
    return del_arr


def load_data(folder, delta=False, svgolay_filt=False, window_length=11, polyorder=3, normalize='minmax', split=0.8):
    csvs = glob.glob(os.path.join(folder, '*.csv'))
    
    x_train = []
    x_val = []
    y_train = []
    y_val = []
    
    # Read the CSVs
    for csv in csvs:
        d = pd.read_csv(csv)
        l = d.iloc[:, 3]
        d = d.iloc[:, :3]
        
        # Apply delta and/or svgolay filtering
        if delta or svgolay_filt:
            deltax = calc_delta(d.iloc[:, 0])
            deltay = calc_delta(d.iloc[:, 1])
            deltaz = calc_delta(d.iloc[:, 2])
            
            if svgolay_filt:
                deltax = pd.Series(sgolayfilt(deltax, window_length, polyorder))
                deltay = pd.Series(sgolayfilt(deltay, window_length, polyorder))
                deltaz = pd.Series(sgolayfilt(deltaz, window_length, polyorder))
            
            d = pd.concat([deltax, deltay, deltaz], axis=1)
        
        # Normalize
        if normalize is not None:
            assert normalize in ['minmax', 'mean'], "Must be one of ['minmax', 'mean']"
            d = (d - d.mean(axis=0)) / (d.max(axis=0) - d.min(axis=0)) if 'minmax' else \
                (d - d.mean(axis=0)) / d.std(axis=0)
        
        # Convert to float32 numpy array to comply with torch cuda backend
        x_train.append(torch.from_numpy(d.iloc[:int(split * len(d)), :].values.astype(np.float32)))
        x_val.append(torch.from_numpy(d.iloc[int(split * len(d)):, :].values.astype(np.float32)))
        y_train.append(l[:int(split * len(d))].values[..., None].astype(np.float32))
        y_val.append(l[int(split * len(d)):].values[..., None].astype(np.float32))
        
        # Convert to one-hot encoding and drop 0th column, belonging to background
        enc = ohe()
        enc.fit(np.vstack(y_train + y_val))
        y_train = [torch.from_numpy(enc.transform(j).toarray()[:, 1:].astype(np.float32)) for j in y_train]
        y_val = [torch.from_numpy(enc.transform(j).toarray()[:, 1:].astype(np.float32)) for j in y_val]

    return x_train, y_train, x_val, y_val


class LSTM(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_dim, n_cells, n_fc=None,
        fc_act=nn.ReLU(), out_act=nn.Sigmoid()
            ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, n_cells, batch_first=True)
        self.n_fc = n_fc

        last_dim = hidden_dim
        if self.n_fc is not None:
            self.fc = []
            for n_dims in n_fc:
                self.fc.append(nn.Linear(last_dim, n_dims))
                last_dim = n_dims
            self.fc = nn.ModuleList(self.fc)

        self.fc_act = fc_act  # For use in the top dense layers
        self.out_fc = nn.Linear(last_dim, output_size)
        self.act = out_act  # For use at the output layer

    def forward(self, x):
        # print(x.shape)  #DEBUG
        out, hidden = self.lstm(x)
        # print(out.shape)  #DEBUG
        if self.n_fc is not None:
            for fc in self.fc:
                out = fc(out)
                out = self.fc_act(out)
                # print(out.shape)  #DEBUG

        out = self.out_fc(out)
        out = self.act(out)
        # print(out.shape)  #DEBUG
        return out
    
    
class Trainer(object):
    def __init__(self, model, X_train, y_train, X_val, y_val, lr, epochs, optimizer, criterion, metric_name='accuracy'):
        self.model = model
        self.epochs = epochs
        self.loss = 0
        self.acc = 0
        self.name = metric_name
        self.loss_history = []
        self.acc_history = []
        
        self.criterion = criterion
        if isinstance(optimizer, str) and optimizer.lower() == "lbfgs":
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr)    
        else:
            self.optimizer = optimizer(self.model.parameters(), lr=lr)
      
    def closure(self, X, y):        
        def _closure():
            self.optimizer.zero_grad()  # Clear existing gradient

            output = self.model(X)
            loss = self.criterion(output.contiguous().view(output.shape[0], -1), y.contiguous().view(y.shape[0], -1))
            loss.backward()

            self.loss = loss.item()
            self.loss_history['train'].append(self.loss)
            self.output = output
            self.acc = self.accuracy(self.output, y)
            self.acc_history['train'].append(self.acc)

            print('\rEpoch: {}/{}.............'.format(self.epoch, self.epochs), end=' ')
            print("Loss: {:.4f} {name}: {:.4f}".format(
                self.loss, self.acc, name=self.name), end='')
            return loss
        return _closure

    def start(self):
        for epoch in range(self.epochs):
            closure = self.closure(X_train, y_train, stochastic=stochastic)
            if isinstance(self.optimizer, str) and self.optimizer.lower() == "lbfgs":
                # Slightly different syntax for LBFGS
                self.optimizer.step(closure)
            else:
                _ = closure().item()
                self.optimizer.step()  # Update the weights

            with torch.no_grad():
                losses = []
                accs = []
                
                # Run for each example
                for X, y in zip(X_val, y_val):
                    output_val, _ = self.model(X)
                    # outputs.append(output_val)
                    losses.append((self.criterion(output_val.contiguous().view(output_val.shape[0], -1), y.contiguous().view(y.shape[0], -1))).item())
                        #  + self.step_loss(output_val.view(output_val.shape[0], -1), y_val.view(y_val.shape[0], -1))).item()
                    self.name, acc_val = self.accuracy(output_val, y)
                    accs.append(acc_val)

                self.val_loss = sum(losses) / len(losses)
                self.loss_history['val'].append(self.val_loss)
                

            print('\rEpoch: {}/{}.............'.format(self.epoch, self.epochs), end=' ')
            print("Avg. Loss: {:.4f} Avg. {name}: {:.4f}, Avg. Val_Loss: {:.4f} Avg. {name}: {:.4f}".format(
                self.loss, self.acc, self.val_loss, self.val_acc, name=self.name), end='\n')
    
    def accuracy(self):
            return 0  # For Now
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Sanity test
    x, _, _, _ = load_data("./", svgolay_filt=True)
    model = LSTM(3, 1, 10, 2, [5, 3]).cuda()
    print("Model output:\n", model(torch.from_numpy(x[0]).cuda()[None, ...]), sep='\n')
    plt.plot(x[2][:, 0])
    plt.show()