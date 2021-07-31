from skorch.callbacks import Checkpoint, LRScheduler, EarlyStopping
from skorch import NeuralNetRegressor

from torch import nn

class ProteinMLP(nn.Module):
    
    def __init__(self, n_in = 1280, n_units = 1024, act = None, drop_p = 0.7, last_drop = False):
        super(ProteinMLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_units)
        self.fc2 = nn.Linear(n_units, n_units//2)
        self.fc3 = nn.Linear(n_units//2, 1)
        
        
        self.drop = nn.Dropout(p=drop_p)
        self.last_drop = last_drop
        self.act = act
        if act is None:
            self.act = nn.LeakyReLU()
            
    def forward(self, x):
        out = self.act(self.drop(self.fc1(x)))
        out = self.act(self.drop(self.fc2(out)))
        out = self.fc3(out)
        
        if self.last_drop: 
            out = self.drop(out)
        return self.act(out)