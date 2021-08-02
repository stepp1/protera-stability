from skorch.callbacks import Checkpoint, LRScheduler, EarlyStopping
from skorch import NeuralNetRegressor

import torch
from torch import nn

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

scoring = "r2"
score = r2_score

def perform_search(model, params, name, strategy = "grid", save_dir = "models"):
    
    
    if strategy == "grid":
        searcher = GridSearchCV
        
    elif strategy == "bayes":
        raise NotImplementedError
        
    
    search = searcher(
        estimator=model,
        param_grid=params,
        scoring=scoring,
        verbose=1,
        n_jobs=-1,
        refit=True,
    )
    
    print("============")
    print(f'Fitting model {name}...')
    grid.fit(X_train, y_train)
    print(name, grid.best_score_)
    print(grid.best_params_)
    print("============")
    
    if "sklearn" in str(type(model)):
        dump(grid, f"{save_dir}/best_{name}.joblib")
        
    else: 
        # this assumes we're using skorch
        torch.save(search.state_dict(), f"{save_dir}/best_{name}.pt")
        
    return search


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