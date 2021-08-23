from skorch import NeuralNetRegressor

import torch
from torch import nn

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from joblib import dump
import copy

scoring = "r2"
score = r2_score


def perform_search(X, y, model, params, name, strategy="grid", save_dir="models"):
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
    print(f"Fitting model {name}...")
    search.fit(X, y)
    print(f"{name} best R2: {search.best_score_}")
    print(f"Best params: {search.best_params_}")
    print("============")

    if "sklearn" in str(type(model)):
        dump(search, f"{save_dir}/best_{name}.joblib")

    else:
        # this assumes we're using skorch
        torch.save(search.state_dict(), f"{save_dir}/best_{name}.pt")

    return search


class ProteinMLP(nn.Module):
    def __init__(
        self, n_in=1280, n_units=1024, n_layers=3, act=None, drop_p=0.7, last_drop=False
    ):
        super(ProteinMLP, self).__init__()

        layers = []
        for i in range(n_layers):
            in_feats = n_in if i == 0 else n_units // i
            if i == 0:
                in_feats = n_in
                out_feats = n_units
            else:
                in_feats = out_feats
                out_feats = in_feats // 2

            out_feats = 1 if i == (n_layers - 1) else out_feats
            fc = nn.Linear(in_feats, out_feats)
            layers.append(fc)
        self.layers = nn.ModuleList(layers)

        self.drop = nn.Dropout(p=drop_p)
        self.last_drop = last_drop
        self.act = act
        if act is None:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i == len(self.layers) - 1:
                continue
            else:
                out = self.act(self.drop(out))

        if self.last_drop:
            out = self.drop(out)
        return out
