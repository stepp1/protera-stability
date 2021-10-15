from joblib import dump
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

scoring = "r2"
score = r2_score


def perform_search(
    X,
    y,
    model,
    params,
    name,
    X_test=None,
    y_test=None,
    strategy="grid",
    save_dir="models",
    verbose=1,
    n_jobs=-1,
):

    searcher_params = {
        "estimator": model,
        "scoring": scoring,
        "verbose": verbose,
        "n_jobs": n_jobs,
        "refit": True
    }
    if strategy == "grid":
        searcher = GridSearchCV
        searcher_params["param_grid"] = params

    elif strategy == "bayes":
        searcher = BayesSearchCV
        searcher_params["search_spaces"] = params

    search = searcher(**searcher_params)

    print("============")
    print(f"Fitting model {name}...")
    search.fit(X, y)
    print(f"{name} best R2: {search.best_score_}")
    print(f"Best params: {search.best_params_}")

    if X_test is not None and y_test is not None:
        r2_test = search.score(X_test, y_test)
        print(f"Test R2: {r2_test}")
    print("============")

    if "sklearn" in str(type(model)):
        dump(search, f"{save_dir}/best_{name}.joblib")

    else:
        # this assumes we're using skorch
        torch.save(search.state_dict(), f"{save_dir}/best_{name}.pt")

    return search


def create_cbs(ckpt_dir):
    ckpt_cb = ModelCheckpoint(ckpt_dir)
    stop_valid = EarlyStopping(
        monitor="valid_loss", patience=20, check_on_train_epoch_end=False
    )
    stop_r2_reached = EarlyStopping(
        monitor="valid_r2",
        patience=1,
        check_on_train_epoch_end=False,
        stopping_threshold=0.72,
        mode="max",
    )
    monitor_lr = LearningRateMonitor(logging_interval="epoch")

    return [ckpt_cb, stop_valid, stop_r2_reached, monitor_lr]
