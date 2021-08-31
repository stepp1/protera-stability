base_train = dict(
    output_dir="./logs",
    ckpt_dir="models",
    max_epochs=int(1e4),
    log_every_n_stringsteps=1,
    amp_backend="apex",
    benchmark=True,
    random_split=0.8,
    sequence_sampling="",  # one of ["", "diversity", "random"]
)
