base_train = dict(
    trainer_params=dict(
        gpus=1,
        max_epochs=int(1e4),
        log_every_n_steps=1,
        amp_backend="apex",
        benchmark=True,
        check_val_every_n_epoch=3
    ),
    output_dir="./logs",
    random_split=0.8,
)
