import argparse
from trainers.train_nerf import NeRF_Trainer
from datasets.blender_datasets import BlenderDataModule
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        help='Hyperparameters of the run')
    parser.add_argument('--checkpoint', nargs='?', type=str, help='checkpoint')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    model = NeRF_Trainer(config)

    train_loader = BlenderDataModule(
        config["dataset_name"], config["batch_size"], **config["rendering_params"])

    checkpoint = ModelCheckpoint(monitor="validation_loss_epoch")

    early_stop = EarlyStopping(monitor="train_loss_epoch",
                               min_delta=1e-8, patience=5, mode="min",
                               stopping_threshold=1e-4, divergence_threshold=10, verbose=False)

    wandb_logger = WandbLogger(
        name=config['dataset_name'], project='NeRF_PL')

    trainer = pl.Trainer(devices=[0], accelerator="gpu", max_epochs=config["epochs"],
                         precision=32, callbacks=[checkpoint, early_stop], logger=wandb_logger)

    trainer.fit(model, train_loader, ckpt_path=args.checkpoint)
