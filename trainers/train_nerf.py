import pytorch_lightning as pl
from networks.mlp import NeRF_MLP, ReplicateNeRFModel, MultiHeadNeRFModel
import torch
from torch.optim import Adam
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


import sys
import numpy as np
from utils.rendering import rendering


class NeRF_Trainer(pl.LightningModule):

    def __init__(self, hparams, val_dataloader):
        super(NeRF_Trainer, self).__init__()

        self.config = hparams
        self.save_hyperparameters()

        self.mlp = ReplicateNeRFModel()
        
        self.val_dataloader = val_dataloader
    def configure_optimizers(self):
        optimizer = Adam(params=self.mlp.parameters(), lr=self.config["lr"])

        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "validation_loss_epoch",
            "strict": True,
            "name": None,
        }
        return [optimizer], lr_scheduler_config

    def training_step(self, batch, batch_idx):
        gt_image, points, directions = batch
        rendered_img = self.forward(points, directions)

        gt_image = gt_image.squeeze().permute(1, 2, 0)

        loss = nn.MSELoss()(gt_image, rendered_img)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        gt_image, points, directions = batch
        rendered_img = self.forward(points, directions)

        gt_image = gt_image.squeeze().permute(1, 2, 0)

        loss = nn.MSELoss()(gt_image, rendered_img)
        self.log("validation_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        gt_image, points = batch
        rendered_img = self.forward(points)

        loss = nn.MSELoss()(gt_image, rendered_img)
        self.log("test_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)

        return loss

    def forward(self, points, direction):
        
        flat_points = points.reshape(-1, 3)
        flat_dirs = direction.reshape(-1, 3)

        output_flat = (self.batchify()(flat_points, flat_dirs)).squeeze()
        flat_rgbs, flat_density = output_flat[..., 0:3], output_flat[..., -1:]

        rgbs, density = torch.reshape(flat_rgbs, points.shape), torch.reshape(flat_density, points.shape[0:-1])

        rendered_img = rendering(
            rgbs, density, self.config["rendering_params"], self.device)

        return rendered_img

    def on_train_epoch_end(self):

        with torch.no_grad():
            gt_image, points, directions = next(iter(self.val_dataloader))
            points = points.to(device=self.device)
            directions = directions.to(device=self.device)

            rendered_img = self.forward(points, directions)

            def sanitize(img): return (torch.squeeze(
                img).detach().cpu().numpy()*255).astype(np.uint8)

            generated_image = [sanitize(rendered_img)] + \
                [sanitize(gt_image.permute(0, 2, 3, 1))]

            self.logger.log_image(
                key="generated image vs real", images=generated_image)
        return

    def batchify(self):
        chunk = self.config["chunk"]
        if chunk is None:
            return self.mlp

        def process_chunks(xyz, dirs):
            assert len(xyz.shape) == len(dirs.shape)
            assert [xyz.shape[i] == dirs.shape[i] for i in range(len(xyz.shape))]

            return torch.cat([self.mlp(xyz[i:i+chunk], dirs[i:i+chunk]) for i in range(0, xyz.shape[0], chunk)], 0)
        return process_chunks
