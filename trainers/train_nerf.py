import pytorch_lightning as pl
from networks.mlp import NeRF_MLP
import torch
from torch.optim import Adam
import torch.nn as nn
import tqdm

import sys
import numpy as np
from utils.rendering import rendering


class NeRF_Trainer(pl.LightningModule):

    def __init__(self, hparams):
        super(NeRF_Trainer, self).__init__()

        self.config = hparams

        self.mlp = NeRF_MLP(self.config["freq_num"])

    def configure_optimizers(self):
        optimizer = Adam(params=self.mlp.parameters(), lr=self.config["lr"])
        return [optimizer]

    def training_step(self, batch, batch_idx):
        gt_image, points = batch
        rendered_img = self.forward(points)

        gt_image = gt_image.squeeze().permute(1, 2, 0)
        loss = nn.MSELoss()(gt_image, rendered_img)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        gt_image, points = batch
        rendered_img = self.forward(points)

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
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def forward(self, points):
        flat_points = points.reshape(-1, 3)

        output_flat = (self.batchify()(flat_points)).squeeze()
        flat_rgbs, flat_density = output_flat[..., 0:3], output_flat[..., -1:]

        rgbs, density = torch.reshape(flat_rgbs, points.shape), torch.reshape(flat_density, points.shape[0:-1])

        rendered_img = rendering(
            rgbs, density, self.config["rendering_params"], self.device)

        return rendered_img

    def on_train_epoch_end(self):
        self.eval()

        gt_image, points = next(iter(self.trainer.train_dataloader))

        points = points.to(device=self.device)
        rendered_img = self.forward(points)

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

        def process_chunks(inputs):
            return torch.cat([self.mlp(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return process_chunks
