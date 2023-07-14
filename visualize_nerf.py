import pytorch_lightning as pl
from networks.mlp import NeRF_MLP
import torch
from torch.optim import Adam
import torch.nn as nn
import tqdm
import argparse

import json
import sys
import numpy as np
from utils.rendering import rendering, pose_to_rays
from utils.lookat import lookat, circle_points
import imageio

def batchify(config, mlp):
    chunk = config["chunk"]
    if chunk is None:
        return self.mlp
    def process_chunks(inputs):
        return torch.cat([mlp(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return process_chunks

def circle_view():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        help='Hyperparameters of the run')
    parser.add_argument('--checkpoint', nargs='?', type=str, help='checkpoint')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    dataset_path = f'NeRF-Scenes/{config["dataset_name"]}/'

    with open(f'{dataset_path}/transforms_train.json', 'r') as f:
        angle = json.load(f)["camera_angle_x"]

        # The default h/w is 800x800, so use that to compute focal
        focal = .5 * 800 / \
            np.tan(.5 * float(angle))


    device = (torch.device('cuda')
            if torch.cuda.is_available() else torch.device('cpu'))
    
    checkpoint = torch.load(args.checkpoint)
    lightning_state_dict = checkpoint["state_dict"]
    mlp = NeRF_MLP(config["freq_num"]).to(device)

    mlp_state_dict = mlp.state_dict() 
    # Need to copy state dict
    for k in mlp_state_dict.keys():
        mod_key = f'mlp.{k}'
        v = lightning_state_dict[mod_key]
        mlp_state_dict[k] = v

    mlp.load_state_dict(mlp_state_dict)

    points = circle_points(2, 4, 110)
    del config["rendering_params"]["num_rays"]
    config["rendering_params"]["focal"] = focal

    def sanitize(img): return (torch.squeeze(
        img).detach().cpu().numpy()*255).astype(np.uint8)

    lego_dataset_path = "./NeRF-Scenes/lego/transforms_train.json"

    with open(lego_dataset_path, 'r') as f:
        frames = json.load(f)["frames"]
        for pose in frames:
            mat = torch.from_numpy(np.asarray(pose["transform_matrix"], dtype=np.float32))
            R_true,t_true = mat[:3, :3], torch.squeeze(mat[:3, 3:])


    with torch.no_grad():
        for i, position in tqdm.tqdm(enumerate(points)):
            c2w = lookat(torch.FloatTensor([0, 0, 0]), position)
            R, t = c2w[:3, :3], torch.squeeze(c2w[:3, 3:])

            rays_points, rays_dirs = pose_to_rays(R, t, **config["rendering_params"])
            flat_points = rays_points.reshape(-1, 3).to(device=device)

            output_flat = (batchify(config, mlp)(flat_points)).squeeze()
            flat_rgbs, flat_density = output_flat[..., 0:3], output_flat[..., -1:]

            rgbs, density = torch.reshape(flat_rgbs, (1, *rays_points.shape)), torch.reshape(flat_density, (1,*rays_points.shape[0:-1]))

            rendered_img = rendering(
                rgbs, density, config["rendering_params"], device = device)

            imageio.imsave(f'visuals/{i}.png', sanitize(rendered_img))


if __name__ == "__main__":
    circle_view()