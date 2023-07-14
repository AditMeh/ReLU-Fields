import torch
import numpy as np


def pose_to_rays(rotation, translation, focal, h, w, t_n, t_f, num_samples):
    xs = torch.arange(w)
    ys = torch.arange(h)

    h_mesh, w_mesh = torch.meshgrid(xs, ys, indexing='ij')

    pixels_unflatten = torch.stack([(w_mesh - w * .5) / focal, -(
        h_mesh - h * .5) / focal, -torch.ones_like(h_mesh)], dim=-1)
    pixels = torch.reshape(pixels_unflatten, (h*w, 3))

    dirs = torch.matmul(rotation, pixels.T).T
    dirs_tformed = torch.reshape(dirs, (h, w, 3))
    # dirs_tformed = dirs_tformed / torch.linalg.vector_norm(dirs_tformed, dim = -1)[..., None]

    # assert torch.isclose(torch.mean(torch.linalg.norm(dirs_tformed, axis= -1)), torch.FloatTensor([1.0]))

    origin = torch.broadcast_to(translation, dirs_tformed.shape)

    ts = torch.linspace(t_n, t_f, steps=num_samples)

    ray_points = origin[..., None, :] + \
        dirs_tformed[..., None, :] * ts[:, None]
    
    return ray_points, torch.broadcast_to(dirs_tformed[:, :, None, :], ray_points.shape)


def pose_to_rays_sampled(rotation, translation, focal, h, w, t_n, t_f, num_samples, num_rays):
    xs = torch.arange(w)
    ys = torch.arange(h)

    h_mesh, w_mesh = torch.meshgrid(xs, ys, indexing='ij')

    # List of points of shape [h*w, 2]
    points = torch.stack([h_mesh, w_mesh], dim=-1).reshape(-1, 2)

    rand_rays_idxs = np.random.choice(
        points.shape[0], size=(num_rays), replace=False)

    # This is a list of shape [num_rays, 2]
    rand_ray_coords = points[rand_rays_idxs, :]

    pixels = torch.stack([(w_mesh - w * .5) / focal, -(
        h_mesh - h * .5) / focal, -torch.ones_like(h_mesh)], dim=-1)
    pixels_flattened = torch.reshape(pixels, (h*w, 3))

    dirs_flattened = torch.matmul(rotation, pixels_flattened.T).T
    dirs = torch.reshape(dirs_flattened, (h, w, 3))
    # dirs = dirs / torch.linalg.vector_norm(dirs, dim = -1)[..., None]

    # (num_rays, 3)
    sampled_dirs = dirs[rand_ray_coords[:, 0], rand_ray_coords[:, 1], :]

    # (num_rays, 3)
    origin = torch.broadcast_to(translation, sampled_dirs.shape)

    ts = torch.linspace(t_n, t_f, steps=num_samples)

    # (num_rays, 3) + (num_rays, 3) * (num_samples) = (num_rays, num_samples, 3)
    # (num_rays, None, 3) + (num_rays, None, 3) * (None, num_samples, None)

    ray_points = origin[:, None, :] + \
        sampled_dirs[:, None, :] * ts[None, :, None]


    return ray_points, torch.broadcast_to(sampled_dirs[:, None, :], ray_points.shape), rand_ray_coords


def rendering(color, density, config, device):
    """
    color: (h, w, num_samples along each ray, 3)
    density: (h, w, num_samples along each ray)
    dist_delta: the distance between the two neighbouring points on a ray, 
                assume it's constant and a float
    """
    dist_delta = (config["t_f"] - config["t_n"]) / config["num_samples"]

    delta_broadcast = torch.ones(density.shape[-1]) * dist_delta
    delta_broadcast[-1] = 1e10
    delta_broadcast = delta_broadcast.to(device=device)

    density_times_delta = density * delta_broadcast
    density_times_delta = density_times_delta.to(device=device)

    dists = torch.ones(density.shape[-1]) * dist_delta
    dists[-1] = 1e10
    density_times_delta = density * dist_delta

    T = torch.exp(-cumsum_exclusive(density_times_delta))

    # roll T to right by one postion and replace the first column with 1
    S = 1 - torch.exp(-density_times_delta)
    points_color = (T * S)[..., None] * color
    C = torch.sum(points_color, dim=-2)

    # full image

    if len(color.shape) == 5:
        rendered_img = torch.reshape(C, (config["h"], config["w"], 3))
        return rendered_img
    else:
        return torch.reshape(C, (-1, 3))

def cumsum_exclusive(t):
    dim = -1
    cumsum = torch.cumsum(t, dim)
    cumsum = torch.roll(cumsum, 1, dim)
    cumsum[..., 0] = 0.
    return cumsum
    
def cumprod_exclusive(t):
    dim = -1
    cumprod = torch.cumprod(t, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0] = 1.
    return cumprod
