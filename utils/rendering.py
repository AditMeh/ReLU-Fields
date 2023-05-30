import torch

def pose_to_rays(rotation, translation, focal, h, w, t_n, t_f, num_samples):
    xs = torch.arange(w)
    ys = torch.arange(h)

    h_mesh, w_mesh = torch.meshgrid(xs, ys, indexing='ij')

    pixels_unflatten = torch.stack([(w_mesh - w * .5) / focal, -(
        h_mesh - h * .5) / focal, -torch.ones_like(h_mesh)], dim=-1)
    pixels = torch.reshape(pixels_unflatten, (h*w, 3))

    dirs = torch.matmul(rotation, pixels.T).T
    dirs_tformed = torch.reshape(dirs, (h, w, 3))

    origin = torch.broadcast_to(translation, dirs_tformed.shape)

    ts = torch.linspace(t_n, t_f, steps=num_samples)

    ray_points = origin[..., None, :] + \
        dirs_tformed[..., None, :] * ts[:, None]

    return ray_points

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

    rendered_img = torch.reshape(C, (config["h"], config["w"], 3))

    return rendered_img


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
