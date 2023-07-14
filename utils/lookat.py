import torch

def lookat(origin, loc):
    dir = loc - origin
    dir = dir / torch.linalg.norm(dir)

    tmp = torch.FloatTensor([0, 0, 1])
    right = torch.cross(tmp, dir)
    up = torch.cross(dir, right)

    R = torch.hstack([right[..., None], up[..., None], dir[..., None]])

    return torch.vstack(
        [torch.hstack([R, loc[..., None]]),
         torch.FloatTensor([0, 0, 0, 1])[None, ...]])

def circle_points(z, radius, num_points):
    split = torch.FloatTensor([(2 * torch.pi) / num_points])

    vals = []
    for i in range(num_points):
        angle = split * i
        vals.append(torch.FloatTensor(
            [radius * torch.cos(angle), radius * torch.sin(angle), z]))
    return vals


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path) as json_file:
        hparams = json.load(json_file)

    device = (torch.device('cuda')
              if torch.cuda.is_available() else torch.device('cpu'))

    images, poses, focal, w, h = load_data('tiny_nerf_data.npz')
    nerf_model = torch.load("model.pt").to(device=device)

    imgs = []
    camera_positions = circle_points(2, 5, 110)
    for position in camera_positions:
        c2w = lookat(np.asarray([0, 0, 0]), position).astype(np.float32)
        imgs.append(
            (255 *
             np.clip(show_view(c2w, focal, h, w, **hparams), 0, 1)).astype(
                 np.uint8))

    f = 'video.mp4'
    imageio.mimwrite(f, imgs, fps=30, quality=7)
