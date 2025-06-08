"""Extract features for temporal action detection datasets"""
import argparse
import os
import random

import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms
import pandas as pd

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--data_set',
        default='SIGNLANGUAGE',
        choices=['THUMOS14', 'FINEACTION', "SIGNLANGUAGE"],
        type=str,
        help='dataset')
    parser.add_argument(
        '--save_path',
        default='/mnt/HDD/embeddings/train/video_mae2_emb_felipe_s',
        type=str,
        help='path for saving features')
    parser.add_argument(
        '--model',
        default='vit_small_patch16_224',
        type=str,
        metavar='MODEL',
        help='Name of model')
    parser.add_argument(
        '--ckpt_path',
        default='/mnt/HDD/models/model_zoo/vit_s_k710_dl_from_giant.pth',
        help='load from checkpoint')

    return parser.parse_args()


def get_start_idx_range(data_set):

    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)

    def sl_range(num_frames):
        return range(0, num_frames - 15, 16)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    elif data_set == 'SIGNLANGUAGE':
        return sl_range
    else:
        raise NotImplementedError()


def aggregate_video_embeddings(
    patch_embeddings: torch.Tensor,
    method: str = "mean"
) -> torch.Tensor:
    """
    Agrega los embeddings de patches para obtener un único embedding por video.

    Args:
        patch_embeddings (torch.Tensor): Tensor de shape [B, N, D]
            B = batch size
            N = número de patches
            D = dimensión del embedding
        method (str): Método de agregación: "mean", "max", "cls" (si tu modelo incluye un token especial)

    Returns:
        torch.Tensor: Embedding agregado por video. Shape: [B, D]
    """
    if method == "mean":
        return patch_embeddings.mean(dim=0)  # Promedio por patch
    elif method == "max":
        return patch_embeddings.max(dim=0).values  # Máximo por dimensión
    else:
        raise ValueError(f"Método de agregación no soportado: {method}")


def extract_feature(args, vid_list, classes_list, to_flip_list):
    # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    video_loader = get_video_loader()
    start_idx_range = get_start_idx_range(args.data_set)
    transform = transforms.Compose(
        [ToFloatTensorInZeroOne(),
         Resize((224, 224))])

    # get model & load ckpt
    model = create_model(
        args.model,
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        drop_path_rate=0.3,
        use_mean_pooling=True)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()

    # extract feature
    num_videos = len(vid_list)
    for idx, (video_path, class_name, to_flip) in enumerate(zip(vid_list, classes_list, to_flip_list)):
        # output_dir = os.path.join(args.save_path, class_name)
        # os.makedirs(output_dir, exist_ok=True)
        print(f"{idx}/{num_videos-1}")
        print(f"class_name: {class_name}")
        print(f"video_path: {video_path}")
        print(f"to_flip: {to_flip}")
        print(f"emb_output_dir: {args.save_path}")

        if "/" in class_name:
            continue

        url = os.path.join(args.save_path, f'{class_name}.npy')

        vr = video_loader(video_path)
        feature_list = []
        for start_idx in start_idx_range(len(vr)):
            data = vr.get_batch(np.arange(start_idx, start_idx + 16)).asnumpy()
            frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
            frame_q = transform(frame)  # torch.Size([3, 16, 224, 224])

            # Flip video
            if to_flip:
                frame_q = torch.flip(frame_q, dims=[2]) # torch.Size([3, 16, 224, 224])
                print(f"Video Flipped | size:({frame_q.shape})")

            input_data = frame_q.unsqueeze(0).cuda()

            with torch.no_grad():
                feature = model.forward_features(input_data)
                feature_list.append(feature.cpu().to(torch.float32))

        # [N, C]
        embedding = torch.cat(feature_list, dim=0)
        embedding = aggregate_video_embeddings(
            patch_embeddings=embedding, method="mean"
        )
        print(f"embedding: {embedding.shape}\n") # [1, 384]

        torch.save(embedding, url)
        print(f"Feature saved at: {url}")


if __name__ == '__main__':
    args = get_args()
    # DATASET_PATH = "/mnt/HDD/datasets/felipe/full_dataset.csv" #Train
    # args.save_path = "/mnt/HDD/embeddings/train/video_mae2_emb_felipe_s"

    DATASET_PATH = "/mnt/HDD/datasets/felipe_test/test_full_dataset.csv" #Test
    args.save_path = "/mnt/HDD/embeddings/test/video_mae2_emb_felipe_s"

    data = pd.read_csv(DATASET_PATH)
    vid_list = data["video_path"].tolist()
    classes_list = data["class_name"].tolist()
    to_flip_list = data["to_flip"].tolist()
    extract_feature(args, vid_list=vid_list, classes_list=classes_list, to_flip_list=to_flip_list)
