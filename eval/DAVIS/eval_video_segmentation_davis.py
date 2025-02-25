import os
import copy
import glob
import queue
import argparse

import cv2
from PIL import Image

import numpy as np

import torch
from torch.nn import functional as F

import vision_transformer as vits

def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    print(checkpoint_key)
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    print(state_dict.keys())
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("base_encoder.", ""): v for k, v in state_dict.items()}
    interpolate_pos_embed(model, state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    # print(
    #     "Pretrained weights found at {} and loaded with msg: {}".format(
    #         pretrained_weights, msg
    #     )
    # )



@torch.no_grad()
def eval_video_tracking_davis(
    args, model, frame_list, video_dir, first_seg, seg_ori, color_palette
):
    """
    Evaluate tracking on a video given first frame & segmentation
    """
    video_folder = os.path.join(args.output_dir, video_dir.split("/")[-1])
    os.makedirs(video_folder, exist_ok=True)

    # The queue stores the n preceeding frames
    que = queue.Queue(args.n_last_frames)

    # first frame
    frame1, ori_h, ori_w = read_frame(frame_list[0])
    # extract first frame feature
    frame1_feat = extract_feature(model, frame1).T  #  dim x h*w

    # saving first segmentation
    out_path = os.path.join(video_folder, "00000.png")
    imwrite_indexed(out_path, seg_ori, color_palette)
    mask_neighborhood = None
    for cnt in range(1, len(frame_list)):
        frame_tar = read_frame(frame_list[cnt])[0]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(
            args, model, frame_tar, used_frame_feats, used_segs, mask_neighborhood
        )

        # pop out oldest frame if neccessary
        if que.qsize() == args.n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(
            frame_tar_avg,
            scale_factor=args.patch_size,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(
            Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0)
        )
        frame_nm = frame_list[cnt].split("/")[-1].replace(".jpg", ".png")
        imwrite_indexed(
            os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette
        )


def restrict_neighborhood(h, w):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * args.size_mask_neighborhood + 1):
                for q in range(2 * args.size_mask_neighborhood + 1):
                    if (
                        i - args.size_mask_neighborhood + p < 0
                        or i - args.size_mask_neighborhood + p >= h
                    ):
                        continue
                    if (
                        j - args.size_mask_neighborhood + q < 0
                        or j - args.size_mask_neighborhood + q >= w
                    ):
                        continue
                    mask[
                        i,
                        j,
                        i - args.size_mask_neighborhood + p,
                        j - args.size_mask_neighborhood + q,
                    ] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if mask_cnt.max() > 0:
            mask_cnt = mask_cnt - mask_cnt.min()
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


def label_propagation(
    args, model, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None
):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    ## we only need to extract feature of the target frame
    feat_tar, h, w = extract_feature(model, frame_tar, return_h_w=True)

    return_feat_tar = feat_tar.T  # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats)  # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(
        torch.bmm(feat_tar, feat_sources) / args.temperature
    )  # nmb_context x h*w (tar: query) x h*w (source: keys)

    if args.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(
        -1, h * w
    )  # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = (
        segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T
    )  # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    return seg_tar, return_feat_tar, mask_neighborhood


def extract_feature(model, frame, return_h_w=False):
    """Extract one frame feature everytime."""
    # out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)

    out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame.shape[1] / model.patch_embed.patch_size), int(
        frame.shape[2] / model.patch_embed.patch_size
    )
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    out = out.reshape(-1, dim)
    if return_h_w:
        return out, h, w
    return out


def imwrite_indexed(filename, array, color_palette):
    """Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format="PNG")


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if n_dims is None:
        n_dims = int(y_tensor.max() + 1)
    _, h, w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h, w, n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir, "*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list


def read_frame(frame_dir, scale_size=[480]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if ori_h > ori_w:
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w


def read_seg(seg_dir, factor, scale_size=[480]):
    seg = Image.open(seg_dir)
    _w, _h = seg.size  # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if _w > _h:
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 64) * 64)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 64) * 64)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return to_one_hot(small_seg), np.asarray(seg)


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation with video object segmentation on DAVIS 2017")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--finetune", default="", type=str,
                        help="Path to pretrained weights to evaluate.")
    parser.add_argument("--model", default="vit_small", type=str)
    parser.add_argument( "--output_dir", default=".")
    parser.add_argument("--data_path", default="/data/DAVIS_480_880", type=str)

    parser.add_argument("--n_last_frames", type=int, default=30)
    parser.add_argument("--size_mask_neighborhood", default=30, type=int)
    parser.add_argument("--topk", type=int, default=7)

    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)

    args = parser.parse_args()

    print(f'[topk {args.topk}] [neighborhood {args.size_mask_neighborhood}] [queue {args.n_last_frames}]')

    # building network
    model = vits.__dict__[args.model](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.model} {args.patch_size}x{args.patch_size} built.")
    model.cuda()

    ckpt = 'model'
    load_pretrained_weights(model, args.finetune, ckpt)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    color_palette = []
    with open('./util/data_palette.txt', 'r') as file:
        lines = file.readlines()
    for line in lines:
        color_palette.append([int(i) for i in line.split("\n")[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1, 3)

    video_list = open(
        os.path.join(args.data_path, "ImageSets/2017/val.txt")
    ).readlines()
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        print(f"[{i}/{len(video_list)}] Begin to segmentate video {video_name}.", end='\r')
        video_dir = os.path.join(args.data_path, "JPEGImages/480p/", video_name)
        frame_list = read_frame_list(video_dir)
        seg_path = (
            frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
        )
        first_seg, seg_ori = read_seg(seg_path, args.patch_size)
        eval_video_tracking_davis(
            args, model, frame_list, video_dir, first_seg, seg_ori, color_palette
        )

