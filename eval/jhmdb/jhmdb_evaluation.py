from __future__ import print_function

import os
import time
import random
import argparse

import numpy as np

import torch

import jhmdb_evaluation.utils as utils
import jhmdb_evaluation.datasets
from util.load_ckpt import load_pretrained_weights
import vision_transformer as vits

def main(args):
    model = vits.__dict__[args.model](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.model} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    if 'mocov3' in args.resume:
        ckpt = 'state_dict'
    elif 'dino' in args.resume:
        ckpt = 'student'
    else:
        ckpt = 'model'
    load_pretrained_weights(model, args.resume, ckpt)

    args.mapScale = utils.infer_downscale(model)

    #args.use_lab = args.model_type == 'uvc'
    dataset = jhmdb_evaluation.datasets.JhmdbSet(args)
    val_loader = torch.utils.data.DataLoader(dataset,
        batch_size=int(args.batchSize), shuffle=False, num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with torch.no_grad():
        test_loss = test(val_loader, model, args)


def test(loader, model, args):
    n_context = args.videoLen
    D = None    # Radius mask

    for vid_idx, (imgs, imgs_orig, lbls, lbl_map) in enumerate(loader):
        t_vid = time.time()
        imgs = imgs.to(args.device)
        B, N = imgs.shape[:2]
        assert(B == 1)

        print('******* Vid %s (%s frames) *******' % (vid_idx, N), end='\r')
        with torch.no_grad():
            t00 = time.time()

            ##################################################################
            # Compute image features (batched for memory efficiency)
            ##################################################################
            bsize = 1   # minibatch size for computing features
            feats = []
            for b in range(0, imgs.shape[1], bsize):
                feat = imgs[:, b:b+bsize]
                feat = feat.reshape(*feat.shape[-3:])
                feat = utils.extract_feature(model, feat)
                dim = feat.shape[1]
                feat = feat.permute(1, 0).reshape(1, dim, 1, args.cropSize//args.patch_size, args.cropSize//args.patch_size)
                feats.append(feat.cpu())
            feats = torch.cat(feats, dim=2).squeeze(1)

            if not args.no_l2:
                feats = torch.nn.functional.normalize(feats, dim=1)

            ##################################################################
            # Compute affinities
            ##################################################################
            torch.cuda.empty_cache()
            t03 = time.time()

            # Prepare source (keys) and target (query) frame features
            key_indices = utils.context_index_bank(n_context, args.long_mem, N - n_context)
            key_indices = torch.cat(key_indices, dim=-1)
            keys, query = feats[:, :, key_indices], feats[:, :, n_context:]

            # Make spatial radius mask TODO use torch.sparse
            restrict = utils.MaskedAttention(args.radius, flat=False)
            D = restrict.mask(*feats.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0

            # Flatten source frame features to make context feature set
            keys, query = keys.flatten(-2), query.flatten(-2)

            Ws, Is = utils.mem_efficient_batched_affinity(query, keys, D,
                        args.temperature, args.topk, args.long_mem, args.device)

            ##################################################################
            # Propagate Labels and Save Predictions
            ###################################################################

            keypts = []
            lbls[0, n_context:] *= 0
            lbl_map, lbls = lbl_map[0], lbls[0]

            for t in range(key_indices.shape[0]):
                # Soft labels of source nodes
                ctx_lbls = lbls[key_indices[t]].to(args.device)
                ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

                # Weighted sum of top-k neighbours (Is is index, Ws is weight)
                pred = (ctx_lbls[:, Is[t]] * Ws[t].to(args.device)[None]).sum(1)
                pred = pred.view(-1, *feats.shape[-2:])
                pred = pred.permute(1,2,0)

                if t > 0:
                    lbls[t + n_context] = pred
                else:
                    pred = lbls[0]
                    lbls[t + n_context] = pred

                if args.norm_mask:
                    pred[:, :, :] -= pred.min(-1)[0][:, :, None]
                    pred[:, :, :] /= pred.max(-1)[0][:, :, None]

                # Save Predictions
                cur_img = imgs_orig[0, t + n_context].permute(1,2,0).numpy() * 255

                if 'jhmdb' in args.filelist.lower():
                    coords, pred_sharp = utils.process_pose(pred, lbl_map)
                    keypts.append(coords)

                outpath = os.path.join(args.save_path, str(vid_idx) + '_' + str(t))

                utils.dump_predictions(
                    pred.cpu().numpy(),
                    lbl_map, cur_img, outpath)


            if len(keypts) > 0:
                coordpath = os.path.join(args.save_path, str(vid_idx) + '.dat')
                np.stack(keypts, axis=-1).dump(coordpath)

            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label Propagation')

    parser.add_argument('--patch_size', type=int, default=16)
    # Datasets
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--manualSeed', type=int, default=777, help='manual seed')

    #Device options
    parser.add_argument('--batchSize', default=1, type=int,
                        help='batchSize')
    parser.add_argument('--temperature', default=0.05, type=float,
                        help='temperature')
    parser.add_argument('--topk', default=10, type=int,
                        help='k for kNN')
    parser.add_argument('--radius', default=5, type=float,
                        help='spatial radius to consider neighbors from')
    parser.add_argument('--videoLen', default=7, type=int,
                        help='number of context frames')

    parser.add_argument('--cropSize', default=320, type=int,
                        help='resizing of test image, -1 for native size')

    parser.add_argument('--root', default='/data')
    parser.add_argument('--filelist', default='/data/JHMDB/val_list.txt', type=str)
    parser.add_argument('--save-path', default='./results', type=str)

    # Model Details
    parser.add_argument('--model', default='vit_small', type=str, help='Architecture (support only ViT atm).')

    parser.add_argument('--model-type', default='scratch', type=str)
    parser.add_argument('--head-depth', default=-1, type=int,
                        help='depth of mlp applied after encoder (0 = linear)')

    parser.add_argument('--remove-layers', default=['layer4'], help='layer[1-4]')
    parser.add_argument('--no-l2', default=False, action='store_true', help='')

    parser.add_argument('--long-mem', default=[0], type=int, nargs='*', help='')
    parser.add_argument('--texture', default=False, action='store_true', help='')
    parser.add_argument('--round', default=False, action='store_true', help='')

    parser.add_argument('--norm_mask', default=False, action='store_true', help='')
    parser.add_argument('--finetune', default=0, type=int, help='')

    args = parser.parse_args()

    args.n_last_frames = args.videoLen
    use_cuda = torch.cuda.is_available()
    args.device = 'cuda'

    # Set seed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    args.imgSize = args.cropSize
    print('Context Length:', args.videoLen, 'Image Size:', args.imgSize)
    print('Arguments', args)

    main(args)