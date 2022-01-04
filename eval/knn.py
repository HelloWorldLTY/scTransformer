# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import models.vits as vits
import models.perceiver_pytorch as Perceiver
from GeneSetCrop import GeneCrop
from DataLoaders import scRNACSV

def get_args_parser():
    parser = argparse.ArgumentParser('knn', add_help=False)

    parser.add_argument('--crop_size', default=1, type=float, help="""The crop size""", required=True)
    parser.add_argument('--seed', default=0, type=int, help="""Random seed""")
    parser.add_argument('--model_category', default='vit', type=str, choices=['vit','Perceiver'], help="""""")
    parser.add_argument('--fuse_mode', default='cat', choices=['cat', 'add'], type=str,
                        help="""Specify the fuse mode for Perceiver models""")
    parser.add_argument('--output_dir', default='./', type=str, help="""Out put directory""",
                        required=True)

    # Perceiver parameters
    parser.add_argument('--num_latents', type=int, default=16, help="""N of latent array""")
    parser.add_argument('--latent_dim', type=int, default=256, help="D of latent array")
    parser.add_argument('--heads', default=2, type=int,
                        help="""Number of multi-heads""")
    parser.add_argument('--gene_embed', default=128, type=int,
                        help="""gene embedding dimension""")
    parser.add_argument('--expression_embed', default=128, type=int,
                        help="""expression embedding dimension""")

    parser.add_argument('--batch_size_per_gpu', default=16, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="(Folder) Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")

    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")', required=True)
    parser.add_argument('--dump_features', default=None,
                        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
            already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--expr_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the expression matrix.')
    parser.add_argument('--meta_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the meta file.')
    parser.add_argument('--label_name', default='perturb', type=str,
                        help='Please specify the name of label column in the meta file.')
    parser.add_argument('--start_epoch', default=0, type=int, help="""The epoch to start the evaluation""")
    parser.add_argument('--end_epoch', default=480, type=int, help="""The epoch to end the evaluation""")
    parser.add_argument('--checkpoint_frequency', default=20, type=int, help="""Fr""")

    return parser


def extract_feature_pipeline(args,data_loader_train,data_loader_val, epoch, gene_number):
    pretrained_weights = args.pretrained_weights + '/checkpoint' + f'{epoch:04d}.pth'
    # ============ building network ... ============
    if args.model_category == 'vit':
        model = vits.__dict__['vit_cat'](gene_number=gene_number)
    elif args.model_category == 'Perceiver':
        model = Perceiver.Perceiver(
            fuse_mode=args.fuse_mode,
            depth=3,
            num_latents=args.num_latents,
            latent_dim=args.latent_dim,
            cross_heads=args.heads,
            latent_heads=args.heads,
            gene_number=gene_number,
            gene_embed=args.gene_embed,
            expression_embed=args.expression_embed
        )
    print(f"Model {args.model_category} built.")
    model.cuda()
    utils.load_pretrained_weights(model, pretrained_weights, args.checkpoint_key)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if args.use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if args.use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5



if __name__ == '__main__':
    parser = argparse.ArgumentParser('knn', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    # Only need load data one time
    expr = pd.read_csv(args.expr_path, index_col=0)
    meta = pd.read_csv(args.meta_path, index_col=0)
    gene_number = expr.shape[0]

    print(f'This dataset has {gene_number} genes!')

    crop = GeneCrop(args.crop_size)

    dataset = scRNACSV(expr, meta, args.label_name, instance=True, transform=crop)

    trainset_length = int(len(dataset) * 0.8)
    testset_length = len(dataset) - trainset_length
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [trainset_length, testset_length],
                                                               generator=torch.Generator().manual_seed(args.seed))

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val cells.")

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        epoch_range = np.arange(args.start_epoch, args.end_epoch, args.checkpoint_frequency)
        for epoch in epoch_range:
            train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args, data_loader_train, data_loader_val, epoch, gene_number)

            if utils.get_rank() == 0:
                if args.use_cuda:
                    train_features = train_features.cuda()
                    test_features = test_features.cuda()
                    train_labels = train_labels.cuda()
                    test_labels = test_labels.cuda()

                print("Features are ready!\nStart the k-NN classification.")
                file_name = args.output_dir + '/' + args.checkpoint_key + "_knn_acc.txt"
                with open(file_name, mode='a') as f:
                    print(f"epoch: {epoch}, ", file=f, end="")
                    for k in args.nb_knn:
                        top1, top5 = knn_classifier(train_features, train_labels,
                            test_features, test_labels, k, args.temperature)
                        print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
                        if k == 10:
                            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}", file = f)
                f.close()
            dist.barrier()


# import umap
# from sklearn.preprocessing import StandardScaler
# import scprep
#
# te_features = test_features.cpu().numpy()
# te_labels = test_labels.cpu().numpy()
# # ================= visualization ... ========================
# umap_operator = umap.UMAP(n_components=2)
# data_umap = umap_operator.fit_transform(te_features)
# figure_name = arg.epochs + "_umap.png"
# scprep.plot.scatter2d(data_umap, c=te_labels, cmap='Spectral', # colormap
#                       ticks=False, label_prefix='umap', title="FashionMNIST",
#                       legend_anchor=(1,1), figsize=(7,5),filename=figure_name))
