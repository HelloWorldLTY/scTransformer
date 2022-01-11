import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from functools import reduce
import seaborn as sns
import cmasher as cmr

import utils
import models.vits as vits
from DataLoaders import scRNACSV
from GeneSetCrop import GeneSetCrop_wo_shuffle
from utils import trunc_normal_

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention')
    parser.add_argument('--fuse_mode', default='cat',
                        choices=['add', 'cat'],
                        type=str,
                        help="""The mode of fusing gene embedding and expression embedding""")
    #parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--expr_path', default=None, type=str,
        help='Please specify path to the expression matrix.')
    parser.add_argument('--meta_path', default=None, type=str,
        help='Please specify path to the meta file.')
    parser.add_argument('--label_name', default='perturb', type=str,
                        help='Please specify the name of label column in the meta file.')
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    args = parser.parse_args()
    parser.add_argument('--num_features', default=100, type=int, help='The number of top k variable genes in each label group.')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    if args.fuse_mode == 'cat':
      model = vits.__dict__['vit_cat']()
    elif args.fuse_mode == 'add':
      model = vits.__dict__['vit_add']()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        print("There is no reference weights available for this model => We use random weights.")

    # load data
    if args.expr_path is None:
        print("Please use the `--expr_path` argument to indicate the path of the expression matrix you wish to visualize.")
    elif args.meta_path is None:
        print("Please use the `--meta_path` argument to indicate the path of the corresponding meta data.")
    elif os.path.isfile(args.expr_path) and os.path.isfile(args.meta_path):
        expr = pd.read_csv(args.expr_path, index_col=0)
        meta = pd.read_csv(args.meta_path, index_col=0)
        gene_number = expr.shape[0]
    else:
        print(f"Provided expression path {args.expr_path} or meta path {args.meta_path} is non valid.")
        sys.exit(1)

    crop = GeneSetCrop_wo_shuffle()
    dataset = scRNACSV(expr, meta, args.label_name, instance=False, transform=crop)

    attention = [None] * len(dataset)
    label = np.empty(len(dataset))
    for i in range(len(dataset)):
      cell, label[i] = dataset[i]
      attentions = model.get_last_selfattention(cell.reshape(1,gene_number*2).cuda())
      attentions = attentions[0, :, 0, 1:]
      attention[i] = attentions.cpu().detach().numpy()
    attention = np.stack(attention, axis=1)
    nh = attention.shape[0]

    hvag = [None] * len(np.unique(label))
    for l in range(len(np.unique(label))):
      idx = np.where(label == l)
      att = np.std(attention[:,idx,:], (1,2))
      hvag[l] = att.argsort()[::-1][:,:args.num_features]
    hvag = np.stack(hvag, axis=1)
    
    order=meta[args.label_name].argsort()
    col_dic = cmr.take_cmap_colors('YlGnBu',len(np.unique(label)),return_fmt='hex')
    label_groups = [col_dic[i] for i in label[order].astype(int)]
    os.makedirs(args.output_dir, exist_ok=True)
    for h in range(nh):
        union = reduce(np.union1d, hvag[h])
        expr_hvag = expr.iloc[union, :]
        expr_hvag = expr_hvag.loc[:,meta['product_name'][order].index]
        col_colors = pd.DataFrame(label_groups,columns =['labels'],index=expr_hvag.columns)
        plt = sns.clustermap(expr_hvag, 
                           col_cluster=False,
                           figsize = (10,10),
                           col_colors = col_colors, 
                           cmap="viridis",
                           standard_scale=1,
                           xticklabels=False)
        fname = os.path.join(args.output_dir, "attn-head" + str(h) + ".png")
        plt.savefig(fname=fname)
        print(f"{fname} saved.")    
