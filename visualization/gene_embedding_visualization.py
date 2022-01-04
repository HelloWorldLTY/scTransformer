import models.vits as vits
import utils
import umap
import matplotlib.pyplot as plt
import scprep
import argparse
import models.perceiver_pytorch as Perceiver
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser('GeneEmbedding', add_help=False)

    parser.add_argument('--checkpoint_path', default='./', type=str,
                        help="""The path to the checkpoint""")
    parser.add_argument('--gene_number', default=2000, type=int,
                        help="""The number of genes in the model""")
    parser.add_argument('--model_category', default='vit', choices=['vit', 'Perceiver'],
                        type=str,
                        help="""The category of model""")
    parser.add_argument('--model_name', default='vit_cat', choices=['vit_cat', 'vit_add'],
                        type=str,
                        help="""Model names""")
    parser.add_argument('--fuse_mode', default='cat', choices=['cat', 'add'], type=str,
                        help="""Specify the fuse mode for Perceiver models""")
    parser.add_argument('--output_dir', default='./', type=str,
                        help="""The output path of the figures""")
    # Perceiver parameters
    parser.add_argument('--num_latents', type=int, default=16, help="""N of latent array""")
    parser.add_argument('--latent_dim', type=int, default=256, help="D of latent array")
    parser.add_argument('--heads', default=2, type=int,
                        help="""Number of multi-heads""")
    parser.add_argument('--gene_embed', default=128, type=int,
                        help="""gene embedding dimension""")
    parser.add_argument('--expression_embed', default=128, type=int,
                        help="""expression embedding dimension""")
    return parser

def gene_embedding_visualization(args):
    if args.model_category == 'vit':
        model = vits.__dict__[args.model_name](gene_number=args.gene_number)
    elif args.model_category == 'Perceiver':
        model = Perceiver.Perceiver(
            fuse_mode=args.fuse_mode,
            depth=3,
            num_latents=args.num_latents,
            latent_dim=args.latent_dim,
            cross_heads=args.heads,
            latent_heads=args.heads,
            gene_number=args.gene_number,
            gene_embed=args.gene_embed,
            expression_embed=args.expression_embed
        )
    utils.load_pretrained_weights(model, args.checkpoint_path, "student")
    pos_embed = model.state_dict().get('Embedding.weight').cpu()

    print(f'The shape of positional embedding is {pos_embed.shape}')
    ## Secondly, calculate 784 pixels' embedding coordinates
    emb_cor = pos_embed  # emb_cor.shape shoule be (784, coordinates dimensions). perhaps you need pandas.DataFrame here

    umap_operator = umap.UMAP(n_components=2, n_neighbors=30)  # n_components = 2: 2-dim umap
    Y_UMAP = umap_operator.fit_transform(emb_cor)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    scprep.plot.scatter2d(Y_UMAP, label_prefix="umap", title="umap_x",
                          # c=x_cor,
                          ticks=False, cmap='Spectral', ax=ax1)
    scprep.plot.scatter2d(Y_UMAP, label_prefix="umap", title="umap_y",
                          # c=y_cor,
                          ticks=False, cmap='Spectral', ax=ax2)
    scprep.plot.scatter2d(Y_UMAP, label_prefix="umap", title="umap_l2_norm",
                          # c=l2_norm,
                          ticks=False, cmap='Spectral', ax=ax3)

    plt.tight_layout()
    fig.savefig(args.output_dir, dpi=100)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GeneEmbedding', parents=[get_args_parser()])
    args = parser.parse_args()

    gene_embedding_visualization(args)