# scTransformer
## Input Data Format
* Expression matrix\
A **gene by cell** matrix stored in .csv file, with row names (gene name) and column names (cell names)
* Meta matrix\
A **cell by label** matrix stored in .csv file.
### Prepared data
* bmcite

* 
* 
## Scripts
### Usage

```bash
python3 main.py --expr_path "/content/drive/Shareddrives/Documentation/Data/bm/rna_scale_bmcite.csv" --meta_path "/content/drive/Shareddrives/Documentation/Data/bm/meta.csv" --label_name "celltype.l2" --output_dir '../result' --fuse_mode "cat" --fix_number True
```
Standarize the code we have in the notebooks now:
* Delete the unused parameters and write documents of the input out parameters.
* Write README file and Add comments to the scripts.
* 

Scripts to finish (Final editing: NAME, DATE)
* [X] scDataLoader.py (Jiaxin Nov 20)
* [X] GeneSetCrop.py (Jiaxin Dec 29)
* [ ] GeneSetCropFixGeneNumber.py (Jiaxin Dec 29)
* [X] main.py (Jiaxin Dec 29)
* [X] model.py (Jiaxin Dec 29)
  - [X] model_add.py
  - [X] model_cat.py
  - [ ] Perceiver.py (Jiaxin Dec 29)
* [ ] knn_acc.py (Michelle)
* [ ] attention_visualization.py (Jie)
* [ ] gene_embedding.py (Jiaxin)
* [ ] GRN_construction.py 

Saturday\ 
TODO\ 
#### Dataset (Sat ~ Mon)
Cell nomalization + log transformation
* [X] bmcite 
* [ ] perturbation immunology(Jiaxin)
* [ ] perturbation (Michelle)
* [ ] Sciplex (Jie)

#### Visualization (Sat)

#### Model
* [ ] Perceiver
* [ ] Attention Downstream analysis (class token) ~ Cell type biomarker
* [ ] Attention map of regulators ~ GRN 
* [ ] gene embedding 




Dec 28th

Three people meeting 
Topics to talk about
1. Main, model, dataloader
2. Local GPU set up; Yale cluster set up (Training time and resource estimation)
3. Version control (Pycharm + github)



January ?th

Deadline for standarize the evluation code
1. knn (curve.. tracking...) (Michelle)
2. gene analogy (Jiaxin: see Webser paper)
3. attention (Jie)
4. Figures:

   knn curve (knn.py)
   
   loss curve (main.py)

   Cell embedding (knn.py)

   Gene emebdding (geneEmbedding.py)

   Attention map (attentionMap.py)

January ?th

Sciplex results anlyses





Parameter:
Depth
head
gene embedding dimension
expression dimension
crop size

