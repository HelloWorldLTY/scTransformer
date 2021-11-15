# Copyright © 2019 Jiaxin Li <jli2274@wisc.edu>
# Distributed under terms of the MIT license.


# Requires expression matrix and meta file
# expr : gene by cell with header (pandas Dataframe object)
# meta: cell by observations (pandas Dataframe object)
# label name: the column name of the label colum in meta file
class scRNACSV(Dataset):
  def __init__(self, expr, meta, label_Name, instance = False, transform = None, target_transform=None):
    self.expr = expr
    self.meta = meta
    self.ifInstance = instance
    self.cells = list(expr.columns)
    self.labels = list(meta[label_Name])
    self.samples = [(self.cells,self.labels[i]) for i in range(len(self.labels))]
    self.label_keys = list(set(self.labels))
    self.label_keys.sort()
    self.label_dic = {}
    for label, i in zip(self.label_keys, range(len(self.label_keys))):
      self.label_dic[label] = i
    print(f"This is the label dictionary of this dataset {self.label_dic}")
    
    self.transform = transform
    self.target_transform = target_transform
  def __len__(self):
    return self.expr.shape[1]

  def __getitem__(self,idx):
    one_cell = torch.from_numpy(np.array(self.expr.iloc[:,idx]))
    if self.transform:
      ret = self.transform(one_cell)
    else:
      ret = one_cell
    lab = self.label_dic[self.labels[idx]]
    if self.ifInstance:
        lab = idx
    return ret, lab
