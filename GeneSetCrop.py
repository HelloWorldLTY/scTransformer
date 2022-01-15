# Copyright © 2019 Jiaxin Li <jli2274@wisc.edu>
# Distributed under terms of the MIT license.
import numpy as np
import torch
from torchvision import transforms

# Crop by a proportion
def crop_proportion(x, size):
    length = x.shape[0]
    crop_size = int(length * size)
    index = np.random.choice(length, size=crop_size, replace=False)
    coor = torch.from_numpy(index)
    data = x[index, ]
    one_input = torch.cat([data, coor]).float()
    return one_input

# Crop by a fixed number
def crop_number(x, number):
    length = x.shape[0]
    crop_size = number
    index = np.random.choice(length, size=crop_size, replace=False)
    coor = torch.from_numpy(index)
    data = x[index, ]
    input = torch.cat([data, coor]).float()
    return input

class GeneSetCrop(object):
    def __init__(self, global_crops_scale=0.25, local_crops_scale=0.125, local_crops_number=8,
                 fix_number=True, global_crop_gene_number=500, local_crop_gene_number=250):
      self.global_crops_scale = global_crops_scale
      self.local_crops_scale = local_crops_scale
      self.local_crops_number = local_crops_number
      self.fix_number = fix_number
      self.global_crop_gene_number = global_crop_gene_number
      self.local_crop_gene_number = local_crop_gene_number

    def __call__(self, x):
        inputs = []
        if self.fix_number:
            global1 = crop_number(x, self.global_crop_gene_number)
            global2 = crop_number(x, self.global_crop_gene_number)
            inputs.append(global1)
            inputs.append(global2)

            for i in range(self.local_crops_number):
                local = crop_number(x, self.local_crop_gene_number)
                inputs.append(local)
        else:
            global1 = crop_proportion(x, self.global_crops_scale)
            global2 = crop_proportion(x, self.global_crops_scale)
            inputs.append(global1)
            inputs.append(global2)

            for i in range(self.local_crops_number):
                local = crop_proportion(x, self.local_crops_scale)
                inputs.append(local)
        return inputs


class GeneCrop(object):
    def __init__(self, crop_size = 1.0):
        self.crop_size = crop_size

    def __call__(self, x):
        ret = crop_proportion(x, self.crop_size)
        return ret

class GeneSetCrop_wo_shuffle(object):
    def __init__(self, pseudoPara = 0):
      self.super = pseudoPara

    def __call__(self, x):
        inputs = []
        data = x
        length = x.shape[0]
        index = np.arange(length)

        corr = torch.from_numpy(index)
        input = torch.cat([data, corr]).float()
        return input

class FashionMNISTCrop(object):
  def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
    self.global_crops_scale = global_crops_scale
    self.local_crops_scale = local_crops_scale
    self.local_crops_number = local_crops_number
    self.totensor = transforms.ToTensor()
  def __call__(self, x):
    inputs = []
    data = self.totensor(x)
    data = torch.flatten(data)
    global_size = int(784 * self.global_crops_scale)
    local_size = int(784 * self.local_crops_scale)
    global_index_1 = np.random.choice(784, size = global_size, replace = False)
    global_index_2 = np.random.choice(784, size = global_size, replace = False)

    global_corr_1 = torch.from_numpy(global_index_1)
    global_corr_2 = torch.from_numpy(global_index_2)

    global_data_1 = data[global_index_1,]
    global_data_2 = data[global_index_2,]
    #print(data.shape)
    #print(corr.shape)
    #labels = torch.from_numpy(label.item()*np.ones(shape = (784,1))).float()
    global_input_1 = torch.cat([global_data_1, global_corr_1]).float()
    global_input_2 = torch.cat([global_data_2, global_corr_2]).float()
    inputs.append(global_input_1)
    inputs.append(global_input_2)
    for i in range(self.local_crops_number):
      index = np.random.choice(784, size = local_size, replace = False)
      corr = torch.from_numpy(index)
      data_local = data[index,]
      input_local = torch.cat([data_local, corr]).float()
      inputs.append(input_local)
    return inputs
