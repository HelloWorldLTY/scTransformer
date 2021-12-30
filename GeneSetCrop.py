# Copyright © 2019 Jiaxin Li <jli2274@wisc.edu>
# Distributed under terms of the MIT license.
import numpy as np
import torch


def crop(x, size):
    length = x.shape[0]
    crop_size = int(length * size)
    index = np.random.choice(length, size=crop_size, replace=False)
    coor = torch.from_numpy(index)
    data = x[index, ]
    input = torch.cat([data, coor]).float()
    return input


class GeneSetCrop(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
      self.global_crops_scale = global_crops_scale
      self.local_crops_scale = local_crops_scale
      self.local_crops_number = local_crops_number

    def __call__(self, x):
        inputs = []
        global1 = crop(x, self.global_crops_scale)
        global2 = crop(x, self.global_crops_scale)
        inputs.append(global1)
        inputs.append(global2)

        for i in range(self.local_crops_number):
            local = crop(x, self.local_crops_scale)
            inputs.append(local)
        return inputs
