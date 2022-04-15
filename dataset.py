"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import os
import numpy as np
import copy

from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, root: str, buffer_size: int):
        """
        Stores train or test data. Given a root directory, filter out all
        files that are not .npz files. Then, store the data in a buffer with
        a maximum size of buffer_size.

        :param root: Root directory of the dataset.
        :param buffer_size: Maximum size of the buffer.
        """
        self.root = root
        self.files = os.listdir(root)
        self.files = [f for f in self.files if f[-3:] == "npz"]
        self.idx_to_buffer_idx = dict()
        buffer_size = min(buffer_size, len(self.files))
        random_indices = np.random.choice(len(self.files), buffer_size, replace=False)
        self.buffer = []
        for idx in random_indices:
            self.buffer.append(self[idx])
        self.idx_to_buffer_idx = {file_idx: buffer_idx for (buffer_idx, file_idx)
                                  in enumerate(random_indices)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.idx_to_buffer_idx:
            return self.buffer[self.idx_to_buffer_idx[idx]]

        fname = self.files[idx]
        with np.load(os.path.join(self.root, fname), allow_pickle=True) as data:
            return copy.deepcopy(dict(data))
