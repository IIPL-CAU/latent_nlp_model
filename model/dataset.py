import torch
from torch.utils.data.dataset import Dataset

import albumentations as A
from skimage import io

class Seq2SeqDataset(Dataset):
    def __init__(self, src_list: list, src_att_list: list, src_img_path: list = None,
                 trg_list: list = None, trg_att_list: list = None,
                 src_max_len: int = 300, trg_max_len: int = 360,
                 pad_idx: int = 0, eos_idx: int = 2,
                 image_transform: A.core.composition.Compose = None):
        # Stop index list
        stop_ix_list = [pad_idx, eos_idx]
        self.tensor_list = []
        for src, src_att, trg, trg_att in zip(src_list, src_att_list, trg_list, trg_att_list):
            if src[src_max_len-1] in stop_ix_list and trg[trg_max_len-1] in stop_ix_list:
                # Source tensor
                src_tensor = torch.tensor(src[:src_max_len], dtype=torch.long)
                src_att_tensor = torch.tensor(src_att[:src_max_len], dtype=torch.long)
                # Target tensor
                trg_tensor = torch.tensor(trg[:trg_max_len], dtype=torch.long)
                trg_att_tensor = torch.tensor(trg_att[:trg_max_len], dtype=torch.long)
                # tensor list
                self.tensor_list.append((src_tensor, src_att_tensor, trg_tensor, trg_att_tensor))

        self.num_data = len(self.tensor_list)

    def __getitem__(self, index):
        return self.tensor_list[index]

    def __len__(self):
        return self.num_data

class Seq2LabelDataset(Dataset):
    def __init__(self, src_list: list, src_att_list: list, src_img_path: list = None,
                 trg_list: list = None, trg_att_list: list = None,
                 min_len: int = 4, src_max_len: int = 300, 
                 pad_idx: int = 0, eos_idx: int = 2,
                 image_transform: A.core.composition.Compose = None):
        self.tensor_list = []
        for src, src_att, trg in zip(src_list, src_att_list, trg_list):
            if min_len <= len(src) <= src_max_len:
                # Source tensor
                src_tensor = torch.zeros(src_max_len, dtype=torch.long)
                src_tensor[:len(src)] = torch.tensor(src, dtype=torch.long)
                src_att_tensor = torch.tensor(src_att, dtype=torch.long)
                # Target tensor
                trg_tensor = torch.tensor(trg, dtype=torch.long)
                #
                self.tensor_list.append((src_tensor, src_att_tensor, trg_tensor))

        self.num_data = len(self.tensor_list)

    def __getitem__(self, index):
        return self.tensor_list[index]

    def __len__(self):
        return self.num_data

class MutlimodalClassificationDataset(Dataset):
    def __init__(self, src_list: list, src_att_list: list, src_img_path: list = None, 
                 trg_list: list = None, trg_att_list: list = None,
                 min_len: int = 4, src_max_len: int = 300, 
                 pad_idx: int = 0, eos_idx: int = 2,
                 image_transform: A.core.composition.Compose = None):
        self.tensor_list = []
        self.image_transform = image_transform
        # For Inference
        if trg_list is None:
            trg_list = [0 for _ in range(len(src_list))]
        for src, src_att, img_path, trg in zip(src_list, src_att_list, src_img_path, trg_list):
            if min_len <= len(src) <= src_max_len:
                # Source text tensor
                src_tensor = torch.zeros(src_max_len, dtype=torch.long)
                src_tensor[:len(src)] = torch.tensor(src, dtype=torch.long)
                src_att_tensor = torch.tensor(src_att, dtype=torch.long)
                # Target tensor
                trg_tensor = torch.tensor(trg, dtype=torch.long)
                # tensor list
                self.tensor_list.append((src_tensor, src_att_tensor, img_path, trg_tensor))

        self.num_data = len(self.tensor_list)

    def __getitem__(self, index):
        src_tensor, src_att_tensor, img_path, trg_tensor = self.tensor_list[index]
        # Image load
        image = io.imread(img_path.decode('utf-8'))
        transformed_image = self.image_transform(image=image)['image']
        return src_tensor, src_att_tensor, transformed_image, trg_tensor

    def __len__(self):
        return self.num_data