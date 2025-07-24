import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def get_qpos_from_hdf5(data_file_path):
    """Extract qpos data from HDF5 file"""
    with h5py.File(data_file_path, 'r') as root:
        return np.array(root['/observations/qpos'][()])

def collate_fn(batch):
    # Stack all tensors at once
    return torch.stack(batch)


class QposDataset(Dataset):
    """Dataset for qpos data only (no images or videos)
    
    This dataset only loads joint position data (qpos) without any image or video processing.
    It can load data from either HDF5 files or PT files.
    """
    
    def __init__(self, dataset_path, disable_pbar=False, type="train"):
        self.data = []
        self.type = type
        self.dataset_path = dataset_path

        self.qpos_data = []  # Store qpos tensors
        self.qpos_lengths = []  # Store lengths of each qpos sequence
        
        for task_name in os.listdir(dataset_path):
            task_path = os.path.join(dataset_path, task_name)
            if not os.path.isdir(task_path):
                continue
            
            for file_name in tqdm(os.listdir(task_path), desc=f"Loading qpos from task {task_name}", disable=disable_pbar):
                if file_name.endswith('_qpos.pt'):
                    qpos_path = os.path.join(task_path, file_name)
                    
                    # Load qpos
                    qpos = torch.load(qpos_path)
                    
                    if len(qpos) < self.select_num:
                        print(f"Skipping {qpos_path} - too short")
                        continue
                        
                    self.qpos_data.append(qpos)
                    self.qpos_lengths.append(len(qpos) - (self.select_num - 1))
        
        self.data_begin = np.cumsum([0] + self.qpos_lengths[:-1])
        self.data_end = np.cumsum(self.qpos_lengths)

    def __len__(self):
        return self.data_end[-1] if self.qpos_lengths else 0

    def __getitem__(self, idx):
        # Find which qpos sequence this index corresponds to
        seq_idx = np.searchsorted(self.data_end, idx, side='right')
        if seq_idx >= len(self.qpos_data):  # Add bounds check
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.qpos_data)} sequences")
        
        local_idx = idx - self.data_begin[seq_idx]
        qpos_tensor = self.qpos_data[seq_idx]
        
        pos = qpos_tensor[local_idx].float()
        return pos

    def __del__(self):
        pass
