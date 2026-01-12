"""
Geneformer Dataset Implementation.

This module implements `GeneformerDataset`, a PyTorch Dataset wrapper around
Hugging Face Datasets (Arrow files). It handles:
- Efficient on-disk loading of large gene sequence datasets.
- Accessing sequences and their associated length properties.

Classes:
    GeneformerDataset: PyTorch map-style dataset for Geneformer data.

Usage Example:
    ```python
    dataset = GeneformerDataset("/path/to/dataset.dataset")
    print(len(dataset))
    
    item = dataset[0] # Returns dict with input_ids, length, etc.
    ```
"""
from torch.utils.data import Dataset
from datasets import load_from_disk

class GeneformerDataset(Dataset):
    def __init__(self, dataset_path_or_obj):
        if isinstance(dataset_path_or_obj, str):
            self.dataset = load_from_disk(dataset_path_or_obj).with_format("torch")
        else:
            self.dataset = dataset_path_or_obj

        self.length = self.dataset.data.column("length").to_numpy()
        
    def __len__(self):
        return len(self.length)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item
    