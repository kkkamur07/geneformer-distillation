# Custom Dataset class for loading Geneformer data.
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
    