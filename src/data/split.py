# Utilities for splitting datasets into train and validation sets.
from pathlib import Path
from datasets import load_from_disk

OUTPUT_DIR = Path("/home/krrish/Desktop/Programming/geneformer-scratch/mainData")
DATA_PATH = "/home/krrish/Desktop/Programming/geneformer-scratch/mainData/genecorpus_30M_2048_int16.dataset"

train_output_path = OUTPUT_DIR / "train_corpus.dataset"
val_output_path = OUTPUT_DIR / "val_corpus.dataset"

def split():
    print("Loading dataset from disk")
    ds = load_from_disk(DATA_PATH)
    print("Splitting dataset into train and validation sets")
    split_ds = ds.train_test_split(test_size=0.0001, seed=100)
    
    train_ds = split_ds['train']
    val_ds = split_ds['test']
    
    print("Saving train and validation datasets to disk")
    train_ds.save_to_disk(str(train_output_path), num_shards=1)
    val_ds.save_to_disk(str(val_output_path), num_shards=1)
    
if __name__ == "__main__":
    split()