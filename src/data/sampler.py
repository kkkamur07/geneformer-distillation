import torch
from torch.utils.data import Sampler
import random
import numpy as np


class AdaptiveLengthGroupedSampler(Sampler):
    
    def __init__(self, lengths, base_batch_size=12, shuffle=True):
        self.base_batch_size = base_batch_size
        self.shuffle = shuffle
        
        # Keep as numpy array
        self.lengths = np.array(lengths, dtype=np.int16)
        self.sorted_indices = np.argsort(self.lengths)
        self.sorted_lengths = self.lengths[self.sorted_indices]
        
    def __iter__(self):
        batches = []
        i = 0
        
        while i < len(self.sorted_indices):
            current_length = self.sorted_lengths[i]
            
            # Adaptive batch size based on sequence length
            if current_length < 200:
                batch_size = self.base_batch_size * 4  # 48
            elif current_length < 500:
                batch_size = self.base_batch_size * 2  # 24
            elif current_length < 1000:
                batch_size = self.base_batch_size  # 12
            elif current_length < 1500:
                batch_size = self.base_batch_size // 2  # 6
            else:
                batch_size = self.base_batch_size // 3  # 4
            
            batch = self.sorted_indices[i:i + batch_size].tolist()
            batches.append(batch)
            i += batch_size
        
        if self.shuffle:
            random.shuffle(batches)
        
        for batch in batches:
            yield from batch
    
    def __len__(self):
        return len(self.sorted_indices)


class LengthGroupedSampler(Sampler):
    
    def __init__(self, lengths, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.lengths = np.array(lengths, dtype=np.int16)
        self.sorted_indices = np.argsort(self.lengths)
        
    def __iter__(self):
        batches = [
            self.sorted_indices[i:i + self.batch_size].tolist()
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        
        if self.shuffle:
            random.shuffle(batches)
        
        for batch in batches:
            yield from batch
    
    def __len__(self):
        return len(self.sorted_indices)


def collate_fn_dynamic_pad(batch, pad_token_id=0):
    """Dynamically pads batch to the longest sequence in that batch."""
    
    lengths = np.array([item['length'] for item in batch], dtype=np.int16)
    max_len = int(lengths.max())
    
    input_ids = []
    attention_masks = []
    
    for i, item in enumerate(batch):
        seq = np.array(item['input_ids'], dtype=np.int16)
        seq_len = lengths[i]
        
        padded_seq = np.pad(seq, (0, max_len - seq_len), constant_values=pad_token_id)
        mask = np.concatenate([np.ones(seq_len, dtype=bool), np.zeros(max_len - seq_len, dtype=bool)])
        
        input_ids.append(padded_seq)
        attention_masks.append(mask)
    
    return {
        'input_ids': torch.from_numpy(np.array(input_ids, dtype=np.int16)),
        'attention_mask': torch.from_numpy(np.array(attention_masks, dtype=bool)),
    }