import torch
from typing import List, Dict, Any

class GeneDataCollator:
    def __init__(self, vocab_size: int = 25426, mask_token_id: int = 1, pad_token_id: int = 0, mlm_probability: float = 0.15):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mlm_probability = mlm_probability

    def pad_batch(self, examples: List[Dict[str, Any]]):
        max_len = max(x['length'] for x in examples)
        
        batch_input_ids = []
        batch_attention_mask = []
        
        for item in examples:
            ids = item['input_ids']
            curr_len = len(ids)
            
            # Create padded tensor
            padded_ids = torch.full((max_len,), self.pad_token_id, dtype=torch.long)
            padded_ids[:curr_len] = ids
            batch_input_ids.append(padded_ids)
            
            # Create attention mask
            mask = torch.zeros(max_len, dtype=torch.long)
            mask[:curr_len] = 1
            batch_attention_mask.append(mask)
            
        return torch.stack(batch_input_ids), torch.stack(batch_attention_mask)

    def __call__(self, examples):
        input_ids, attention_mask = self.pad_batch(examples)
        labels = input_ids.clone()
        
        # Create Probability mask matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = (input_ids == self.pad_token_id)
        
        # Set probability of special tokens to 0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 5. Set Labels : Only calculate loss for masked tokens
        labels[~masked_indices] = -100
        
        # BERT Masking Strategy : 80% masked
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10% random token and rest left unchanged
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
