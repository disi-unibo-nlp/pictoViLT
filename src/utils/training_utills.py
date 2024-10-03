import os
from torch.utils.data import DataLoader

def collate_fn(batch):
    # Remove items with None values from the batch
    batch = [item for item in batch if item is not None]

    # Extract specific elements from each item in the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]

    # Create a new batch by stacking tensors for each element
    collated_batch = {}
    collated_batch['input_ids'] = torch.stack(input_ids)
    collated_batch['attention_mask'] = torch.stack(attention_mask)
    collated_batch['token_type_ids'] = torch.stack(token_type_ids)
    collated_batch['labels'] = torch.stack(labels)
    collated_batch['pixel_values'] = torch.stack(pixel_values)

    return collated_batch
