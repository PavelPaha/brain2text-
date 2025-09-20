import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    return {
        'day_idx': torch.tensor([item['day_idx'] for item in batch]),
        'neural_data': pad_sequence([torch.tensor(item['neural_data']) for item in batch], batch_first=True),
        'phonemes_ids': pad_sequence([torch.tensor(item['phonemes_ids']) for item in batch], batch_first=True, padding_value=0)
    }
    
class Loader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4):
        super(Loader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return Loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
