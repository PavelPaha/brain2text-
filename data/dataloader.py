import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    return {
        'neural_data': torch.nn.utils.rnn.pad_sequence([torch.tensor(item['neural_data']) for item in batch], batch_first=True),
        'phonems_ids': torch.nn.utils.rnn.pad_sequence([torch.tensor(item['phonems_ids']) for item in batch], batch_first=True, padding_value=0)
    }
    
class Loader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4):
        super(Loader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return Loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
