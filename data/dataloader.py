from torch.utils.data import DataLoader

class Loader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4):
        super(Loader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return Loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

