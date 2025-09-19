from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, data: pd.DataFrame, output_size):
        self.data = data
        self.output_size = output_size
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        res = self.data.iloc[idx].to_dict()
        # print(res)
        return res


def create_dataset(data: pd.DataFrame, output_size) -> MyDataset:
    return MyDataset(data, output_size=output_size)