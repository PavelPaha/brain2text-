from utils import get_labels_transitions_ids
import pandas as pd



def get_raw_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    return dataset


def get_dataset_with_diphones(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset['diphone_ids'] = dataset['phonems'].apply(get_labels_transitions_ids)
    return dataset