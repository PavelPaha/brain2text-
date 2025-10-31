import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from multiprocessing import Pool
from data.utils import cut_phonemes_ids
from config import NEURAL_DATA_KEY, TRANSCRIPTION_KEY



def decode_transcription_fixed(ids):
    try:
        ids_array = np.array(ids)
        zero_indices = np.where(ids_array == 0)[0]
        if len(zero_indices) > 0:
            first_zero = zero_indices[0]
            char_list = [chr(c) for c in ids[:first_zero]]
        else:
            char_list = [chr(c) for c in ids]
    except (ValueError, TypeError): # If no zero is found or other error
        char_list = [chr(c) for c in ids]
        
    return "".join(char_list)


def get_data(sessions, split='train') -> pd.DataFrame:
    train_data = []
    trial_key_to_day = {}
    cur_day = 0

    for session_path in tqdm(sessions):
        file_path = os.path.join(session_path, f'data_{split}.hdf5')
        
        if os.path.exists(file_path):
            if split not in file_path:
                continue
            with h5py.File(file_path, 'r') as f:
                for trial_key in f.keys():
                    trial_group = f[trial_key]
                    
                    if isinstance(trial_group, h5py.Group) and NEURAL_DATA_KEY in trial_group:
                        neural_data = trial_group[NEURAL_DATA_KEY][()]
                        phonemes_ids = trial_group['seq_class_ids'][()]
                        phonemes_ids = cut_phonemes_ids(phonemes_ids)
                        
                        transcription_text = None
                        if TRANSCRIPTION_KEY in trial_group:
                            transcription_ids = trial_group[TRANSCRIPTION_KEY][()]
                            transcription_text = decode_transcription_fixed(transcription_ids)

                        if trial_key not in trial_key_to_day:
                            trial_key_to_day[trial_key] = cur_day
                            cur_day += 1
                        
                        train_data.append({
                            'session': session_path,
                            'trial_id': trial_key,
                            'neural_data': neural_data,
                            'transcription': transcription_text,
                            'day_idx': trial_key_to_day[trial_key],
                            'phonemes_ids': phonemes_ids,
                            'num_words': len(transcription_text.split()) if transcription_text else 0,
                        })

    return pd.DataFrame(train_data)


def get_sessions(directory):
    return sorted([os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])


def split_sessions(sessions, workers):
    parts = np.array_split(sessions, workers)
    return [list(p) for p in parts]
    

def read_dataset(directory, split, workers=1):
    sessions = get_sessions(directory)
    worker_to_sessions = split_sessions(sessions, workers)
    args = [(chunk, split) for chunk in worker_to_sessions if len(chunk) > 0]
    with Pool(processes=min(workers, len(args))) as pool:
        dfs = pool.starmap(get_data, args)
        
    if len(dfs) == 0:
        return pd.DataFrame([])
    return pd.concat(dfs, ignore_index=True)

