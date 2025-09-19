import h5py
import numpy as np
import pandas as pd
import os


BASE_DIR = 'brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final'
NEURAL_DATA_KEY = 'input_features'
TRANSCRIPTION_KEY = 'transcription'


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

def get_data(split='train') -> pd.DataFrame:
    session_dirs = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

    train_data = []

    for session in session_dirs[1:2]:
        session_path = os.path.join(BASE_DIR, session)
        file_path = os.path.join(session_path, f'data_{split}.hdf5')
        
        if os.path.exists(file_path):
            if split not in file_path:
                continue
            with h5py.File(file_path, 'r') as f:
                for trial_key in f.keys():
                    trial_group = f[trial_key]
                    
                    if isinstance(trial_group, h5py.Group) and NEURAL_DATA_KEY in trial_group:
                        neural_data = trial_group[NEURAL_DATA_KEY][()]
                        phonems = trial_group['seq_class_ids'][()]
                        
                        transcription_text = None
                        if TRANSCRIPTION_KEY in trial_group:
                            transcription_ids = trial_group[TRANSCRIPTION_KEY][()]
                            transcription_text = decode_transcription_fixed(transcription_ids)
                        
                        train_data.append({
                            'session': session,
                            'trial_id': trial_key,
                            'neural_data': neural_data,
                            'transcription': transcription_text,
                            'num_time_bins': neural_data.shape[0],
                            'num_features': neural_data.shape[1],
                            'phonems_ids': phonems,
                            'num_words': len(transcription_text.split()) if transcription_text else 0,
                        })

    return pd.DataFrame(train_data)

if __name__ == "__main__":
    print(get_data('train').head())