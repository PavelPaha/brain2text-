import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


BASE_DIR = 'brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final'
NEURAL_DATA_KEY = 'input_features'
TRANSCRIPTION_KEY = 'transcription'


def decode_transcription_fixed(ids):
    try:
        # Convert to numpy array if it isn't already, then find first zero
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

def load_metadata_from_hdf5(file_path):
    metadata = []
    try:
        with h5py.File(file_path, 'r') as f:
            # The top-level keys ARE the trials.
            for trial_key in f.keys():
                trial_group = f[trial_key]
                
                # Check if the group contains the correct dataset names
                if isinstance(trial_group, h5py.Group) and NEURAL_DATA_KEY in trial_group and TRANSCRIPTION_KEY in trial_group:
                    
                    num_time_bins = trial_group[NEURAL_DATA_KEY].shape[0]
                    
                    # The transcription is an array of integers, not a string.
                    # We will load it as a list of numbers for now.
                    transcription_ids = list(trial_group[TRANSCRIPTION_KEY][()])
                    
                    metadata.append({
                        'trial_id': trial_key,
                        'num_time_bins': num_time_bins,
                        'transcription_ids': transcription_ids,
                        # We can't get num_words directly yet, so we'll estimate from the length of the ID list.
                        # This might not be perfect but is a good start.
                        'num_words_estimate': len(transcription_ids) 
                    })
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
    return metadata


def get_data(split='train') -> pd.DataFrame:
    session_dirs = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

    train_data = []

    for session in tqdm(session_dirs[:1]):
        session_path = os.path.join(BASE_DIR, session)
        file_path = os.path.join(session_path, 'data_train.hdf5')
        
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