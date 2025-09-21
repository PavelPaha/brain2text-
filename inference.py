import torch.nn.functional as F
from data.utils import PHONEMES, DIPHONES, get_phonemes
from config import BASE_DIR, NUM_THREADS_DATA_READING
from data.read_dataset import read_dataset
from data.dataset import create_dataset
from data.dataloader import create_dataloader
from omegaconf import OmegaConf
import argparse
import torch

def generate(model, query):
    logits, _ = model(query) # [b, seq_len, vocab_size]
    probs = F.softmax(logits, dim=-1)
    ids_batch = probs.argmax(dim=-1, keepdim=False) # [b, seq_len]
    return [get_phonemes(ids) for ids in ids_batch]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/base.pt')
    parser.add_argument('--config', type=str, default='base.yaml')
    args = parser.parse_args()

    cfg = OmegaConf.load(f'train_configs/{args.config}')

    val_data = read_dataset(BASE_DIR, split='val', workers=NUM_THREADS_DATA_READING)
    output_size = len(PHONEMES) if cfg.data.labels_type == 'phonemes' else len(DIPHONES)
    val_dataset = create_dataset(val_data, output_size=output_size)

    val_loader = create_dataloader(val_dataset, 
                                   batch_size=cfg.val.batch_size, 
                                   shuffle=cfg.val.shuffle, 
                                   num_workers=cfg.val.num_workers
                                   )

    model = torch.load(args.checkpoint, weights_only=False)

    for batch in val_loader:
        answers = generate(model, batch)
        print(batch.keys())
        # transcriptions = batch['transcription']
        for answer in answers:
            print(answer)
            break
        
    


    
