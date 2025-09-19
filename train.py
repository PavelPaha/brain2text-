from data.dataloader import create_dataloader
from data.dataset import create_dataset
from data.read_dataset import get_data
from data.utils import PHONEMES, DIPHONES_TO_IDS

from model.model import GRU
import torch
from tqdm import tqdm
import argparse
from omegaconf import DictConfig, OmegaConf


def prepare_batch(batch, device):
    neural_data = batch['neural_data'].to(device)
    labels = batch['phonems_ids'].to(device)
    return neural_data, labels 


def train_step(model, data_loader, criterion, device):
    model.train()
    total_loss = 0
    for _, batch in enumerate(data_loader):
        input, labels = prepare_batch(batch, device)
        logits, output = model(input)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
    return total_loss / len(data_loader)


def evaluate_step(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input, labels = prepare_batch(batch, device)
            logits, _ = model(input)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def main(cfg: DictConfig, args):
    labels_type = cfg.data.labels_type
    output_size = len(PHONEMES) if labels_type == 'phonemes' else len(DIPHONES_TO_IDS)

    train_data = get_data(split='train')
    val_data = get_data(split='val')
    train_dataset = create_dataset(train_data, output_size=output_size)
    val_dataset = create_dataset(val_data, output_size=output_size)

    train_loader = create_dataloader(train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle, num_workers=cfg.train.num_workers)
    val_loader = create_dataloader(val_dataset, batch_size=cfg.val.batch_size, shuffle=cfg.val.shuffle, num_workers=cfg.val.num_workers)

    

    model = GRU(input_size=cfg.model.input_size, hidden_size=cfg.model.hidden_size, output_size=output_size)
    criterion = torch.nn.CTCLoss()
    
    print(OmegaConf.to_yaml(cfg))

    model.train()
    # device = f'cuda:{args.device_id}'
    device = 'cpu'
    model.to(device)
    for epoch in tqdm(range(cfg.train.epochs)):
        train_loss = train_step(model, train_loader, criterion, device)
        val_loss = evaluate_step(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-id', type=int)
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(f'configs/{args.config}')
    print(cfg)
    main(cfg, args)

