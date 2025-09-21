from data.dataloader import create_dataloader
from data.dataset import create_dataset
from data.read_dataset import read_dataset
from data.utils import PHONEMES, DIPHONES
from config import BASE_DIR, NUM_THREADS_DATA_READING, CHECKPOINT_PATH

from model.model import GRU
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import mlflow


def prepare_batch(batch):
    return batch


def calc_loss(criterion, raw_logits, labels):
    logits_permuted = raw_logits.permute(1, 0, 2)
    log_probs = F.log_softmax(logits_permuted, dim=2)
    loss = criterion(log_probs, labels, 
                    input_lengths=torch.full((raw_logits.size(0),), raw_logits.size(1), dtype=torch.long),
                    target_lengths=torch.full((labels.size(0),), labels.size(1), dtype=torch.long))
    
    return loss


def train_step(model, data_loader, optimizer, criterion, global_step):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        # input = batch['neural_data']
        labels = batch['phonemes_ids']

        batch = prepare_batch(batch)
        logits, _ = model(batch)
        loss = calc_loss(criterion, logits, labels)
        
        mlflow.log_metric('train_loss', loss.item(), step=global_step)
        global_step += 1

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)


def evaluate_step(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            batch = prepare_batch(batch)
            labels = batch['phonemes_ids']

            logits, _ = model(batch)
            loss = calc_loss(criterion, logits, labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def run_training(cfg, args):
    device = f'cuda:{args.device_id}'
    labels_type = cfg.data.labels_type
    output_size = len(PHONEMES) if labels_type == 'phonemes' else len(DIPHONES)

    train_data = read_dataset(BASE_DIR, split='train', workers=NUM_THREADS_DATA_READING)
    val_data = read_dataset(BASE_DIR, split='val', workers=NUM_THREADS_DATA_READING)
    train_dataset = create_dataset(train_data, output_size=output_size)
    val_dataset = create_dataset(val_data, output_size=output_size)

    train_loader = create_dataloader(train_dataset, 
                                     batch_size=cfg.train.batch_size, 
                                     shuffle=cfg.train.shuffle, 
                                     num_workers=cfg.train.num_workers
                                    )
    
    val_loader = create_dataloader(val_dataset, 
                                   batch_size=cfg.val.batch_size, 
                                   shuffle=cfg.val.shuffle, 
                                   num_workers=cfg.val.num_workers
                                   )

    days_count = train_data['trial_id'].nunique()
    print(days_count)
    model = GRU(input_size=cfg.model.input_size, 
                hidden_size=cfg.model.hidden_size, 
                output_size=output_size, 
                device=device, 
                days_count=days_count)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    model.train()
    model.to(device)
    print(f"Training on device: {device}")

    global_step = 0
    for epoch in range(cfg.train.epochs):
        train_loss = train_step(model, train_loader, optimizer, criterion, global_step)
        global_step += len(train_loader)

        val_loss = evaluate_step(model, val_loader, criterion)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        mlflow.log_metrics({
            "train_epoch_loss": train_loss,
            "val_epoch_loss": val_loss
        }, step=epoch+1)

        torch.save(model, f'{CHECKPOINT_PATH}/{cfg.model_name}_epoch_{epoch}.pth')

    mlflow.pytorch.log_model(model, artifact_path="model")
