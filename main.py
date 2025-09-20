import mlflow
import argparse
from omegaconf import OmegaConf
from config import MLFLOW_URI
from training import run_training


def main(cfg, args):
    mlflow.set_tracking_uri(uri=MLFLOW_URI)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run():
        run_training(cfg, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-id', type=int, default=3)
    parser.add_argument('--config', type=str, default='base.yaml')
    args = parser.parse_args()

    cfg = OmegaConf.load(f'train_configs/{args.config}')
    main(cfg, args)