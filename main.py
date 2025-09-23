import mlflow
import argparse
from omegaconf import OmegaConf
from config import MLFLOW_URI
from training.training import run


def main(cfg, args):
    mlflow.set_tracking_uri(uri=MLFLOW_URI)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=cfg.run_name):
        run(cfg, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-id', type=int)
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(f'train_configs/{args.config}')
    main(cfg, args)