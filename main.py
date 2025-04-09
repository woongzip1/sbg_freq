import torch
import yaml
import gc
import warnings
import random
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler

import wandb
from tqdm import tqdm
from box import Box

from utils import *

## models
from models.prepare_models import MODEL_MAP, prepare_discriminator, prepare_generator
## dataset
from dataset import Dataset, make_dataset
from trainer import Trainer

### values
DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
TIME = datetime.now()
print(f"DEVICE: {DEVICE}")
print(TIME.strftime("%Y-%m-%d %H:%M:%S"))
print(MODEL_MAP)

def parse_args():
    parser = argparse.ArgumentParser(description="mbseanet Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--wandb', type=lambda x:x.lower()=='true', default='False', help="wandb logging (True/False)")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as file:
        return Box(yaml.safe_load(file))

def prepare_dataloader(config):
    # config = load_config(config_path)
    train_dataset = make_dataset(config, 'train')
    val_dataset = make_dataset(config, 'val')

    # Optional ratio split
    if config.dataset.ratio < 1:
        train_size = int(config.dataset.ratio * len(train_dataset))
        _, train_dataset = random_split(train_dataset, [len(train_dataset) - train_size, train_size])

    train_loader = DataLoader(train_dataset, shuffle=True, **config.dataloader)
    
    val_loader_args = config.dataloader
    val_loader_args.batch_size = 1
    val_loader = DataLoader(val_dataset, shuffle=False, **val_loader_args)

    if hasattr(config.dataset, 'val_speech'):
        val_speech_dataset = make_dataset(config, 'val_speech')
        val_speech_loader = DataLoader(val_speech_dataset, shuffle=False, **val_loader_args)
    else:
        val_speech_loader = None

    return train_loader, val_loader, val_speech_loader

def main(if_log_step):
    args = parse_args()
    if_log_to_wandb = args.wandb
    config = load_config(args.config)
    
    torch.manual_seed(42)
    random.seed(42)

    print_config(config)    
    if if_log_to_wandb: # if log
        wandb.init(project=config['project_name'], entity='woongzip1', config=config, name=config['run_name'], notes=config['run_name'])
    
    # Prepare dataloader
    train_loader, val_loader, _ = prepare_dataloader(config)

    # Model selection
    generator = prepare_generator(config, MODEL_MAP)
    discriminator = prepare_discriminator(config)

    # Optimizers
    if config.generator.fine_tune:
        print("------------Fine Tuning!------------")
        pass
    #     non_fe_params = [p for p in generator.parameters() if p not in set(generator.feature_encoder.parameters())]
    #     optim_G = torch.optim.Adam(
    #         [
    #             {'params': generator.feature_encoder.parameters(), 'lr': config['optim']['learning_rate_ft']}, 
    #             {'params': non_fe_params, 'lr': config['optim']['learning_rate']}  
    #         ],
    #         betas=(config['optim']['B1'], config['optim']['B2'])
    #     )
    #     optim_D = torch.optim.Adam(discriminator.parameters(), lr=config['optim']['learning_rate'], betas=(config['optim']['B1'], config['optim']['B2']))
    else: # scratch
        optim_G = torch.optim.Adam(generator.parameters(), lr=config['optim']['learning_rate'], betas=(config['optim']['B1'], config['optim']['B2']))
        optim_D = torch.optim.Adam(discriminator.parameters(), lr=config['optim']['learning_rate'], betas=(config['optim']['B1'], config['optim']['B2']))
        
    # Schedulers
    if config['use_tri_stage']:
        pass
        # from scheduler import TriStageLRScheduler
        # print("*** TriStageLRScheduler ***")
        # scheduler_G = TriStageLRScheduler(optimizer=optim_G, **config['tri_scheduler'])
        # scheduler_D = TriStageLRScheduler(optimizer=optim_D, **config['tri_scheduler'])
    else:
        print("🚀 *** Exp LRScheduler ***")
        scheduler_G = lr_scheduler.ExponentialLR(optim_G, gamma=config['optim']['scheduler_gamma'])
        scheduler_D = lr_scheduler.ExponentialLR(optim_D, gamma=config['optim']['scheduler_gamma'])

    # Trainer initialization
    trainer = Trainer(generator, discriminator, train_loader, val_loader, optim_G, optim_D, config, DEVICE, 
                      scheduler_G=scheduler_G, scheduler_D=scheduler_D, if_log_step=if_log_step, if_log_to_wandb=if_log_to_wandb)
    
    if config['train']['ckpt']:
        trainer.load_checkpoints(config['train']['ckpt_path'])
    
    torch.manual_seed(42)
    random.seed(42)
    
    # Train
    # warnings.filterwarnings("ignore", category=UserWarning, message="At least one mel filterbank has")
    # warnings.filterwarnings("ignore", category=UserWarning, message="Plan failed with a cudnnException")
    trainer.train(num_epochs=config['train']['max_epochs'])

if __name__ == "__main__":
    main(if_log_step=True)
