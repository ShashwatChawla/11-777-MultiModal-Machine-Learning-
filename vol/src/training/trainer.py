import time
import os
import sys
sys.path.append('/ocean/projects/cis220039p/pkachana/projects/11-777-MultiModal-Machine-Learning-/vol/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb

from configs.base_config import BaseConfig
from dataloader.dataloader import get_dataloader
from model.vol_net import VOLNet


class Trainer():

    def __init__(self, 
        config, 
        model, 
        train_loader,
        val_loader,
        loss_fn,
        data_process_fn = None,
        optimizer = None,
        scheduler = None,
        logger = None
    ):

        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.data_process_fn = data_process_fn

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optimizer = optimizer

        # if scheduler is None:
        #     scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        # self.scheduler = scheduler

        # if logger is None:
        #     logger = wandb.init(
        #         project=config.project_name,
        #         # config=config.__dict__
        #     )
        # self.logger = logger

        self.num_steps = self.config.num_steps
        self.val_freq = self.config.val_freq

        self.log_freq = self.config.log_freq
        self.log_steps = self.config.log_steps
        
        current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
        self.ckpt_save_dir = f"{self.config.ckpt_save_dir}/{current_time}"
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        self.ckpt_save_dir = self.config.ckpt_save_dir

    
    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(data)
        loss = self.loss_fn(
            output['rotation'], 
            data['rotations'], 
            output['translation'], 
            data['translations'],
            alpha=self.config.loss_alpha
        )

        breakpoint()
        total_loss = loss['total_loss'].float()
        total_loss.backward()
        self.optimizer.step()

        return loss
    

    def val_step(self, data):
        self.model.eval()

        output = self.model(data)
        loss = self.loss_fn(
            output['rotation'], 
            data['rotations'], 
            output['translation'], 
            data['translations'],
            alpha=self.config.loss_alpha
        )

        return loss
    

    def train(self):
        for step in range(self.num_steps):
            
            batch = self.train_loader.load_sample()
            processed_batch = self.data_process_fn(batch)

            loss = self.train_step(processed_batch)

            # self.scheduler.step()

            # if step % self.log_freq == 0:
                # self.logger.log({
                #     "train_loss": loss,
                #     "learning_rate": self.optimizer.param_groups[0]['lr']
                # })
            
            if step % self.val_freq == 0:
                val_loss = 0
                for val_step in range(self.val_steps):
                    val_batch = self.val_loader.load_sample()
                    val_processed_batch = self.model.preprocess(val_batch)

                    val_loss += self.val_step(val_processed_batch)

                val_loss /= self.val_steps

                # self.logger.log({
                #     "val_loss": val_loss
                # })

                torch.save(self.model.state_dict(), f"{self.ckpt_save_dir}/model_step_{step}.pt")


if __name__ == '__main__':
    pass
