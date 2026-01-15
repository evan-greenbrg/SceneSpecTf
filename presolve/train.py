import math
import os
from datetime import datetime

import click
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import schedulefree
from spectral import envi
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error
)

from spectf.utils import envi_header
from spectf.utils import get_device

from presolve.model import ConvEncoder, Discriminator
from spectf.utils import seed as useed
from presolve.dataset import ImageDataset


os.environ["WANDB__SERVICE_WAIT"] = "300"
torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, lambda_adv=0.5, lambda_lat=0.2, gpu=0):

        self.device = get_device(gpu)

        # Loss weightings for the discriminator and the latent match
        self.lambda_adv = lambda_adv
        self.lambda_lat = lambda_lat

        # Loss for the discrimininator: Look at why we use this
        self.bce = nn.BCEWithLogitsLoss()

        # Loss for the latent match-up: Best option?
        self.mse = nn.MSELoss()

        # For latent prior sampling
        self.latent_scale = 6
        self.gamma_dist = torch.distributions.gamma.Gamma(
            concentration=4,
            rate=0.5
        )

    def step(self, x, latent_target, encoder, opt_encoder, disc, opt_disc):
        # prior N(0,I): How is this setting the prior distirbution?
        # Set prior of head 1: .05 -> 6
        # Set prior of head 2: 0 -> 1
        # Hard code the latent set up
        # Uniform dist for the h2o vals
        prior_head_0, _ = torch.sort(torch.rand(
            x.shape[0],
            1, 40,
            device=self.device
        ) * self.latent_scale)

        torch.sort(prior_head_0)

        # Curious if the distribution of the values matters.
        prior_head_1 = self.gamma_dist.sample((x.shape[0], 1, 40)).to(self.device)
        # prior_head_1 = torch.abs(torch.randn(
        #     x.shape[0], 
        #     1, 40,
        #     device=self.device
        # ))
        latent_prior = torch.cat(
            [prior_head_0, prior_head_1], 
            dim=1
        )

        latent_pred = encoder(x)

        # Update Discriminator
        logits_prior = disc(latent_prior)
        # logits_prior = disc(latent_target)
        logits_pred = disc(latent_pred)  

        loss_disc = (
            self.bce(logits_prior, torch.ones_like(logits_prior)) 
            + self.bce(logits_pred, torch.zeros_like(logits_pred))
        ) / 2

        opt_disc.zero_grad()
        loss_disc.backward()
        # torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 
        opt_disc.step()

        # Update Encoder
        latent_pred = encoder(x)
        logits_pred = disc(latent_pred)
        loss_adv_enc = self.bce(
            logits_pred,
            torch.ones_like(logits_pred)
        )
        loss_lat = self.mse(latent_pred, latent_target)
        loss_enc = (
            (self.lambda_adv * loss_adv_enc)
            + (self.lambda_lat * loss_lat)
        )

        opt_encoder.zero_grad()
        loss_enc.backward()
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        opt_encoder.step()

        return (
            loss_disc.item(),
            loss_lat.item(),
            loss_adv_enc.item()
        ), encoder, opt_encoder, disc, opt_disc


def evaluation(_dataloader, encoder, device):
    train_true_x = []
    train_pred_x = []
    train_true_y = []
    train_pred_y = []
    for idx, batch_ in enumerate(_dataloader):
        rdn = batch_['images'].to(device)
        latent = batch_['latent'].to(device)

        with torch.no_grad():
            pred = encoder(rdn)

        pred_cpu = pred.detach().cpu().numpy()
        latent_cpu = latent.detach().cpu().numpy()

        train_pred_x += list(pred_cpu[:, 0, :].flatten())
        train_pred_y += list(pred_cpu[:, 1, :].flatten())
        train_true_x += list(latent_cpu[:, 0, :].flatten())
        train_true_y += list(latent_cpu[:, 1, :].flatten())

    train_pred_x = np.array(train_pred_x)
    train_pred_y = np.array(train_pred_y)
    train_true_x = np.array(train_true_x)
    train_true_y = np.array(train_true_y)

    train_r2_x = r2_score(train_true_x, train_pred_x)
    train_r2_y = r2_score(train_true_y, train_pred_y)
    train_mape_x = mean_absolute_percentage_error(train_true_x, train_pred_x)
    train_mape_y = mean_absolute_percentage_error(train_true_y, train_pred_y)

    return train_r2_x, train_r2_y, train_mape_x, train_mape_y

@click.command()
@click.argument('train_rdn_path')
@click.argument('train_atm_path')
@click.argument('test_rdn_path')
@click.argument('test_atm_path')
@click.argument('outdir')
@click.option('--nchunks', default=32)
@click.option('--chunksize', default=512)
@click.option('--arch_nbands', default=285)
@click.option('--arch_heads', default=2)
@click.option('--arch_dim_output', default=40)
@click.option('--arch_num_blocks', default=2)
@click.option('--lr', default=1e-6)
@click.option('--gpu', is_flag=True, default=False)
@click.option('--batch', default=32)
@click.option('--epochs', default=50)
@click.option('--seed', default=42)
@click.option('--wandb_name', default='')
@click.option('--wandb_entity', default='')
@click.option('--wandb_project', default='')
@click.option('--save_every_epoch', is_flag=True, default=False)
def train(
    train_rdn_path: str,
    train_atm_path: str,
    test_rdn_path: str,
    test_atm_path: str,
    outdir: str,
    nchunks: int = 32,
    chunksize: int = 512,
    arch_nbands: int = 285,
    arch_heads: int = 2,
    arch_rows_input: int = 512,
    arch_cols_input: int = 512,
    arch_dim_output: int = 40,
    arch_num_blocks: int = 2,
    lr: float = 1e-6,
    gpu: int = 0,
    batch: int = 32,
    epochs: int = 50,
    seed: int = 42,
    wandb_name: str = '',
    wandb_entity: str = '',
    wandb_project: str = '',
    save_every_epoch: bool = False,
):
    useed(seed)
    device = get_device(0)

    with open(train_rdn_path, 'r') as f:
        train_rdn_paths = [line.strip() for line in f.readlines()]

    with open(train_atm_path, 'r') as f:
        train_atm_paths = [line.strip() for line in f.readlines()]

    with open(test_rdn_path, 'r') as f:
        test_rdn_paths = [line.strip() for line in f.readlines()]

    with open(test_atm_path, 'r') as f:
        test_atm_paths = [line.strip() for line in f.readlines()]

    train_dataset = ImageDataset(
        train_rdn_paths,
        train_atm_paths,
        nchunks=nchunks,
        chunksize=chunksize,
        nbins=arch_dim_output
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=False
    )

    test_dataset = ImageDataset(
        test_rdn_paths,
        test_atm_paths,
        nchunks=nchunks,
        chunksize=chunksize,
        nbins=arch_dim_output
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch,
        shuffle=False
    )

    # Define wandb
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_%f_{wandb_name}")
    try:
        run = wandb.init(
            project = wandb_project,
            entity = wandb_entity,
            name = timestamp,
            dir = './',
            config = {
                'dataset_path': '',
                'lr': lr,
                'epochs': epochs,
                'batch': batch,
                'arch_nbands': arch_nbands,
                'chunksize': chunksize,
                'arch_dim_output': arch_dim_output,
                'arch_heads': arch_heads,
                'arch_num_blocks': arch_num_blocks,
            },
            settings=wandb.Settings(_service_wait=300)
        )
    except Exception as e:
        print("WandB error!")
        print(e)
        sys.exit(1)

    encoder = ConvEncoder(
        nbands=285,
        rows_input=chunksize,
        cols_input=chunksize,
        dim_output=arch_dim_output,
        nblocks=arch_num_blocks,
        heads=arch_heads
    ).to(device)
    disc = Discriminator(arch_dim_output, heads=arch_heads).to(device)
    trainer = Trainer(lambda_adv=0.2, lambda_lat=0.8)

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=lr)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)

    for epoch in range(epochs):
        encoder.train()
        # opt_enc.train()
        disc.train()
        # opt_disc.train()

        train_epoch_loss_disc = 0
        train_epoch_loss_latent = 0
        train_epoch_loss_encoding = 0
        for ite, batch_ in enumerate(train_dataloader):
            print(ite)
            rdn = batch_['images'].to(device)
            latent = batch_['latent'].to(device)

            loss, encoder, opt_enc, disc, opt_disc = trainer.step(
                rdn, latent, encoder, opt_enc, disc, opt_disc
            )
            loss_disc, loss_latent, loss_encoding = loss

            run.log({"loss_disc": loss_disc})
            run.log({"loss_latent": loss_latent})
            run.log({"loss_encoding": loss_encoding})

            train_epoch_loss_disc += loss_disc
            train_epoch_loss_latent += loss_latent
            train_epoch_loss_encoding += loss_encoding

            train_epoch_loss_disc /= len(train_dataloader)
            train_epoch_loss_latent /= len(train_dataloader)
            train_epoch_loss_encoding /= len(train_dataloader)

        ## MODEL EVALUATION
        encoder.eval()
        # opt_encoder.eval()
        disc.eval()
        # opt_disc.eval()

        # training eval
        (
            train_r2_x,
            train_r2_y,
            train_mape_x,
            train_mape_y
        ) = evaluation(train_dataloader, encoder, device)

        # testing eval
        (
            test_r2_x,
            test_r2_y,
            test_mape_x,
            test_mape_y
        ) = evaluation(test_dataloader, encoder, device)

        run.log({
            "train/loss_disc": train_epoch_loss_disc,
            "train/loss_latent": train_epoch_loss_latent,
            "train/loss_encoding": train_epoch_loss_encoding,
            "train/r2_val": train_r2_x,
            "train/r2_dens": train_r2_y,
            "train/mape_val": train_mape_x,
            "train/mape_dens": train_mape_y,
            "test/r2_val": test_r2_x,
            "test/r2_dens": test_r2_y,
            "test/mape_val": test_mape_x,
            "test/mape_dens": test_mape_y,
            "epoch": epoch,
        })

        if save_every_epoch:
            os.makedirs(os.path.join(outdir, 'epochs', exist_ok=True)
            torch.save(
                encoder.state_dict(), 
                os.path.join(
                    outdir, 
                    'epochs',
                    f"scene_presolve_encoder_{timestamp}_{epoch}.pt"
                )
            )

        torch.save(
            encoder.state_dict(), 
            os.path.join(outdir, f"scene_presolve_encoder_{timestamp}.pt")
        )
        torch.save(
            disc.state_dict(), 
            os.path.join(outdir, f"scene_presolve_discriminator_{timestamp}.pt")
        )
    run.finish()


if __name__ == '__main__':
    train()
