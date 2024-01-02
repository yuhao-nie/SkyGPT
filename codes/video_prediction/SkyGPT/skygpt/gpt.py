import os
import itertools
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import math
import time

from .resnet import resnet34
from .attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed
from .utils import shift_dim
from .constrain_moments import K2M

class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        assert math.sqrt(F_hidden_dim) ** 2 == F_hidden_dim
        self.F.add_module('bn1',nn.GroupNorm(int(math.sqrt(F_hidden_dim)) ,F_hidden_dim))        
        # self.F.add_module('bn1',nn.GroupNorm(24 ,F_hidden_dim))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                              out_channels= self.input_dim,
                              kernel_size=(3,3),
                              padding=(1,1), bias=self.bias)

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)        # prediction
        next_hidden = hidden_tilde + K * (x-hidden_tilde)   # correction , Haddamard product     
        return next_hidden

# Incorporate PhyCell into VideoGPT architecture
class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, heads, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []  
        self.device = device
        # self.heads = heads
        
        cell_list = []
        # for head in range(self.heads):
        for i in range(self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                        F_hidden_dim=self.F_hidden_dims,
                                        kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]
        input_ = input_.permute(0, 3, 1, 2)
        if first_timestep:
            self.H = []
            # for _ in range(self.heads):
            for _ in range(self.n_layers):
                self.H.append( torch.zeros_like(input_))
        
        for j, cell in enumerate(self.cell_list):
            # if j % self.heads == 0: # bottom layer
            if j == 0: # bottom layer
                # transformed_input = input_[:, j // self.heads].permute(0, 3, 1, 2)
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])

        return torch.stack(self.H), torch.stack(self.H)

    def setHidden(self, H):
        self.H = H

class SkyGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load VQ-VAE and set all parameters to no grad
        from .vqvae import VQVAE
        # Path to trained VQ-VAE model
        args.vqvae = '/scratch/groups/abrandt/solar_forecasting/GAN_project/models/VideoGPT/trained_models/VQVAE_full_2min/lightning_logs/version_0/checkpoints/epoch=47-step=160031.ckpt'    
        self.vqvae =  VQVAE.load_from_checkpoint(args.vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # ResNet34 for frame conditioning
        self.use_frame_cond = args.n_cond_frames > 0
        if self.use_frame_cond:
            frame_cond_shape = (args.n_cond_frames,
                                args.resolution // 4,
                                args.resolution // 4,
                                240)
            self.resnet = resnet34(1, (1, 4, 4), resnet_dim=240)
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=frame_cond_shape[:-1], embd_dim=frame_cond_shape[-1]
            )
        else:
            frame_cond_shape = None

        # SkyGPT transformer
        self.shape = self.vqvae.latent_shape

        self.fc_in = nn.Linear(self.vqvae.embedding_dim, args.hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_stack = AttentionStack(
            self.shape, args.hidden_dim, args.heads, args.layers, args.dropout,
            args.attn_type, args.attn_dropout, args.class_cond_dim, frame_cond_shape
        )

        self.norm = LayerNorm(args.hidden_dim, args.class_cond_dim)

        self.fc_out = nn.Linear(args.hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, args.hidden_dim))

        # caches for faster decoding (if necessary)
        self.frame_cond_cache = None

        self.save_hyperparameters()

        self.kernel_size = 7
        F_hidden_dims = self.kernel_size ** 2

        self.phycell = PhyCell(
            input_shape=(32,32), input_dim=args.hidden_dim, F_hidden_dims=F_hidden_dims, 
            n_layers=1, kernel_size=(self.kernel_size,self.kernel_size), heads=args.heads, device='cuda'
        )

        self.constraint_criterion = nn.MSELoss()
        self.constraints = torch.zeros((F_hidden_dims,self.kernel_size,self.kernel_size))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.constraints[i * self.kernel_size + j, i, j] = 1

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, n, batch=None):
        device = self.fc_in.weight.device

        cond = dict()
        if self.use_frame_cond or self.args.class_cond:
            assert batch is not None
            video = batch['video']

            if self.args.class_cond:
                label = batch['label']
                cond['class_cond'] = F.one_hot(label, self.args.class_cond_dim).type_as(video)
            if self.use_frame_cond:
                cond['frame_cond'] = video[:, :, :self.args.n_cond_frames]

        samples = torch.zeros((n,) + self.shape).long().to(device)
        
        samples_phycell = torch.zeros((n,) + self.shape).long().to(device)
        samples_transformer = torch.zeros((n,) + self.shape).long().to(device)
        
        idxs = list(itertools.product(*[range(s) for s in self.shape]))
        sample_time = int(time.time())

        with torch.no_grad():
            prev_idx = None
            max_idx = 0
            _, base_embeddings = self.vqvae.encode(cond['frame_cond'], include_embeddings=True)
            base_embeddings = shift_dim(base_embeddings, 1, -1)
            for i, idx in enumerate(tqdm(idxs)):
                batch_idx_slice = (slice(None, None), *[slice(i, i + 1) for i in idx])
                batch_idx = (slice(None, None), *idx)
                if idx[1] == 0 and idx[2] == 0:
                    # if idx[0] <= len(cond['frame_cond']):
                    if idx[0] == 0:
                        _, base_embeddings = self.vqvae.encode(cond['frame_cond'], include_embeddings=True)
                        base_embeddings = shift_dim(base_embeddings, 1, -1)
                    else:                        
                        base_embeddings = torch.cat([
                            base_embeddings, 
                            self.vqvae.codebook.dictionary_lookup(samples)[:, idx[0] - 1][:, None]
                        ], axis=1)
                    reuse_phycell = False
 
                embeddings = self.vqvae.codebook.dictionary_lookup(samples)
                
                if prev_idx is None:
                    # set arbitrary input values for the first token
                    # does not matter what value since it will be shifted anyways
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]
                
                # newly added
                logits,embedding_phycell,embedding_transformer = self(embeddings_slice, samples_slice, cond,
                              decode_step=i, decode_idx=idx, x_full=base_embeddings, include_loss=False, reuse_phycell=reuse_phycell)
                
                # squeeze all possible dim except batch dimension
                logits = logits.squeeze().unsqueeze(0) if logits.shape[0] == 1 else logits.squeeze()
                probs = F.softmax(logits, dim=-1)
                samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)
                
                # newly added
                samples_phycell[batch_idx] = embedding_phycell
                samples_transformer[batch_idx] = embedding_transformer
                
                prev_idx = batch_idx_slice
                reuse_phycell = True

                if idx[1] == idx[2] and idx[1] == self.vqvae.latent_shape[-1] - 1 and idx[0] == 15:
                    max_idx = max(max_idx, idx[1])
                    torchvision.utils.save_image(torch.flip(
                        torch.clamp(batch['video'][0], -0.5, 0.5) + 0.5, [0]
                    ).permute(1, 0, 2, 3), f'condition.png')

                    vis_samples = self.vqvae.decode(samples)
                    vis_samples = torch.clamp(vis_samples, -0.5, 0.5) + 0.5
                    for vis_idx, vis_sample in enumerate(vis_samples):
                        torchvision.utils.save_image(torch.flip(vis_sample, [0]).permute(1, 0, 2, 3), f'samples/forecast_{sample_time}_{vis_idx}_{idx[0]}.png')

        return self.vqvae.decode(samples),self.vqvae.decode(samples_phycell),self.vqvae.decode(samples_transformer) # BCTHW in [0, 1]

    def forward(self, x, targets, cond, decode_step=None, decode_idx=None, include_loss=True, x_full=None, reuse_phycell=False):
        if self.use_frame_cond:
            if decode_step is None:
                cond['frame_cond'] = self.cond_pos_embd(self.resnet(cond['frame_cond']))
            elif decode_step == 0:
                self.frame_cond_cache = self.cond_pos_embd(self.resnet(cond['frame_cond']))
                cond['frame_cond'] = self.frame_cond_cache
            else:
                cond['frame_cond'] = self.frame_cond_cache

        h = self.fc_in(x)
        h_1 = self.attn_stack(h, cond, decode_step, decode_idx)
        if decode_idx is None:
            h_1 = h_1[:, 1:]

        # print(h_1.shape, h.shape)
        h_full = h[:, :-1] if x_full is None else self.fc_in(x_full)

        # Iterate over the timesteps for phycell
        if reuse_phycell:
            h_0 = self.h_0_cached.to(h_full.device)
        else:
            h_0 = [self.phycell(h_full[:,ei,:,:,:], (ei==0))[0][-1] for ei in range(h_full.shape[1])]
            h_0 = torch.stack(h_0).permute(1, 0, 3, 4, 2)
            self.h_0_cached = h_0.detach()
        if decode_idx is not None:
            h_0 = h_0[:, decode_idx[0], decode_idx[1], decode_idx[2], :]
            h_0 = h_0.reshape_as(h_1)
        h = h_0 + h_1

        h = self.norm(h, cond)
        logits = self.fc_out(h)
        
        # newly added
        h_0 = self.norm(h_0, cond)
        logits_0 = self.fc_out(h_0)
        h_1 = self.norm(h_1, cond)
        logits_1 = self.fc_out(h_1)
        
        if not include_loss:
            return logits,logits_0,logits_1
        loss = F.cross_entropy(shift_dim(logits, -1, 1), targets[:, 1:])

        # Add the constraint loss from PhyDNet
        k2m = K2M([self.kernel_size, self.kernel_size]).to(x.device)
        for b in range(0,self.phycell.cell_list[0].input_dim):
            filters = self.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)
            m = k2m(filters.double())
            m  = m.float()
            loss += self.constraint_criterion(m, self.constraints.to(x.device)) # constrains is a precomputed matrix   

        return loss, logits

    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch['video']

        cond = dict()
        if self.args.class_cond:
            label = batch['label']
            cond['class_cond'] = F.one_hot(label, self.args.class_cond_dim).type_as(x)
        if self.use_frame_cond:
            cond['frame_cond'] = x[:, :, :self.args.n_cond_frames]

        with torch.no_grad():
            targets, x = self.vqvae.encode(x, include_embeddings=True)
            x = shift_dim(x, 1, -1)

        # loss, _ = self(x, targets, cond)
        loss, logits = self(x, targets, cond)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, default='kinetics_stride4x4x4',
                            help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--n_cond_frames', type=int, default=0)
        parser.add_argument('--class_cond', action='store_true')

        # SkyGPT hyperparmeters
        parser.add_argument('--hidden_dim', type=int, default=576)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--layers', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--attn_type', type=str, default='full',
                            choices=['full', 'sparse'])
        parser.add_argument('--attn_dropout', type=float, default=0.3)

        return parser
