import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.sf_simple import SFSimpleAgent


class PixelsReconstruction(nn.Module):
    def __init__(self, obs_shape, feature_dim, sf_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim_original = 32 * 35 * 35
        self.repr_dim = 32 * 42 * 42

        self.decompress = nn.Sequential(
            nn.Linear(sf_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, self.repr_dim),
        )

        # Transposed convolutional layers
        self.deconvnet = nn.Sequential(
            # Start reconstructing the spatial dimensions; adjustments might be needed based on the exact encoder output dimensions
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),  # Maintain size
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            # Final layer to upsample and reduce channels to match the original image depth
            nn.ConvTranspose2d(32, obs_shape[0], 4, stride=2, padding=1),
        )

        self.apply(utils.weight_init)

    def forward(self, x):
        x = self.decompress(x)
        x = x.view(-1, 32, 42, 42)
        return self.deconvnet(x)


class SFReconstructAgent(SFSimpleAgent):
    def __init__(
        self,
        update_task_every_step,
        sf_dim,
        num_init_steps,
        lr_task,
        w_reconstruction,
        normalize_basis_features,
        normalize_task_params,
        **kwargs,
    ):

        self.w_reconstruction = w_reconstruction

        # create actor and critic
        super().__init__(
            update_task_every_step,
            sf_dim,
            num_init_steps,
            lr_task,
            normalize_basis_features,
            normalize_task_params,
            **kwargs,
        )

        self.pixels_reconstruction = PixelsReconstruction(
            self.obs_shape, self.feature_dim, self.sf_dim
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.pixels_reconstruction_opt = torch.optim.Adam(
            self.pixels_reconstruction.parameters(), lr=self.lr
        )

        self.train()
        self.critic_target.train()

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, reward, discount, next_obs, task = utils.to_torch(
            batch, self.device
        )

        original_img_aug = self.aug(obs)
        original_img_aug_normalized = original_img_aug / 255.0 - 0.5
        obs_encoded = self.encoder(original_img_aug)

        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
            if self.normalize_basis_features:
                next_basis_features = F.normalize(next_obs, p=2, dim=-1)
            else:
                next_basis_features = next_obs

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs_encoded = obs_encoded.detach()
            next_obs = next_obs.detach()

        if self.normalize_task_params:
            # normalize task
            task_normalized = F.normalize(task, p=2, dim=-1)
        else:
            task_normalized = task

        # extend observations with normalized task
        obs = torch.cat([obs_encoded, task_normalized], dim=1)
        next_obs = torch.cat([next_obs, task_normalized], dim=1)

        # update meta
        if step % self.update_task_every_step == 0:
            metrics.update(
                self.regress_meta_grad_descent(
                    next_obs, task, reward, step, next_basis_features
                )
            )

        metrics.update(
            self.update_critic(
                obs.detach(),  # critic does not update encoder in sf reconstruction
                action,
                reward,
                discount,
                next_obs,
                task.detach(),
                step,
                obs_encoded,
                original_img_aug_normalized,
            )
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), task.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def update_critic(
        self,
        obs,
        action,
        reward,
        discount,
        next_obs,
        task,
        step,
        obs_encoded,
        original_img_aug=None,
    ):
        """diff is critic takes task as input"""
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_h = self.featureNet(next_obs, next_action)[0]
            target_Q1, target_Q2 = self.critic_target(next_h, next_action, task)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        h = self.featureNet(obs, action)[0]
        Q1, Q2 = self.critic(h, action, task)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # compute reconstruction loss
        assert original_img_aug is not None
        reconstruct_img = self.pixels_reconstruction(obs_encoded)
        assert reconstruct_img.shape == original_img_aug.shape

        reconstruction_loss = F.mse_loss(reconstruct_img, original_img_aug)

        total_loss = critic_loss + (self.w_reconstruction * reconstruction_loss)

        if self.use_tb or self.use_wandb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()
            metrics["reconstruction_loss"] = reconstruction_loss.item()
            metrics["total_loss"] = total_loss.item()

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        self.feature_opt.zero_grad(set_to_none=True)
        self.pixels_reconstruction_opt.zero_grad(set_to_none=True)

        total_loss.backward()

        self.critic_opt.step()
        self.feature_opt.step()
        self.pixels_reconstruction_opt.step()

        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics
