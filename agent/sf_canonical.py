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


class CriticSF(nn.Module):
    def __init__(self, obs_type, action_dim, hidden_dim, sf_dim):
        super().__init__()

        self.obs_type = obs_type

        # a small difference compared to aps is that aps uses an additional state_feat_net
        # whereas we directly use the trunk to get the features. Therefore, to get the
        # basis features dim to match the sf_dim, we include an additional linear layer.

        if obs_type == "pixels":
            trunk_dim = sf_dim + action_dim
        else:
            trunk_dim = sf_dim

        def make_q():
            q_layers = []
            q_layers += [nn.Linear(trunk_dim, hidden_dim), nn.ReLU(inplace=True)]
            if obs_type == "pixels":
                q_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            q_layers += [nn.Linear(hidden_dim, sf_dim)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, h, action, task):
        h = torch.cat([h, action], dim=-1) if self.obs_type == "pixels" else h

        sf1 = self.Q1(h)
        sf2 = self.Q2(h)

        q1 = torch.einsum("bi,bi->b", task, sf1).reshape(-1, 1)
        q2 = torch.einsum("bi,bi->b", task, sf2).reshape(-1, 1)

        return q1, q2, sf1, sf2


class SFCanonicalAgent(SFSimpleAgent):
    def __init__(
        self,
        update_task_every_step,
        sf_dim,
        num_init_steps,
        lr_task,
        normalize_basis_features,
        **kwargs
    ):
        self.sf_dim = sf_dim
        self.update_task_every_step = update_task_every_step
        self.num_init_steps = num_init_steps
        self.lr_task = lr_task

        # increase obs shape to include task dim
        kwargs["meta_dim"] = self.sf_dim

        # create actor and critic
        super().__init__(
            update_task_every_step,
            sf_dim,
            num_init_steps,
            lr_task,
            normalize_basis_features,
            **kwargs
        )

        # overwrite critic with critic sf
        self.critic = CriticSF(
            self.obs_type,
            self.action_dim,
            self.hidden_dim,
            self.sf_dim,
        ).to(self.device)

        self.critic_target = CriticSF(
            self.obs_type,
            self.action_dim,
            self.hidden_dim,
            self.sf_dim,
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.train()
        self.critic_target.train()

    def update_critic(
        self, obs, action, reward, discount, next_obs, task, step, basis_features
    ):
        """
        critic here is the sf-td loss. We learn both the basis features and the successor features at the same
        time.
        """
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_h = self.featureNet(next_obs, next_action)[0]
            target_Q1, target_Q2, target_sf1, target_sf2 = self.critic_target(
                next_h, next_action, task
            )

            # compute the l2 norm of the target sf
            target_sf1_norm = torch.norm(target_sf1, p=2, dim=-1)
            target_sf2_norm = torch.norm(target_sf2, p=2, dim=-1)
            sf_norm_compare = target_sf1_norm < target_sf2_norm

            target_sf = []

            # for rows in sf_norm_compare if true then target_sf1 else target_sf2. The result is a tensor
            # of shape (batch_size, sf_dim)
            for idx, row in enumerate(sf_norm_compare):
                if row:
                    target_sf.append(target_sf1[idx, :])
                else:
                    target_sf.append(target_sf2[idx, :])

            target_sf = torch.stack(target_sf)

            target = basis_features + (discount * target_sf)

        h = self.featureNet(obs, action)[0]
        Q1, Q2, SF1, SF2 = self.critic(h, action, task)
        critic_loss = F.mse_loss(SF1, target) + F.mse_loss(SF2, target)

        if self.use_tb or self.use_wandb:
            metrics["critic_target_sf"] = target_sf.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_sf1"] = SF1.mean().item()
            metrics["critic_sf2"] = SF2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self.feature_opt.zero_grad(set_to_none=True)

        critic_loss.backward()
        self.critic_opt.step()
        self.feature_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, task, step):
        """diff is critic takes task as input"""
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        h = self.featureNet(obs, action)[0]
        Q1, Q2, _, _ = self.critic(h, action, task)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, reward, discount, next_obs, task = utils.to_torch(
            batch, self.device
        )

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.normalize_basis_features:
            basis_features = F.normalize(obs, p=2, dim=-1)
        else:
            basis_features = obs

        next_obs = next_obs.detach()

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # normalize task
        task_normalized = F.normalize(task, p=2, dim=-1)

        # extend observations with normalized task
        obs = torch.cat([obs, task_normalized], dim=1)
        next_obs = torch.cat([next_obs, task_normalized], dim=1)

        # update critic which includes the sf-td loss, reward prediction loss and the critic loss
        metrics.update(
            self.update_critic(
                obs,
                action,
                reward,
                discount,
                next_obs,
                task.detach(),
                step,
                basis_features,
            )
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), task.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
