import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.ddpg import DDPGAgent, Actor


class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, sf_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.compress = nn.Sequential(
            nn.Linear(self.repr_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, sf_dim),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.compress(h)
        return h


class FeatureNet(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, sf_dim):
        super().__init__()

        self.obs_type = obs_type

        # a small difference compared to aps is that aps uses an additional state_feat_net
        # whereas we directly use the trunk to get the features. Therefore, to get the
        # basis features dim to match the sf_dim, we include an additional linear layer.

        if obs_type == "pixels":
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, sf_dim),
            )
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, sf_dim),
            )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == "pixels" else torch.cat([obs, action], dim=-1)
        h = self.trunk(inpt)
        basis_features = F.normalize(h, p=2, dim=-1)
        return h, basis_features


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

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        q1 = torch.einsum("bi,bi->b", task, q1).reshape(-1, 1)
        q2 = torch.einsum("bi,bi->b", task, q2).reshape(-1, 1)

        return q1, q2


class SFSimpleAgent(DDPGAgent):
    def __init__(
        self,
        update_task_every_step,
        sf_dim,
        num_init_steps,
        lr_task,
        normalize_basis_features,
        normalize_task_params,
        **kwargs
    ):

        self.sf_dim = sf_dim
        self.update_task_every_step = update_task_every_step
        self.num_init_steps = num_init_steps
        # self.update_encoder = update_encoder
        self.lr_task = lr_task
        self.normalize_basis_features = normalize_basis_features
        self.normalize_task_params = normalize_task_params

        # increase obs shape to include task dim
        kwargs["meta_dim"] = self.sf_dim

        # create actor and critic
        super().__init__(**kwargs)

        if self.obs_type == "pixels":
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(self.obs_shape, self.feature_dim, self.sf_dim).to(
                self.device
            )
            self.obs_dim = self.sf_dim + self.meta_dim
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = self.obs_shape[0] + self.meta_dim
            self.encoder_opt = None

        self.featureNet = FeatureNet(
            self.obs_type,
            self.obs_dim,
            self.action_dim,
            self.feature_dim,
            self.hidden_dim,
            self.sf_dim,
        ).to(self.device)

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

        self.actor = Actor(
            self.obs_type,
            self.obs_dim,
            self.action_dim,
            self.feature_dim,
            self.hidden_dim,
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.feature_opt = torch.optim.Adam(self.featureNet.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.task_params = nn.Parameter(
            torch.randn(self.sf_dim, requires_grad=True, device=self.device)
        )

        self.task_opt = torch.optim.Adam([self.task_params], lr=self.lr_task)

        # set solved_meta to the value of the task_params
        with torch.no_grad():
            self.solved_meta = OrderedDict()
            self.solved_meta["task"] = self.task_params.detach().cpu().numpy()

        self.train()
        self.critic_target.train()
        self.featureNet.train()

    def get_meta_specs(self):
        """
        Meta dimension always follows the successor feature dimension.
        """
        return (specs.Array((self.sf_dim,), np.float32, "task"),)

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        task = torch.randn(self.sf_dim)
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta["task"] = task
        return meta

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, reward, discount, next_obs, task = utils.to_torch(
            batch, self.device
        )

        if self.normalize_task_params:
            # normalize task
            task_normalized = F.normalize(task, p=2, dim=-1)
        else:
            task_normalized = task

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
            if self.normalize_basis_features:
                next_basis_features = F.normalize(next_obs, p=2, dim=-1)
            else:
                next_basis_features = next_obs

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with normalized task
        obs = torch.cat([obs, task_normalized], dim=1)
        next_obs = torch.cat([next_obs, task_normalized], dim=1)

        # update meta
        if step % self.update_task_every_step == 0:
            metrics.update(
                self.regress_meta_grad_descent(
                    next_obs, task, reward, step, next_basis_features
                )
            )

        # update critic
        metrics.update(
            self.update_critic(
                obs, action, reward, discount, next_obs, task.detach(), step
            )
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), task.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_task_every_step == 0:
            return self.init_meta()
        return meta

    def regress_meta_grad_descent(
        self, next_obs, task, reward, step, next_basis_features=None
    ):
        metrics = dict()

        if self.obs_type == "pixels":
            predicted_reward = torch.einsum(
                "bi,i->b", next_basis_features, self.task_params
            ).reshape(-1, 1)

        else:
            with torch.no_grad():
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                next_rep = self.featureNet(next_obs, next_action)[-1]
            predicted_reward = torch.einsum(
                "bi,i->b", next_rep, self.task_params
            ).reshape(-1, 1)

        reward_prediction_loss = F.mse_loss(predicted_reward, reward)

        self.task_opt.zero_grad(set_to_none=True)
        reward_prediction_loss.backward()
        self.task_opt.step()

        # set solved_meta to the value of the task_params
        with torch.no_grad():
            meta = self.task_params.detach().cpu().numpy()
            # normalize solved meta using l2 norm
            # meta = meta / np.linalg.norm(meta)
            self.solved_meta = OrderedDict()
            self.solved_meta["task"] = meta

        if self.use_tb or self.use_wandb:
            metrics["reward_prediction_loss"] = reward_prediction_loss.item()
            metrics["task_grad_norm"] = self.task_params.grad.norm().item()

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, task, step):
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

        if self.use_tb or self.use_wandb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
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
        Q1, Q2 = self.critic(h, action, task)
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

    @torch.no_grad()
    def solved_meta(self):
        return self.solved_meta

    @torch.no_grad()
    def num_params(self):
        all_params = (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.featureNet.parameters())
            + list(self.encoder.parameters())
        )

        num_parameters = sum([params.numel() for params in all_params])
        num_parameters += self.meta_dim

        return num_parameters
