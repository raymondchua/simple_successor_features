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

class SFRandomFeatAgent(SFSimpleAgent):
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
            normalize_task_params,
            **kwargs
        )
