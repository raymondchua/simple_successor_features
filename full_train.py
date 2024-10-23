import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import dmc
import utils
import uuid
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader

from video import TrainVideoRecorder, VideoRecorder
from collections import OrderedDict

torch.backends.cudnn.benchmark = True
from absl import logging

from dmc_benchmark import (
    PRIMAL_TASKS,
    PRIMAL_TASKS_RUN_BACKWARD,
    CRL_TASKS_DIFF_REWARD,
    CRL_DIFF_DOMAINS_SAME_REWARD,
    CRL_WALKER_WALK_RUN_TASKS,
    CRL_WALKER_STAND_RUN_TASKS,
    CRL_RUN_JUMP_TASKS,
)


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    logging.info("agent config: %s", cfg)
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        # set the work_dir to the current working directory
        self.work_dir = Path.cwd()
        # add a subdirectory to the work_dir based on mode
        self.work_dir = self.work_dir / cfg.mode

        # create the work_dir if it does not exist
        self.work_dir.mkdir(exist_ok=True, parents=True)

        # print the work_dir
        print(f"workspace: {self.work_dir}")


        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

        # create envs

        if self.cfg.single_task_run_fast:
            logging.info("Single task run fast")
            self.tasks = [PRIMAL_TASKS_FAST_RUN[self.cfg.domain]]
            self.cfg.terminate_after_first_task = True

        elif self.cfg.single_task_run_backward:
            logging.info("Single task run backward")
            self.tasks = [PRIMAL_TASKS_RUN_BACKWARD[self.cfg.domain]]
            self.cfg.terminate_after_first_task = True

        # walk run tasks (only for walker domain)
        elif self.cfg.walk_run_tasks:
            assert self.cfg.domain == "walker", "Walk run tasks only for walker domain"
            logging.info("Walk run tasks for walker")

            self.tasks = CRL_WALKER_WALK_RUN_TASKS[self.cfg.domain]

        # stand run tasks (only for walker domain)
        elif self.cfg.stand_run_tasks:
            assert self.cfg.domain == "walker", "Stand run tasks only for walker domain"
            logging.info("Stand run tasks for walker")

            self.tasks = CRL_WALKER_STAND_RUN_TASKS[self.cfg.domain]

        # run jump tasks (only for quadruped domain)
        elif self.cfg.run_jump_tasks:
            assert self.cfg.domain == "quadruped", "Run jump tasks only for quadruped domain"
            logging.info("Run jump tasks for quadruped")

            self.tasks = CRL_RUN_JUMP_TASKS[self.cfg.domain]


        # different dynamics but same reward function for all tasks
        elif (
            self.cfg.diff_dynamics_for_all_tasks and self.cfg.same_reward_for_all_tasks
        ):
            logging.info("Different dynamics but same reward function for all tasks")
            self.tasks = CRL_TASKS_SAME_REWARD[self.cfg.domain]

        # different dynamics and different reward function for all tasks
        elif (
            self.cfg.diff_dynamics_for_all_tasks
            and not self.cfg.same_reward_for_all_tasks
        ):
            logging.info(
                "Different dynamics and different reward function for all tasks"
            )
            self.tasks = CRL_DIFF_DYNAMICS_DIFF_REWARD[self.cfg.domain]

        # different run speed but same reward function for all tasks
        elif self.cfg.diff_run_speed_for_all_tasks:
            logging.info("Different run speed but same reward function for all tasks")
            self.tasks = CRL_TASKS_DIFF_RUN_SPEED_REWARD[self.cfg.domain]

        # different domains (eg. cheetah-run and walker-run) with same reward function for all tasks
        elif self.cfg.diff_domains_same_reward:
            logging.info("Different domains with same reward function for all tasks")
            self.tasks = CRL_DIFF_DOMAINS_SAME_REWARD

        # same dynamics but different reward function for all tasks (eg. run forward and backward)
        else:
            assert (
                not self.cfg.diff_dynamics_for_all_tasks
                and not self.cfg.same_reward_for_all_tasks
            ), "Invalid tasks configuration"
            logging.info("Same dynamics but different reward function for all tasks")
            self.tasks = CRL_TASKS_DIFF_REWARD[self.cfg.domain]

        self.num_tasks = len(self.tasks)
        self._current_task_id = 0  # task id always starts from 0

        # create video recorders
        # self.eval_video_recorder = VideoRecorder(
        #     self.work_dir if cfg.save_eval_video else None,
        #     camera_id=0 if "quadruped" not in self.cfg.domain else 2,
        #     use_wandb=self.cfg.use_wandb,
        # )

        # self.train_video_recorder = TrainVideoRecorder(
        #     self.work_dir if cfg.save_train_video else None,
        #     camera_id=0 if "quadruped" not in self.cfg.domain else 2,
        #     use_wandb=self.cfg.use_wandb,
        # )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._exposure_id = 0

        self.train_envs = []
        self.eval_envs = []

        for i in range(self.num_tasks):
            task = self.tasks[i]

            # create new training and eval environment
            train_env = dmc.make(
                task,
                self.cfg.obs_type,
                self.cfg.frame_stack,
                self.cfg.action_repeat,
                self.cfg.seed,
            )
            eval_env = dmc.make(
                task,
                self.cfg.obs_type,
                self.cfg.frame_stack,
                self.cfg.action_repeat,
                self.cfg.seed,
            )

            self.train_envs.append(train_env)
            self.eval_envs.append(eval_env)

            logging.info("task: %s", task)

        obs_spec = self.train_envs[0].observation_spec()
        action_spec = self.train_envs[0].action_spec()

        # create agent
        self.agent = make_agent(
            cfg.obs_type,
            obs_spec,
            action_spec,
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )

        logging.info("num of parameters: %s", self.agent.num_params())

        logging.info("agent: %s", self.agent.__class__.__name__)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (
            obs_spec,
            action_spec,
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # generate a uuid for the replay directory
        self.uuid = uuid.uuid4().hex

        # add only uuid to replay directory
        replay_dir = self.work_dir / "buffer" / self.uuid

        # log the replay directory
        logging.info("replay_dir: %s", replay_dir)

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs, replay_dir, self.cfg.env_type)

        # create replay buffer
        self.replay_loader = make_replay_loader(
            self.replay_storage,
            cfg.replay_buffer_size,
            cfg.batch_size,
            cfg.replay_buffer_num_workers,
            False,
            cfg.nstep,
            cfg.discount,
        )

        self._replay_iter = None

        # flatten the cfg file
        self._cfg_flatten = utils.dictionary_flatten(self.cfg)

        logging.info("{}\n".format(self._cfg_flatten))

        # create logger
        if cfg.use_wandb:
            exp_name = "_".join(
                [
                    cfg.experiment,
                    cfg.agent.name,
                    cfg.domain,
                    cfg.obs_type,
                    str(cfg.seed),
                ]
            )

            # get current working directory and add wandb_dir
            wandb_dir_absolute = Path.cwd()

            # convert wandb_dir_absolute to string
            wandb_dir_str = wandb_dir_absolute.as_posix()

            # log wandb_dir_str
            logging.info("wandb_dir_str: %s", wandb_dir_str)

            project_name = "continual_rl" + self.cfg.domain + "_" + self.cfg.mode
            wandb.init(
                project=project_name,
                group=cfg.agent.name,
                name=exp_name,
                config=self._cfg_flatten,
                dir=wandb_dir_str,
                mode="offline",
                settings=wandb.Settings(
                    start_method="thread"
                ),  # required for offline mode
            )

        else:
            wandb.init(mode="disabled")

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self, task_id: int = None, meta=None):

        assert meta is not None, "meta must be provided for evaluation"

        current_eval_env = self.eval_envs[task_id]
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = current_eval_env.reset()
            # self.eval_video_recorder.init(current_eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, meta, self.global_step, eval_mode=True
                    )
                time_step = current_eval_env.step(action)
                # self.eval_video_recorder.record(current_eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            # self.eval_video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            log("task_id", self._current_task_id)
            log("exposure_id", self._exposure_id)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        total_returns = 0

        for exposure_id in range(self.cfg.num_exposures):
            if self.cfg.terminate_after_first_task and exposure_id > 0:
                break
            for task_id in range(self.num_tasks):
                total_returns_task = 0
                if self.cfg.terminate_after_first_task and task_id > 0:
                    break
                task_step = 0
                self._current_task_id = task_id
                self._exposure_id = exposure_id

                # create new training and eval environment
                current_train_env = self.train_envs[task_id]

                if self.cfg.reset_buffer_every_task:
                    self.replay_storage.clear()

                episode_step, episode_reward = 0, 0
                time_step = current_train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                # self.train_video_recorder.init(time_step.observation)
                metrics = None
                while train_until_step(task_step + 1):

                    if time_step.last():
                        self._global_episode += 1
                        # self.train_video_recorder.save(f"{self.global_frame}.mp4")
                        # wait until all the metrics schema is populated
                        if metrics is not None:
                            # log stats
                            elapsed_time, total_time = self.timer.reset()
                            episode_frame = episode_step * self.cfg.action_repeat
                            if self.global_episode % self.cfg.log_freq == 0:
                                with self.logger.log_and_dump_ctx(
                                    self.global_frame, ty="train"
                                ) as log:
                                    log("fps", episode_frame / elapsed_time)
                                    log("total_time", total_time)
                                    log("episode_reward", episode_reward)
                                    log("episode_length", episode_frame)
                                    log("episode", self.global_episode)
                                    log("buffer_size", len(self.replay_storage))
                                    log("step", self.global_step)
                                    log("task_id", task_id)
                                    log("total_returns", total_returns)
                                    log("total_returns_task", total_returns_task)
                                    log("exposure_id", exposure_id)

                        # reset env
                        time_step = current_train_env.reset()

                        meta = self.agent.solved_meta
                        self.replay_storage.add(time_step, meta)
                        # self.train_video_recorder.init(time_step.observation)
                        # try to save snapshot
                        episode_step = 0
                        episode_reward = 0

                    if seed_until_step(self.global_step):
                        meta = self.agent.init_meta()
                    else:
                        meta = self.agent.solved_meta

                    # try to evaluate
                    if eval_every_step(self.global_step):
                        self.logger.log(
                            "eval_total_time",
                            self.timer.total_time(),
                            self.global_frame,
                        )
                        self.eval(task_id, meta)

                    # sample action
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            meta,
                            self.global_step,
                            eval_mode=False,
                        )

                    # try to update the agent
                    if not seed_until_step(self.global_step):
                        # if agent has consolidation attribute, check if the exposure and task id is 0.
                        # if yes, set no_back_consolidation to True else False
                        if self.agent.consolidation:
                            no_back_consolidation = exposure_id == 0 and task_id == 0
                            if no_back_consolidation:
                                logging.log_first_n(
                                    logging.INFO,
                                    "Using no back flow consolidation loss",
                                    1,
                                )
                            else:
                                logging.log_first_n(
                                    logging.INFO,
                                    "Using the normal consolidation loss",
                                    1,
                                )
                            metrics = self.agent.update(
                                self.replay_iter,
                                self.global_step,
                                no_back_consolidation,
                            )
                        else:
                            metrics = self.agent.update(
                                self.replay_iter, self.global_step
                            )
                        self.logger.log_metrics(metrics, self.global_frame, ty="train")

                    time_step = current_train_env.step(action)
                    episode_reward += time_step.reward
                    total_returns += time_step.reward
                    total_returns_task += time_step.reward

                    self.replay_storage.add(time_step, meta)
                    # self.train_video_recorder.record(time_step.observation)
                    episode_step += 1
                    self._global_step += 1
                    task_step += 1

                # save snapshot at the end of each task
                if self.cfg.save_snapshot_after_each_task:
                    self.save_snapshot()

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)

        if self.agent.consolidation:
            snapshot_dir = (
                snapshot_dir / f"beaker_capacity_{self.cfg.agent.beaker_capacity}"
            )
            snapshot_dir = snapshot_dir / f"init_tube_{self.cfg.agent.init_tube}"
            snapshot_dir = snapshot_dir / f"num_beakers_{self.cfg.agent.num_beakers}"
            snapshot_dir = (
                snapshot_dir
                / f"max_grad_norm_consolidation{self.cfg.agent.max_grad_norm_consolidation}"
            )

        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f"snapshot_{self.global_frame}.pt"
        keys_to_save = [
            "agent",
            "_global_step",
            "_global_episode",
            "_exposure_id",
            "_current_task_id",
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)
            logging.info(f"snapshot saved to {snapshot}")

    def load_snapshot(self, snapshot_path):
        with snapshot_path.open("rb") as f:
            payload = torch.load(f)
            for k, v in payload.items():
                self.__dict__[k] = v
            logging.info(f"snapshot loaded from {snapshot_path}")


@hydra.main(config_path=".", config_name="full_train", version_base=None)
def main(cfg):
    from full_train import Workspace as W

    workspace = W(cfg)
    workspace.train()


if __name__ == "__main__":
    main()
