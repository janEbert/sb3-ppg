from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from stable_baselines3.common import logger
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, \
    Schedule
from stable_baselines3.common.utils import get_schedule_fn, \
    update_learning_rate
from stable_baselines3.ppo import PPO
import torch as th
from torch import distributions as td
from torch.nn import functional as F

from .aux_ac_policy import AuxActorCriticPolicy


class PPG(PPO):
    """
    Phasic Policy Gradient algorithm (PPG) (with PPO clip version)
    This version does not support a different number of policy and value
    optimization phases.
    Paper: https://arxiv.org/abs/2009.04416
    Code: This implementation borrows code from Stable Baselines 3
    (PPO from https://github.com/DLR-RM/stable-baselines3)
    and the OpenAI implementation
    (https://github.com/openai/phasic-policy-gradient)
    Introduction to PPO (which PPG improves upon):
    https://spinningup.openai.com/en/latest/algorithms/ppo.html
    :param policy: The policy model to use (AuxMlpPolicy, AuxCnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can
        be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param aux_learning_rate: The learning rate for the auxiliary optimizer,
        it can be a function of the current progress remaining (from 1 to 0).
        If None, use the same function as learning_rate.
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of
        environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the
        advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param aux_batch_size: Auxiliary minibatch size
    :param n_phases: Number of optimization phases (most outer loop)
    :param n_epochs: Number of epochs when optimizing the surrogate loss
    :param n_aux_epochs: Number of epochs for the auxiliary phase
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for
        Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current
        progress remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is
        passed (default), no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param beta_clone: Trade-off between optimizing auxiliary objective and
        original policy
    :param vf_true_coef: Non-auxiliary value function coefficient for the joint
        loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent
        Exploration (gSDE) instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when
        using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213
        (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None,
        no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when
        passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy
        on creation. Note that in the PPG paper, no activation function and a
        flat model was used.
    :param use_paper_parameters: Whether to overwrite the other parameters with
        those from the PPG paper
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the
        creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[AuxActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,  # 5e-4 in paper
        # 5e-4 in paper
        aux_learning_rate: Union[None, float, Schedule] = None,
        n_steps: int = 2048,  # 256 in paper
        batch_size: Optional[int] = 64,  # 8 in paper
        aux_batch_size: Optional[int] = 32,  # 4 in paper
        n_phases: int = 1,
        n_epochs: int = 10,  # 32 in paper
        n_aux_epochs: int = 2,  # 6 in paper
        gamma: float = 0.99,  # 0.999 in paper
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,  # 0.01 in paper
        vf_coef: float = 0.5,
        beta_clone: float = 1.0,
        vf_true_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        use_paper_parameters: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(PPG, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        if aux_learning_rate is None:
            self.aux_learning_rate = learning_rate
        else:
            self.aux_learning_rate = aux_learning_rate
        self.aux_batch_size = aux_batch_size
        self.n_phases = n_phases
        self.n_aux_epochs = n_aux_epochs
        self.beta_clone = beta_clone
        self.vf_true_coef = vf_true_coef
        self._n_aux_updates = 0

        if use_paper_parameters:
            self._set_paper_parameters()

        self._name2coef = {
            "pol_distance": beta_clone,
            "vf_true": vf_true_coef,
        }

        if _init_setup_model:
            self._setup_model()

    def _set_paper_parameters(self):
        if self.env.num_envs != 64:
            print("Warning: Paper uses 64 environments. "
                  "Change this if you want to have the same setup.")

        self.learning_rate = 5e-4
        self.aux_learning_rate = 5e-4
        self.n_steps = 256
        self.batch_size = 8
        self.aux_batch_size = 4
        self.n_phases = 1
        self.n_epochs = 32
        self.n_aux_epochs = 6
        self.gamma = 0.999
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.clip_range_vf = None
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.beta_clone = 1.0
        self.vf_true_coef = 1.0
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.policy_kwargs["activation_fn"] = th.nn.Identity

    def _setup_model(self) -> None:
        self.aux_lr_schedule = get_schedule_fn(self.aux_learning_rate)
        self.policy_kwargs["aux_lr_schedule"] = self.aux_lr_schedule

        super(PPG, self)._setup_model()

    def _update_learning_rate(
            self,
            optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer],
    ) -> None:
        super(PPG, self)._update_learning_rate(optimizers)
        logger.record("train/aux_learning_rate",
                      self.aux_lr_schedule(self._current_progress_remaining))
        update_learning_rate(
            self.policy.aux_optimizer,
            self.aux_lr_schedule(self._current_progress_remaining))

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, var_list = super(PPG, self)._get_torch_save_params()
        state_dicts.append("policy.aux_optimizer")
        return state_dicts, var_list

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        for phase in range(self.n_phases):
            super(PPG, self).train()

            indices = np.arange(self.rollout_buffer.buffer_size
                                * self.rollout_buffer.n_envs)
            # In the paper, these are re-calculated after updating the policy
            old_pds = np.empty(self.rollout_buffer.observations.shape[0],
                               dtype=object)

            with th.no_grad():
                start_idx = 0
                while start_idx < len(indices):
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    batch_indices = indices[
                        start_idx:start_idx + self.aux_batch_size]
                    obs = self.rollout_buffer.observations[batch_indices]
                    # Convert to pytorch tensor
                    obs_tensor = th.as_tensor(obs).to(self.policy.device)
                    distribution, _, _ = self.policy.forward_policy(obs_tensor)
                    old_pds[start_idx // self.aux_batch_size] = \
                        distribution.distribution
                    start_idx += self.aux_batch_size

            unscaled_aux_losses = []
            aux_losses = []
            for aux_epoch in range(self.n_aux_epochs):
                start_idx = 0
                while start_idx < len(indices):
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    batch_indices = indices[
                        start_idx:start_idx + self.aux_batch_size]
                    obs = self.rollout_buffer.observations[batch_indices]
                    obs_tensor = th.as_tensor(obs).to(self.policy.device)
                    old_pds_batch = old_pds[start_idx // self.aux_batch_size]

                    distribution, value, aux = self.policy.forward_aux(
                        obs_tensor)
                    vtarg = self.rollout_buffer.returns[batch_indices]
                    vtarg = th.as_tensor(vtarg).to(self.policy.device)

                    name2loss = {}
                    name2loss["pol_distance"] = td.kl_divergence(
                        old_pds_batch, distribution.distribution).mean()
                    name2loss["vf_aux"] = 0.5 * F.mse_loss(aux, vtarg)
                    name2loss["vf_true"] = 0.5 * F.mse_loss(value, vtarg)

                    unscaled_losses = {}
                    losses = {}
                    loss = 0
                    for name in name2loss.keys():
                        unscaled_loss = name2loss[name]
                        coef = self._name2coef.get(name, 1)

                        scaled_loss = unscaled_loss * coef
                        unscaled_losses[name] = \
                            unscaled_loss.detach().cpu().numpy()
                        losses[name] = scaled_loss.detach().cpu().numpy()
                        loss += scaled_loss
                    unscaled_aux_losses.append(unscaled_losses)
                    aux_losses.append(losses)

                    self.policy.aux_optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                self.max_grad_norm)
                    self.policy.aux_optimizer.step()
                    start_idx += self.aux_batch_size

            self._n_aux_updates += self.n_aux_epochs
            logger.record("train/n_aux_updates", self._n_aux_updates,
                          exclude="tensorboard")
            for name in name2loss.keys():
                logger.record(f"train/unscaled_aux_{name}_loss", np.mean(
                    [entry[name] for entry in unscaled_aux_losses]))
                logger.record(f"train/aux_{name}_loss", np.mean(
                    [entry[name] for entry in aux_losses]))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPG",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPG":

        return super(PPG, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
