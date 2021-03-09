Implementation of the [phasic policy
gradient](https://arxiv.org/abs/2009.04416) (PPG) algorithm for
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

The CNN policy with an auxiliary head is currently missing, so you can
only use the `AuxMlpPolicy`.

To initialize the policy with the paper's initialization values,
uncomment the code for `init_weights` in [./ppg/aux_ac_policy.py]
