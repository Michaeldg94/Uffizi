"""Phase 3 ablations: vanilla PPO and DQN *without* action masking.

What is an ablation study?
--------------------------
An ablation study isolates the contribution of a single component by
removing it and measuring the resulting performance change.  Here, the
component under test is *action masking*.  The MaskablePPO agent in
``train_maskable_ppo.py`` zeroes out log-probabilities for illegal actions
before the softmax, so the policy never wastes probability mass on moves
that violate the gallery's graph topology.

This module trains two *unmasked* baselines (vanilla PPO and DQN from
stable-baselines3) on the same environment.  Because these agents can
select any of the ~100 action indices at every step, they will frequently
attempt illegal moves (visiting a room that is not adjacent to the current
room).  Illegal moves receive a fixed penalty (``invalid_action_penalty``),
which must be learned from experience rather than being structurally
prevented.

Why PPO *and* DQN?
------------------
PPO is an on-policy actor-critic; DQN is an off-policy value-based method.
Including both ablations shows that the benefit of action masking is not
specific to one RL algorithm family.  If MaskablePPO outperforms both
unmasked PPO and unmasked DQN, the advantage is attributable to masking,
not to PPO's particular update rule.

Why these serve as controls
---------------------------
- They use the *same* environment, reward function, and observation space
  as MaskablePPO, differing only in the absence of action masking.
- The ``invalid_action_penalty=-0.4`` is a moderate negative reward that
  teaches the agent to avoid illegal moves, but wastes training signal
  that MaskablePPO can devote entirely to learning good routes. [assumption]
- Simpler hyperparameters (no curriculum, no warmup schedule) are
  intentional: ablations should be "reasonable defaults," not separately
  tuned, so the comparison is fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from uffizi_rl import config
from uffizi_rl.environment.uffizi_env import UffiziEnv

# =============================================================================
# Result container
# =============================================================================

@dataclass
class AblationResult:
    """Summary result for one deep-RL ablation run.

    Fields
    ------
    algorithm : str
        "PPO", "DQN", or a fallback variant (e.g. "PPO_fallback_random").
    timesteps : int
        Total environment steps used for training (0 for fallback runs).
    mean_episode_reward / std_episode_reward : float
        Reward statistics over deterministic evaluation episodes.
    dependency_available : bool
        False when stable-baselines3 is not installed.
    """

    algorithm: str
    timesteps: int
    mean_episode_reward: float
    std_episode_reward: float
    dependency_available: bool


# =============================================================================
# Random-action fallback model
# =============================================================================

class _RandomModel:
    """Uniform-random policy over the full (unmasked) action space.

    Used when stable-baselines3 is not installed, providing a lower-bound
    baseline.  Unlike the masked random fallback in ``train_maskable_ppo``,
    this model does *not* respect action masks, so it will frequently
    select illegal actions (and incur ``invalid_action_penalty``).
    """

    def __init__(self, env: UffiziEnv, seed: int):
        self.env = env
        self.rng = config.get_rng(seed)

    def predict(self, obs, deterministic: bool = True):
        """Return a uniformly random action index and None (no state)."""
        return int(self.rng.integers(0, self.env.action_space.n)), None


# =============================================================================
# Evaluation helper
# =============================================================================

def _evaluate(model, env: UffiziEnv, episodes: int = 4) -> tuple[float, float]:
    """Run ``episodes`` deterministic rollouts and return (mean, std) reward.

    No action masking is applied: the model picks any action in [0, N).
    Illegal actions incur the env's ``invalid_action_penalty``, which is
    part of what this ablation is measuring.

    Uses Bessel-corrected std (ddof=1); returns 0.0 for single episodes.
    """
    rews: List[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        total = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(int(action))
            total += r
        rews.append(total)
    return float(np.mean(rews)), float(np.std(rews, ddof=1) if len(rews) > 1 else 0.0)


# =============================================================================
# Main ablation entrypoint
# =============================================================================

def train_ablations(
    timesteps: int = config.ABLATION_TIMESTEPS_DEFAULT,
    seed: int = config.DEFAULT_SEED,
) -> list[AblationResult]:
    """Train (or smoke-test) vanilla PPO and DQN ablations without action masking.

    Parameters
    ----------
    timesteps : int
        Total environment steps for each algorithm.  Both PPO and DQN
        receive the same budget for a fair comparison.
    seed : int
        Base random seed.  Each algorithm gets ``seed + i`` (i=0 for PPO,
        i=1 for DQN) so their random streams are independent but
        reproducible.

    Returns
    -------
    list[AblationResult]
        One result per algorithm (PPO first, DQN second).

    Smoke-test shortcut
    -------------------
    When ``timesteps < 512``, both algorithms are replaced by
    ``_RandomModel`` to allow fast import/integration testing.
    """

    # ---- Smoke-test shortcut -----------------------------------------------
    if timesteps < 512:
        results: list[AblationResult] = []
        for i, name in enumerate(["PPO", "DQN"]):
            # invalid_action_penalty=-0.4: the env returns this reward
            # whenever the agent selects a move that violates graph
            # adjacency.  This is what unmasked agents must learn to avoid.
            env = UffiziEnv(seed=seed + i, invalid_action_penalty=-0.4, episode_minutes=180)
            model = _RandomModel(env, seed=seed + i)
            # Evaluate on a separate env (seed + 100 + i) to avoid data leakage.
            mean_r, std_r = _evaluate(
                model,
                UffiziEnv(seed=seed + 100 + i, episode_minutes=180),
                episodes=3,
            )
            results.append(
                AblationResult(
                    algorithm=f"{name}_smoke_random",
                    timesteps=0,
                    mean_episode_reward=mean_r,
                    std_episode_reward=std_r,
                    dependency_available=True,
                )
            )
        return results

    # ---- Dependency guard --------------------------------------------------
    try:
        from stable_baselines3 import DQN, PPO

        dep_ok = True
    except Exception:
        DQN = PPO = None
        dep_ok = False

    # ---- Train each ablation algorithm -------------------------------------
    results: list[AblationResult] = []
    algs = ["PPO", "DQN"]
    for i, name in enumerate(algs):
        # Same invalid_action_penalty for both algorithms: -0.4 per illegal
        # move.  This is deliberately moderate; too harsh and the agent
        # learns to stay in place (safe but useless), too mild and it
        # ignores the constraint.  [assumption]
        env = UffiziEnv(seed=seed + i, invalid_action_penalty=-0.4, episode_minutes=180)

        if dep_ok and PPO is not None and DQN is not None:
            if name == "PPO":
                # Vanilla PPO (no masking, no curriculum).
                # learning_rate 3e-4: SB3 default, adequate for MLP policies.
                # n_steps 512: shorter than MaskablePPO (2048) because there
                #   is no parallel env; keeps memory usage low.
                # n_epochs 5: fewer than MaskablePPO (8); without masking,
                #   more epochs risk overfitting to noise from illegal moves.
                # gamma 0.99: slightly lower than MaskablePPO (0.995) since
                #   ablations use simpler hyperparameters as a baseline.
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    n_steps=512,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.99,
                    seed=seed + i,
                    verbose=0,
                )
            else:
                # Vanilla DQN (value-based, off-policy).
                # learning_rate 2.5e-4: slightly lower than PPO; DQN is more
                #   sensitive to learning rate because the value target is
                #   non-stationary (moving Q-target). [assumption]
                # buffer_size 50_000: replay buffer.  50k transitions is
                #   sufficient for the ablation budget of 10k steps; larger
                #   buffers waste memory without benefit at this scale.
                # learning_starts 1_000: collect 1k transitions of random
                #   exploration before the first gradient update, ensuring
                #   the replay buffer has enough diversity. [assumption]
                model = DQN(
                    "MlpPolicy",
                    env,
                    learning_rate=2.5e-4,
                    batch_size=64,
                    buffer_size=50_000,
                    learning_starts=1_000,
                    gamma=0.99,
                    seed=seed + i,
                    verbose=0,
                )

            model.learn(total_timesteps=int(timesteps))
        else:
            # stable-baselines3 not installed; use random fallback.
            model = _RandomModel(env, seed=seed + i)

        # ---- Evaluate on a held-out environment ----------------------------
        # Seed offset +100 ensures the evaluation NPC sequences differ from
        # the training ones.
        mean_r, std_r = _evaluate(
            model,
            UffiziEnv(seed=seed + 100 + i, episode_minutes=180),
            episodes=4,
        )
        results.append(
            AblationResult(
                algorithm=name if dep_ok else f"{name}_fallback_random",
                timesteps=int(timesteps) if dep_ok else 0,
                mean_episode_reward=mean_r,
                std_episode_reward=std_r,
                dependency_available=dep_ok,
            )
        )

    return results


if __name__ == "__main__":
    for r in train_ablations():
        print(r)
