"""Phase 3 headline: MaskablePPO training entrypoint.

Why MaskablePPO?
----------------
The Uffizi environment is a *graph*: at each step the agent occupies a room
and can only move to adjacent rooms (plus "stay").  Standard PPO treats every
action index as equally plausible, meaning ~80 % of the 100+ actions are
physically impossible at any given state.  Without masking, the policy must
learn from scratch that these actions are illegal, wasting gradient signal on
penalizing impossible moves and dramatically slowing convergence.

MaskablePPO (from sb3-contrib) solves this by zeroing out log-probabilities
for invalid actions *before* the softmax, so the policy only ever assigns
probability mass to legal moves.  This is mathematically equivalent to
restricting the action space at each step, but avoids the overhead of
rebuilding a variable-size action space every timestep.

Why a curriculum?
-----------------
Training directly on the full environment (5 000 visitors, 180-minute
episodes) makes early learning extremely noisy: the agent receives sparse,
delayed rewards from hundreds of visitor interactions.  A 3-stage curriculum
starts with a simpler version (fewer visitors, shorter episodes) so the
agent can learn basic room-selection heuristics quickly, then gradually
transitions to the real-scale problem.  See ``_curriculum_stages`` for
the stage definitions.

Parallel environments
---------------------
When n_envs > 1, SubprocVecEnv spawns each environment in its own process.
Each subprocess collects rollouts independently; PPO aggregates the
experience.  Near-linear speedup on multi-core CPUs.  The default
``_DEFAULT_N_ENVS`` uses half the available cores (clamped to [1, 8])
to leave headroom for the OS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np

from uffizi_rl import config
from uffizi_rl.environment.uffizi_env import UffiziEnv, get_action_mask

# =============================================================================
# Parallelism defaults
# =============================================================================

# Default number of parallel environments.  Uses half the available CPU
# cores (leaving headroom for OS and other processes), clamped to [1, 8].
# Each subprocess runs a full copy of UffiziEnv, so memory scales linearly
# with n_envs.  8 is a practical ceiling for a laptop; a cluster job can
# override via the ``n_envs`` argument.
_DEFAULT_N_ENVS = max(1, min(8, (os.cpu_count() or 2) // 2))


# =============================================================================
# Result container
# =============================================================================

@dataclass
class TrainResult:
    """Summary statistics returned after a PPO training run (or fallback).

    Fields
    ------
    algorithm : str
        Identifier string, e.g. "MaskablePPO_x4" or "random_fallback".
    timesteps : int
        Total environment steps consumed during training (0 for fallbacks).
    mean_episode_reward / std_episode_reward : float
        Reward statistics over ``episodes_eval`` post-training evaluation
        episodes.
    episodes_eval : int
        Number of deterministic rollouts used to compute the reward stats.
    dependency_available : bool
        False when sb3-contrib is not installed and the run fell back to a
        random-action baseline.
    """

    algorithm: str
    timesteps: int
    mean_episode_reward: float
    std_episode_reward: float
    episodes_eval: int
    dependency_available: bool


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_policy(model, env: UffiziEnv, n_episodes: int = 5) -> Dict[str, float]:
    """Evaluate a trained policy deterministically over ``n_episodes`` rollouts.

    Action masks are passed to ``model.predict`` when the model supports them
    (MaskablePPO does; vanilla PPO/DQN do not).  The try/except handles the
    latter case transparently so the same function works for ablations.

    Returns a dict with "mean" and "std" of cumulative episode rewards.
    Uses Bessel-corrected std (ddof=1) for an unbiased estimate; returns
    0.0 when only one episode was run (std undefined).
    """

    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ep_reward = 0.0
        while not (done or trunc):
            action_mask = env.get_action_mask()
            try:
                # MaskablePPO accepts action_masks; vanilla models raise TypeError.
                action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            except TypeError:
                # Fallback for models that do not support masking (ablations).
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(int(action))
            ep_reward += reward
        rewards.append(ep_reward)
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0),
    }


# =============================================================================
# Fallback (no sb3-contrib)
# =============================================================================

def _fallback_random_rollout(seed: int, eval_episodes: int) -> TrainResult:
    """Fallback evaluation used when sb3-contrib is not installed.

    Runs a mask-aware *random* policy: at each step, uniformly samples from
    the set of legal (mask == 1) actions.  This provides a lower-bound
    baseline that still respects graph topology, unlike truly uniform random
    which would attempt illegal moves.

    The result is returned with ``dependency_available=False`` so callers
    can detect that no real training occurred.
    """

    env = UffiziEnv(seed=seed, episode_minutes=180)
    rng = config.get_rng(seed)

    rewards = []
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ep_reward = 0.0
        while not (done or trunc):
            mask = env.get_action_mask()
            # np.flatnonzero(mask) gives indices where mask == 1 (legal actions).
            action = int(rng.choice(np.flatnonzero(mask)))
            obs, reward, done, trunc, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)

    return TrainResult(
        algorithm="random_fallback",
        timesteps=0,
        mean_episode_reward=float(np.mean(rewards)),
        std_episode_reward=float(np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0),
        episodes_eval=eval_episodes,
        dependency_available=False,
    )


# =============================================================================
# Environment factories
# =============================================================================

def _make_env(seed: int, rank: int, env_kwargs: Dict[str, int | float | bool]):
    """Factory that returns a zero-argument callable for SubprocVecEnv.

    SubprocVecEnv expects a *list of callables*, each returning a Gym env.
    The callable is invoked inside the child process, so the import of
    ActionMasker happens there (avoiding pickling issues).

    Each subprocess gets a distinct seed (``seed + rank``) for independent
    randomness across parallel environments.

    ActionMasker wraps UffiziEnv so that MaskablePPO can query the current
    legal-action mask via ``env.action_masks()``.
    """

    def _init():
        from sb3_contrib.common.wrappers import ActionMasker

        env = UffiziEnv(seed=seed + rank, **env_kwargs)
        return ActionMasker(env, get_action_mask)

    return _init


# =============================================================================
# Curriculum design
# =============================================================================

def _curriculum_stages(total_timesteps: int) -> list[tuple[int, Dict[str, int | float | bool]]]:
    """Return a list of (timesteps, env_kwargs) pairs defining the curriculum.

    3-stage curriculum (when total_timesteps >= 20 000)
    ---------------------------------------------------
    Stage 1 (15 % of budget): 1 800 visitors, 120-min episodes.
        Few visitors and short episodes let the agent learn basic movement
        heuristics (do not stay in one room forever, visit high-importance
        rooms) without drowning in noise from 5 000 interacting NPCs.
    Stage 2 (35 % of budget): 3 200 visitors, 150-min episodes.
        Intermediate crowd density.  The agent starts learning crowd-aware
        routing: avoid the Botticelli chain when it is saturated, time
        visits to magnet rooms during lulls.
    Stage 3 (50 % of budget): full-scale (5 000 visitors, 180 min).
        The real problem.  The agent refines its policy under realistic
        conditions, building on the heuristics acquired in earlier stages.

    For very short runs (< 20 000 steps, e.g. smoke tests), the curriculum
    is skipped entirely and training runs at full scale in a single stage.

    The last stage absorbs any rounding residual so total steps are exact.
    """

    # Skip curriculum for short runs (smoke tests, CI).
    if total_timesteps < 20_000:
        return [(int(total_timesteps), {"daily_total": config.DAILY_VISITORS_NORMAL, "episode_minutes": 180})]

    # Fraction of total budget allocated to each stage. [assumption]
    stage_fracs = [0.15, 0.35, 0.50]
    stage_steps = [int(total_timesteps * frac) for frac in stage_fracs]
    # Absorb rounding residual into the last (hardest) stage.
    stage_steps[-1] = int(total_timesteps - sum(stage_steps[:-1]))
    stage_envs = [
        # Stage 1: easy. Low crowd, short episodes.
        {"daily_total": 1800, "episode_minutes": 120},
        # Stage 2: medium. Moderate crowd, medium episodes.
        {"daily_total": 3200, "episode_minutes": 150},
        # Stage 3: full scale. Matches the real Uffizi scenario.
        {"daily_total": config.DAILY_VISITORS_NORMAL, "episode_minutes": 180},
    ]
    return list(zip(stage_steps, stage_envs))


# =============================================================================
# Main training entrypoint
# =============================================================================

def train_maskable_ppo(
    timesteps: int = config.PPO_TIMESTEPS_DEFAULT,
    seed: int = config.DEFAULT_SEED,
    eval_episodes: int = 5,
    n_envs: int = 1,
) -> TrainResult:
    """Train MaskablePPO with a 3-stage curriculum and tuned hyperparameters.

    Parameters
    ----------
    timesteps : int
        Total environment steps across all curriculum stages.
    seed : int
        Base random seed.  Parallel envs get seed+rank; curriculum stages
        get seed + 1000*stage_idx to avoid correlated trajectories.
    eval_episodes : int
        Number of deterministic rollouts for post-training evaluation.
    n_envs : int
        Number of parallel SubprocVecEnv workers (1 = single-process).

    Returns
    -------
    TrainResult
        Summary statistics including mean/std reward and algorithm label.

    Smoke-test shortcut
    -------------------
    When ``timesteps < 512``, real training is skipped and a random-action
    fallback is run instead.  This lets CI pipelines import-test the full
    training path without waiting for actual PPO convergence.
    """

    # ---- Smoke-test shortcut (< 512 steps) --------------------------------
    if timesteps < 512:
        fallback = _fallback_random_rollout(seed=seed, eval_episodes=eval_episodes)
        fallback.algorithm = "MaskablePPO_smoke_random"
        return fallback

    # ---- Dependency guard --------------------------------------------------
    # sb3-contrib provides MaskablePPO.  If the package is missing (e.g. in a
    # minimal install), fall back to the random baseline gracefully.
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except Exception:
        return _fallback_random_rollout(seed=seed, eval_episodes=eval_episodes)

    # ---- Vectorized environment builder ------------------------------------
    def _build_vec_env(stage_seed: int, env_kwargs: Dict[str, int | float | bool]):
        """Build a SubprocVecEnv (multi-process) or a single ActionMasker env.

        SubprocVecEnv is preferred when n_envs > 1: it spawns one OS process
        per environment, collecting rollout data in parallel.  Falls back to
        a single-process env if SubprocVecEnv import fails (e.g. on Windows
        without proper multiprocessing support).
        """
        if n_envs > 1:
            try:
                from stable_baselines3.common.vec_env import SubprocVecEnv

                return SubprocVecEnv([_make_env(stage_seed, i, env_kwargs) for i in range(n_envs)])
            except Exception:
                pass
        # Single-process fallback: wrap env directly with ActionMasker.
        env = UffiziEnv(seed=stage_seed, **env_kwargs)
        return ActionMasker(env, get_action_mask)

    # ---- Build curriculum stages -------------------------------------------
    stages = _curriculum_stages(int(timesteps))
    vec_env = _build_vec_env(seed, stages[0][1])

    # ---- Hyperparameters ---------------------------------------------------
    # learning_rate: linear warmup schedule.  Starts at 1e-4 (conservative,
    #   prevents destructive updates early when the policy is random) and
    #   ramps to 3e-4 as training progresses.  ``progress`` goes from 1.0
    #   (start of training) to 0.0 (end), so 1e-4 + 2e-4 * progress gives
    #   3e-4 at the start and 1e-4 at the end: a linear decay. [assumption]
    # n_steps: rollout length per environment before each PPO update.
    #   Base 2048 divided across parallel envs (each env contributes its
    #   share of experience), floored at 1024 to keep variance manageable.
    # batch_size: 256 minibatch size for SGD updates. [assumption]
    # n_epochs: 8 passes over each rollout buffer.  Higher than the PPO
    #   default of 4 because action masking reduces wasted gradient signal,
    #   so more epochs do not overfit as quickly. [assumption]
    # gamma: 0.995.  Close to 1.0 because museum visits are long-horizon
    #   (180 minutes); the agent must value rooms visited late in the
    #   episode nearly as much as early ones. [assumption]
    # gae_lambda: 0.98.  High lambda for low-bias advantage estimates,
    #   appropriate for the long episode lengths. [assumption]
    # ent_coef: 0.01.  Mild entropy bonus to encourage exploration
    #   without making the policy too stochastic. [assumption]
    # clip_range: 0.2.  Standard PPO clipping; limits the policy ratio
    #   to [0.8, 1.2] per update step, preventing catastrophic policy
    #   shifts. [assumption]
    # net_arch: [256, 256].  Two hidden layers of 256 units each.
    #   Sufficient capacity for the ~100-dimensional observation space;
    #   larger networks showed no improvement in preliminary tests. [assumption]
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lambda progress: 1e-4 + 2e-4 * progress,
        n_steps=max(1024, 2048 // max(1, n_envs)),
        batch_size=256,
        n_epochs=8,
        gamma=0.995,
        gae_lambda=0.98,
        ent_coef=0.01,
        clip_range=0.2,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [256, 256]},
    )

    # ---- Curriculum loop ---------------------------------------------------
    for stage_idx, (stage_steps, env_kwargs) in enumerate(stages):
        if stage_idx > 0:
            # Swap to a harder environment for the next curriculum stage.
            # Each stage gets a different seed offset (seed + 1000*stage_idx)
            # to avoid replaying identical NPC arrival sequences.
            next_env = _build_vec_env(seed + 1000 * stage_idx, env_kwargs)
            model.set_env(next_env)
            try:
                vec_env.close()  # release subprocess resources from the old stage
            except Exception:
                pass
            vec_env = next_env
        # reset_num_timesteps=True only on stage 0 so the global step counter
        # accumulates across stages (needed for correct learning-rate decay).
        model.learn(total_timesteps=int(stage_steps), reset_num_timesteps=stage_idx == 0)

    # ---- Post-training evaluation ------------------------------------------
    # Use a fresh env with a distant seed to avoid evaluating on training data.
    eval_env = UffiziEnv(seed=seed + 10_000, episode_minutes=180)
    metrics = evaluate_policy(model, eval_env, n_episodes=eval_episodes)

    # ---- Cleanup -----------------------------------------------------------
    try:
        vec_env.close()
    except Exception:
        pass

    return TrainResult(
        algorithm=f"MaskablePPO_x{n_envs}",
        timesteps=int(timesteps),
        mean_episode_reward=metrics["mean"],
        std_episode_reward=metrics["std"],
        episodes_eval=eval_episodes,
        dependency_available=True,
    )


if __name__ == "__main__":
    res = train_maskable_ppo()
    print(res)
