# Agents, states, and objectives: one-page map

Where to find each agent's **state** (observation) and the **function it optimises**
(reward), in one place. Line numbers are exact at time of writing; the descriptions
are the substance, the line numbers are the pointer.

## The shape of it (not as scattered as it looks)

There are only three kinds of thing:

- **One toy world** (foundations phase): `ToyTabularEnv` in `agents/q_learning.py`.
- **One core environment** that holds the real state and the real reward:
  `UffiziEnv` in `environment/uffizi_env.py`.
- **Three thin wrappers** that do **not** re-implement the reward; they change the
  action interface and delegate the per-minute reward back to the core:
  `PlannedRouteEnv` (baseline walk), `RamaArtLoverEnv` (booking), and
  `RamaTouristEnv` (the art-lover env with the tourist's taste swapped in).

So there are **two reward source functions** in the whole project:
`q_learning.py:419` (toy) and `uffizi_env.py:833` (everything deep). The booking
agents add a few booking-phase terms on top in `rama_art_lover_env.py:355-402`.

## The map

| Agent | Env class | State (file:line) | Reward / objective (file:line) |
|---|---|---|---|
| Toy Q-learning | `ToyTabularEnv` | `agents/q_learning.py:304` (`get_state`) | `agents/q_learning.py:353-449`, core at `:419` |
| Deep navigation | `UffiziEnv` | `environment/uffizi_env.py:573` (`_build_observation`) | `environment/uffizi_env.py:659-841`, total at `:833` |
| Art-lover booking | `RamaArtLoverEnv` | `environment/rama_art_lover_env.py:278` (`_obs`) | core + booking terms `rama_art_lover_env.py:355-402` |
| Normal-tourist booking | `RamaTouristEnv` | inherits the art-lover `_obs` | inherits; only the knobs differ (`rama_tourist_env.py:25-73`) |
| Matched baseline walk | `PlannedRouteEnv` | `environment/planned_route_env.py:127` (`_obs`) | core, delegated at `planned_route_env.py:165` (no booking) |

## Per-agent detail

### 1. Toy Q-learning, `ToyTabularEnv` (`agents/q_learning.py`)
- **State** (`:304`, a discrete 5-tuple, about 123k cells): `(room, time-bin[0-9],
  Botticelli-crowd[0-3], Leonardo-crowd[0-3], visited-mask[0-63])`.
- **Action**: `0 = stay`, `1..max_degree = move to a sorted neighbour`; legality mask
  at `:188` (`valid_actions`).
- **Reward** (`:353-449`): core at `:419` is
  `importance * novelty / (1 + TYPE_A_CROWD_ALPHA * density^2) - step_cost`,
  plus closing pressure `:422-424`, gallery congestion `:426-427`, exit bonus `:437`,
  no-exit penalty `:441`.
- **Toy reward constants** (`:129-134`): `novelty_decay=0.5`, `step_cost=0.5`,
  `exit_bonus=15.0`, `closing_pressure=4.0`, `egress_buffer=5`, `noexit_penalty=-50.0`;
  `TYPE_A_CROWD_ALPHA=6.0` from `config`.

### 2. Deep navigation, `UffiziEnv` (`environment/uffizi_env.py`) -- the core
- **State** (`observation_space` at `:326-346`, built in `_build_observation` at
  `:573-653`). Default layout (`5*N+3` dims): one-hot current room, per-room density,
  per-room density trend, normalised time, egress slack, visited flags, per-room
  appreciation progress, fatigue. (A partial-observability variant adds a visibility
  mask block.)
- **Action**: `:326`, `Discrete(1 + max_degree)` (stay / move to a neighbour); legal
  mask via `get_action_mask`.
- **Reward** (`compute_reward` at `:659-841`); total assembled at `:833`:
  `r_art + r_time + r_congestion + r_closing + r_completion + r_boredom + r_checkin + r_wait`.
  - `r_art` = `importance * crowd_factor * novelty`, with
    `crowd_factor = 1/(1 + art_crowd_alpha * density^2)` (`:720`) and plateau
    satiation `novelty` (`:775`);
  - `r_closing` = slack-based egress pressure (`:817-819`);
  - `r_completion` = one-time bonus for fully appreciating a must-see (`:799-804`);
  - `r_checkin` / `r_wait` are the RAMA check-in bonus / waiting penalty (`:831-832`);
  - potential-based tour shaping is added later in `step` at `:1165`.

### 3. Art-lover booking, `RamaArtLoverEnv` (`environment/rama_art_lover_env.py`)
- **State** (`_OBS_DIM=25` at `:67`, space `:79`, built in `_obs` `:278-308`):
  booking phase flag, booking step, chosen lead, **slots-open-by-lead** (the booking
  screen, 4 values), time, the **3 windows held**, the **3 check-ins done**, the
  **access read** of the current room (`is_masterpiece / unbooked / access_now /
  time-to-open`), the **3 learned-arrival priors**, current-room drain, importance,
  density, egress slack.
- **Action**: `:78`, `Discrete(4)`. Phase 1: pick lead in `{1,7,21,35}` days, then
  per-masterpiece book/decline. Phase 2: stay / move-on.
- **Reward**: two parts.
  - the per-minute walk reward is the core `UffiziEnv.compute_reward`, reached via
    `inner.step` at `:389`;
  - the booking-phase terms are in `step` `:355-402`: decline penalty `:369`, the
    decisive **immediate off-pace penalty** `mismatch_k * |window - estimate|` at
    `:377-379`, and the no-show penalty at `:398-402`.

### 4. Normal-tourist booking, `RamaTouristEnv` (`environment/rama_tourist_env.py`)
- **Same state and reward structure as the art lover** (it subclasses it). The only
  changes are in `__init__` (`:25-73`): a recognition-based `importance_vector` (`:36-54`),
  shorter dwells (`dwell_per_importance=1.4`, `:55`), lighter crowd-aversion
  (`art_crowd_alpha=1.0`, `:31`), and the penalty constants `decline_penalty=50`,
  `noshow_penalty=80`, `allow_decline=False` (`:65-67`).

### 5. Matched baseline walk, `PlannedRouteEnv` (`environment/planned_route_env.py`)
- **State** (`_OBS_DIM=8` at `:47`, space `:63`, built in `_obs` `:127-153`): drain,
  importance, density, at-target, fraction-of-plan-done, time, egress, distance-to-next.
- **Action**: `:62`, `Discrete(2)` (stay / move-on).
- **Reward**: delegated to `UffiziEnv.compute_reward` via `inner.step` at `:165`, in
  the **no-intervention** world (crowded masterpieces, no RAMA gate), no booking terms.

## Where the reward weights live (the override chain)

The per-minute reward weights are defined once on the core env and then overridden by
the booking wrappers. To know an agent's effective settings, read these three places
top to bottom.

`UffiziEnv.__init__` (`environment/uffizi_env.py`), the defaults:

| knob | value | line |
|---|---|---|
| `dwell_per_importance` | 3.0 | `:280` |
| `dwell_importance_floor` | 0.0 | `:281` |
| `satiation_shoulder` | 5.0 | `:282` |
| `dwell_crowd_beta` | 1.0 | `:283` |
| `art_crowd_alpha` | 0.5 | `:284` |
| `step_cost` | 0.15 | `:285` |
| `closing_pressure` | 4.0 | `:287` |
| `egress_buffer` | 5 | `:288` |
| `completion_k` | 15.0 | `:290` |
| `boredom_k` | 0.0 (off) | `:291` |
| `tour_weight` | 1.0 | `:297` |
| `tour_threshold` | 7.0 | `:298` |
| `rama_checkin_bonus` | 0.0 (set by wrapper) | `:254` |
| `rama_wait_penalty` | 0.0 (set by wrapper) | `:259` |

`RamaArtLoverEnv.__init__` (`environment/rama_art_lover_env.py`), booking overrides:

| knob | value | line |
|---|---|---|
| `inner.rama_checkin_bonus` | 50.0 (`CHECKIN_BONUS`) | `:80` |
| `inner.rama_wait_penalty` | 2.0 | `:99` |
| `inner.boredom_k` | 4.0 | `:102` |
| `min_gap` | `SLOT_DUR` | `:91` |
| `entry_tolerance` | 150 | `:92` |
| `early_grace` | 150 | `:93` |
| `decline_penalty` | 200.0 | `:103` |
| `allow_decline` | False | `:112` |
| `noshow_penalty` | 100.0 | `:113` |
| `mismatch_k` | 2.0 | `:121` |
| `_learned_arrival` | [0.25, 0.5, 0.75] | `:129` |
| `_arr_alpha` | 0.0 (frozen prior) | `:136` |

Booking constants (top of `rama_art_lover_env.py`): `LEAD_DAYS=[1,7,21,35]`,
`LEAD_MAX=35.0`, `GROUPS=["bott","leo","raph"]`, `CHECKIN_BONUS=50.0`, `_ACTION_N=4`,
`_OBS_DIM=25`.

`RamaTouristEnv.__init__` (`environment/rama_tourist_env.py`), tourist overrides:
`art_crowd_alpha=1.0` (`:31`), recognition `importance_vector` (`:36-54`),
`dwell_per_importance=1.4` (`:55`), `dwell_importance_floor=5.0` (`:56`),
`decline_penalty=50.0` (`:66`), `noshow_penalty=80.0` (`:67`), `allow_decline=False` (`:65`).

## "If you want to change X, edit Y"

- **The per-minute reward (art value, crowd discount, dwell, closing pressure)**:
  `uffizi_env.py:659-841`, weights at `:280-298`.
- **The booking incentives (decline / off-pace / no-show / check-in)**:
  `rama_art_lover_env.py:355-402` (logic) and `:80-121` (weights).
- **What a profile cares about (importance vector, dwell length, crowd-aversion)**:
  the wrapper `__init__` (art lover: defaults; tourist: `rama_tourist_env.py:31-56`).
- **The toy reward**: `q_learning.py:419` and constants `:129-134`.

Note: the trained policies and every committed figure/number depend on the exact
observation vectors and reward weights above. Changing an observation shape
invalidates the saved models (they would need retraining); changing a reward weight
shifts the reported points. A full consolidation into a single reward module is worth
doing for the Vatican reuse, after the presentation, not before.
