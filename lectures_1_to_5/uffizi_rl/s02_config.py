"""Global configuration for the Uffizi RL project.

[READING ORDER: file 2 of 12 - read after README.md]

Single source of truth for every modeling parameter, constant, and
piece of room metadata. Every value is annotated with its source: a
published reference, an official public source, or an explicit
[assumption] label.

# READING ORDER
The file is organized so the reader can scroll top-to-bottom. Each
section is self-contained and uses only what came before it.

  Section 1. Time and capacity            (museum clock, fire limit)
  Section 2. Demand parameters            (daily visitor counts, RL budgets)
  Section 3. Visitor type model           (Type A vs Type B behavior)
  Section 4. Visit duration model         (slot-dependent dwell times)
  Section 5. Room metadata                (98 rooms: name, importance, capacity)
  Section 6. Derived room indexes         (ROOM_DATA, ROOM_IDS, ROOM_TO_IDX)
  Section 7. Graph topology               (119 edges between rooms)
  Section 8. Routes and named room sets   (RECOMMENDED_ROUTE, VASARI_ROUTE, KIOSK_ROOMS)
  Section 9. Reward shaping weights       (experience quality components)
  Section 10. Extended parameters         (intervention controls, not needed
                                           to follow the value-based baseline)
  Section 11. Utility functions           (RNG seeding, capacity check, helpers)

References
----------
[A22]  Attanasio et al. (2022). "Visitors flow management at Uffizi Gallery
       in Florence, Italy." Information Technology & Tourism, 24(3), 409-434.
       DOI: 10.1007/s40558-022-00231-y
[C21]  Centorrino et al. (2021). "Managing crowded museums: Visitors flow
       measurement, analysis, modeling, and optimization." Journal of
       Computational Science, 53, 101357.
[VL83] Veron & Levasseur (1983). "Ethnographie de l'exposition." BPI,
       Centre Georges Pompidou. Visitor typology: Ant/Butterfly/Fish/Grasshopper.
[UFF]  Official Uffizi website and visitor FAQ (uffizi.it, visituffizi.org).
[MAP]  Official 2023 Uffizi Gallery floor plan (uffizi.it).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


# =============================================================================
# 1. Time and capacity
# =============================================================================
# Two clocks: museum-open duration vs bookable-entry duration. [UFF]
MUSEUM_OPEN_MINUTES = 615  # 08:15 -> 18:30 = 10h15m
LAST_ENTRY_MINUTES = 555   # 08:15 -> 17:30 =  9h15m

# [A22] Table 2: 37 time slots of 15 minutes each, from 08:15 to 17:30.
ENTRY_SLOT_MINUTES = 15
N_ENTRY_SLOTS = 37
TIME_STEP_MINUTES = 1

# Hard occupancy cap. "The law dictates that given the number of
# entrances/exits at the Uffizi, no more than 900 people can be inside
# the museum at any time." [UFF]. We treat this as a hard constraint:
# arrivals are queued outside whenever inside-count reaches 900.
MAX_MUSEUM_CAPACITY = 900

# Seasonal hour variants. The Uffizi extends evening hours in summer
# (last entry at 22:30) and shortens in winter (last entry at 16:30).
SUMMER_OPEN_MINUTES = 855
WINTER_OPEN_MINUTES = 495


# =============================================================================
# 2. Demand parameters
# =============================================================================
# ~5M visitors annually in 2023 [UFF]; ~2M in 2018 [A22]. Peak free-access
# days reached ~8,000 visitors [A22, Section 3]. The two values below are
# modeling parameters calibrated so that the simulator produces realistic
# occupancy pressure under the 900-person cap.
DAILY_VISITORS_NORMAL = 5000   # [assumption] typical non-peak weekday
DAILY_VISITORS_PEAK = 12000    # [assumption] high-season Saturday

# Random seed used everywhere (simulator, RL, sweeps) for reproducibility.
DEFAULT_SEED = 42

# RL training budgets. Quick defaults for local/smoke execution;
# full values for converged runs.
TABULAR_EPISODES_DEFAULT = 2500
TABULAR_EPISODES_FULL = 50000

# Simulator calibration: how many days to average over when reporting
# calibration statistics (e.g., mean peak inside).
SIM_DAYS_CALIBRATION_DEFAULT = 25
SIM_DAYS_CALIBRATION_FULL = 100

# Observation-space sentinel: value placed in unobserved rooms when
# visibility masking is active (used by the Gymnasium environment).
UNKNOWN_DENSITY_SENTINEL = -1.0


# =============================================================================
# 3. Visitor type model (Type A vs Type B)
# =============================================================================
# Two-type model inspired by the Veron & Levasseur (1983) taxonomy [VL83].
#   Type A ("art lovers") ~ VL83 "Butterfly": heterogeneous preferences,
#     crowd-sensitive, willing to deviate from the default path.
#   Type B ("checkbox tourists") ~ VL83 "Ant": concentrated preferences
#     on canonical masterpieces, route-following, crowd-insensitive.
# The 30/70 split is a modeling assumption; no published source provides
# an exact ratio for the Uffizi. [assumption]
TYPE_A_FRACTION_DEFAULT = 0.30
TYPE_B_FRACTION_DEFAULT = 0.70

# Crowd sensitivity in the reward function: r = importance / (1 + alpha * dens).
# Higher alpha means the visitor loses more utility from crowding. [assumption]
TYPE_A_CROWD_ALPHA = 6.0   # strong crowd penalty
TYPE_B_CROWD_ALPHA = 0.5   # weak crowd penalty

# Route-following tendency: probability weight on the "next" room in the
# recommended itinerary when choosing where to move. [assumption]
TYPE_A_ROUTE_BIAS = 0.55
TYPE_B_ROUTE_BIAS = 0.88

# Backtrack probability: per-step chance an NPC reverses direction on
# the recommended route. Prevents perfectly deterministic flow. [assumption]
TYPE_A_BACKTRACK_PROBABILITY = 0.05
TYPE_B_BACKTRACK_PROBABILITY = 0.02   # Type B rarely backtracks

# Anti-crowd bonus: extra weight toward less-crowded neighbors. [assumption]
TYPE_A_ANTI_CROWD_BONUS = 1.0
TYPE_B_ANTI_CROWD_BONUS = 0.0

# Asymmetric externality: Type A perceives Type B visitors as this factor
# more disruptive than fellow Type A visitors. Captures the idea that
# large tour groups with cameras and guides impose disproportionate
# negative externalities on contemplative visitors. [assumption]
TYPE_A_CROSS_TYPE_EXTERNALITY = 1.5

# Type B importance vector: background importance for non-magnet rooms
# (a Type B visitor ignores them); magnet rooms get TYPE_B_MAGNET_IMPORTANCE. [assumption]
TYPE_B_BACKGROUND_IMPORTANCE = 0.7
TYPE_B_MAGNET_IMPORTANCE = 9.5


# =============================================================================
# 4. Visit duration model
# =============================================================================
# Base expected dwell in a room with magnetism=1.0. Actual expected
# dwell is base_dwell * room_magnetism * profile.dwell_multiplier.
BASE_DWELL_MINUTES = 3.0
TYPE_A_DWELL_MULTIPLIER = 1.0
TYPE_B_DWELL_MULTIPLIER = 0.3   # Type B moves on quickly


# =============================================================================
# 5. Room metadata (98 rooms x 6 columns)
# =============================================================================
# Room data derived from the official 2023 Uffizi floor plan [MAP].
# Names follow the museum's own labeling. A41 does not exist in the
# floor plan. A37 exists but is unnamed.
#
# Columns: (room_id, name, section, importance, magnetism, capacity)
#   importance: 1-10, cultural significance of the artworks [assumption,
#               informed by guidebook prominence and visitor survey data]
#   magnetism:  dwell-time multiplier (1.0 ~ 3 min base dwell) [assumption]
#   capacity:   comfortable max occupancy in persons. Derived from
#               pixel-area analysis of the official floor plan
#               (proportional to room footprint). Reference: A7 is
#               13,694 px scaled at ~0.00258 people/px. Corridors use
#               a 0.6x density (transit, not viewing).

SECOND_FLOOR_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    # Entry and Western Corridor
    ("A1", "Lorenese Vestibule", "entry", 2, 0.5, 20),           # 7,587 px
    ("A2", "Western Corridor", "corridor", 3, 0.5, 200),         # 129,817 px (transit)
    ("A3", "Medieval Painting", "13th-14th", 4, 1.0, 51),        # 19,708 px
    # Giotto loop: A4 -> A5 -> A6 -> A7 -> A4
    ("A4", "Giotto & Cimabue", "13th-14th", 8, 2.5, 65),         # 25,380 px
    ("A5", "Lorenzetti & Simone Martini", "13th-14th", 6, 1.5, 17),  # 6,528 px
    ("A6", "14th Century", "13th-14th", 4, 1.0, 24),             # 9,396 px
    ("A7", "Lorenzo Monaco & Gentile da Fabriano", "early_renaissance", 7, 2.0, 35),  # 13,694 px
    # Sequential off A4
    ("A8", "Masaccio & Beato Angelico", "early_renaissance", 7, 2.0, 26),  # 9,999 px
    ("A9", "Uccello, F. Lippi, Piero della Francesca", "early_renaissance", 8, 2.5, 52),  # 20,115 px
    # Botticelli forced chain: A10 -> A11 -> A12 -> A13
    ("A10", "Pollaiolo", "early_renaissance", 5, 1.5, 22),       # 8,712 px
    ("A11", "Botticelli - The Spring", "early_renaissance", 10, 5.0, 55),  # 42,347 px / 2
    ("A12", "Botticelli - Venus", "early_renaissance", 10, 5.0, 55),       # 42,347 px / 2
    ("A13", "Hugo van der Goes", "early_renaissance", 5, 1.0, 43),         # 16,860 px
    ("A14", "Uffizi Maps Terrace", "early_renaissance", 3, 1.0, 14),       # 5,245 px
    ("A15", "Room of Mathematics", "early_renaissance", 2, 0.5, 17),       # 6,547 px
    # Tribune (Medici Venus)
    ("A16", "Tribune (Medici Venus)", "special", 8, 3.0, 37),    # 14,182 px (octagonal)
    # Forced chain: A17 -> A18 -> A19 -> A20 -> A21
    ("A17", "15th Century Siena", "renaissance_other", 4, 1.0, 20),        # 7,740 px
    ("A18", "Mantegna, Bellini, Antonello da Messina", "renaissance_other", 7, 2.0, 25),  # 9,675 px
    ("A19", "15th Century Veneto", "renaissance_other", 4, 1.0, 20),       # 7,740 px
    ("A20", "15th Century Emilia Romagna", "renaissance_other", 3, 0.8, 26),  # 9,933 px
    ("A21", "15th Century Lombardia", "renaissance_other", 3, 0.8, 22),    # 8,651 px
    ("A22", "Boudoir of Miniatures", "renaissance_other", 3, 1.0, 14),     # 5,457 px
    # Corridors
    ("A23", "Southern Corridor", "corridor", 3, 0.5, 53),        # 34,325 px (transit)
    ("A24", "Eastern Corridor", "corridor", 3, 0.5, 240),        # 154,384 px (transit)
    # U-loop: A25 -> A26 -> A27 -> A28 -> A29 -> (A30 dead end) -> A31 -> A32 -> A33 -> A34
    ("A25", "Domenico Ghirlandaio", "high_renaissance", 6, 1.5, 39),       # 15,200 px
    ("A26", "Cosimo Rosselli", "high_renaissance", 4, 1.0, 30),            # 11,520 px
    ("A27", "Pietro Perugino", "high_renaissance", 7, 2.0, 31),            # 11,926 px
    ("A28", "Filippino Lippi & Piero di Cosimo", "high_renaissance", 6, 1.5, 36),  # 13,936 px
    ("A29", "Lorenzo di Credi", "high_renaissance", 4, 1.0, 12),           # 4,818 px
    ("A30", "Doryphoros", "high_renaissance", 3, 0.8, 10),                 # 4,025 px
    ("A31", "Luca Signorelli", "high_renaissance", 5, 1.2, 14),            # 5,341 px
    ("A32", "Luca Signorelli II", "high_renaissance", 4, 1.0, 22),         # 8,424 px
    ("A33", "Greek Portrait Sculptures", "high_renaissance", 3, 0.8, 22),  # 8,589 px
    ("A34", "Antique & Garden of San Marco", "high_renaissance", 4, 1.0, 13),  # 5,029 px
    # Leonardo-Michelangelo group (connected via A36)
    ("A35", "Leonardo da Vinci", "high_renaissance", 9, 4.0, 52),          # 20,297 px
    ("A36", "Hall of Ancient Inscriptions", "high_renaissance", 3, 0.8, 45),  # 17,474 px
    ("A37", "Unnamed Room", "high_renaissance", 2, 0.5, 22),               # 8,501 px
    ("A38", "Raphael & Michelangelo", "high_renaissance", 9, 4.0, 53),     # 20,743 px
    # Dead-end branches off Eastern Corridor
    ("A39", "Niobe Room", "high_renaissance", 5, 1.5, 93),                 # 36,224 px
    ("A40", "Hermaphrodite", "high_renaissance", 4, 1.0, 14),              # 5,532 px
    ("A42", "Dürer & Transalpine Renaissance", "high_renaissance", 6, 1.5, 34),  # 13,060 px
]

FIRST_FLOOR_B_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    ("B1", "Contini Bonacossi Corridor", "contini_bonacossi", 2, 0.3, 16),   # 6,220 px
    ("B2", "Sassetta & Andrea del Castagno", "contini_bonacossi", 5, 1.2, 20),  # 7,627 px
    ("B3", "Gold Ground Paintings", "contini_bonacossi", 4, 1.0, 13),       # 4,904 px
    ("B4", "Giovanni Bellini", "contini_bonacossi", 6, 1.5, 21),            # 8,143 px
    ("B5", "Bramantino", "contini_bonacossi", 4, 1.0, 19),                  # 7,410 px
    ("B6", "Furniture", "contini_bonacossi", 3, 0.8, 17),                   # 6,674 px
    ("B7", "Majolica", "contini_bonacossi", 3, 0.8, 16),                    # 6,068 px
    ("B8", "Bernini", "contini_bonacossi", 7, 2.0, 39),                     # 15,090 px
]

FIRST_FLOOR_C_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    ("C1", "Self-Portraits: Origins to 17th Century", "self_portraits", 4, 1.0, 85),  # 33,059 px
    ("C2", "Water and Light", "self_portraits", 3, 0.8, 15),                # 5,634 px
    ("C3", "17th Century Self-Portraits", "self_portraits", 3, 0.8, 24),    # 9,139 px
    ("C4", "Showing the Artwork", "self_portraits", 3, 0.8, 22),            # 8,701 px
    ("C5", "Early 18th Century", "self_portraits", 3, 0.8, 21),             # 8,023 px
    ("C6", "Late 18th Century", "self_portraits", 3, 0.8, 38),              # 14,690 px
    ("C7", "Works on Paper I", "self_portraits", 3, 0.8, 25),               # 9,672 px
    ("C8", "Works on Paper II", "self_portraits", 3, 0.8, 29),              # 11,370 px
    ("C9", "19th Century Self-Portraits", "self_portraits", 4, 1.0, 19),    # 7,232 px
    ("C10", "Early 20th Century", "self_portraits", 4, 1.0, 37),            # 14,431 px
    ("C11", "From the War to Our Days", "self_portraits", 4, 1.0, 37),      # 14,336 px
    ("C12", "Contemporary Self-Portraits", "self_portraits", 4, 1.0, 40),   # 15,522 px
]

FIRST_FLOOR_D_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    # 16th century galleries, plus three corridor-like passages (D5, D9, D27).
    ("D1", "The 16th Century in Lombardia", "16th_century", 4, 1.0, 46),    # 17,744 px
    ("D2", "Dosso Dossi and His Circle", "16th_century", 5, 1.2, 49),       # 18,884 px
    ("D3", "Work in Progress", "16th_century", 1, 0.2, 10),                 # 3,198 px (closed)
    ("D4", "Correggio and Parmigianino", "16th_century", 6, 1.5, 21),       # 8,308 px
    ("D5", "Painters from Ferrara", "16th_century", 3, 0.5, 30),            # 11,727 px
    ("D6", "The 16th Century in Bologna", "16th_century", 4, 1.0, 25),      # 9,744 px
    ("D7", "Sebastiano del Piombo and the Influence of Michelangelo", "16th_century", 5, 1.2, 21),  # 8,200 px
    ("D8", "Daniele da Volterra, Francesco Salviati", "16th_century", 4, 1.0, 23),  # 8,800 px
    ("D9", "Andrea del Sarto and Pontormo", "16th_century", 3, 0.5, 45),    # 17,324 px
    ("D10", "Work in Progress", "16th_century", 1, 0.2, 22),                # 8,712 px (closed)
    ("D11", "Work in Progress", "16th_century", 1, 0.2, 29),                # 11,286 px (closed)
    ("D12", "Pontormo, Rosso Fiorentino", "16th_century", 7, 2.0, 21),      # 8,118 px
    ("D13", "Bachiacca, Portraiture in Florence", "16th_century", 4, 1.0, 41),  # 16,016 px
    ("D14", "Bronzino", "16th_century", 6, 1.5, 29),                        # 11,088 px
    ("D15", "Room of the Dynasties", "16th_century", 5, 1.2, 94),           # 36,624 px
    ("D16", "Room of the Pillar", "16th_century", 3, 0.8, 66),              # 25,500 px
    ("D17", "Classic Tradition", "16th_century", 4, 1.0, 22),               # 8,664 px
    ("D18", "Room of the Counter-Reformation", "16th_century", 4, 1.0, 21), # 8,103 px
    ("D19", "Vasari Corridor Entrance", "16th_century", 2, 0.3, 22),        # 8,520 px (passage)
    ("D20", "Venetian Chapel", "16th_century", 4, 1.0, 33),                 # 12,720 px
    ("D21", "El Greco", "16th_century", 6, 1.5, 23),                        # 8,850 px
    ("D22", "Antechamber of Venus", "16th_century", 4, 1.0, 28),            # 10,868 px
    ("D23", "Titian, Venus of Urbino", "16th_century", 9, 3.5, 43),         # 16,733 px
    ("D24", "Naturalismo Veneto", "16th_century", 4, 1.0, 27),              # 10,626 px
    ("D25", "Tintoretto", "16th_century", 6, 1.5, 43),                      # 16,564 px
    ("D26", "Veronese", "16th_century", 6, 1.5, 53),                        # 20,496 px
    ("D27", "Veronese Corridor", "16th_century", 3, 0.5, 120),              # 46,787 px
    ("D28", "Verone", "16th_century", 4, 1.0, 30),                          # no pixel data
]

FIRST_FLOOR_E_ROOMS: List[Tuple[str, str, str, int, float, int]] = [
    # E1-E3 and E8 removed (cancelled in current floor plan).
    # Chain: D28 -> E4 -> E5 -> E6 -> E7 -> EXIT (Magliabecchiana).
    ("E4", "Caravaggio The Medusa", "17th_century", 9, 3.5, 30),
    ("E5", "Caravaggio Il Bacco", "17th_century", 8, 3.0, 25),
    ("E6", "Painting by Candlelight", "17th_century", 5, 1.2, 25),
    ("E7", "Rembrandt, Rubens, Van Dyck", "17th_century", 7, 2.0, 30),
]

SPECIAL_NODES: List[Tuple[str, str, str, int, float, int]] = [
    ("ENTRY", "Museum Entrance (Gran Ducale)", "entry_gate", 1, 0.0, 120),
    ("PANORAMIC_TERRACE", "Panoramic Terrace", "terrace", 3, 1.0, 60),
    ("LANZI_STAIRCASE", "Lanzi Staircase", "staircase", 1, 0.0, 40),
    ("BUONTALENTI_STAIRCASE", "Buontalenti Staircase", "staircase", 1, 0.0, 40),
    ("EXIT", "Museum Exit", "exit_gate", 1, 0.0, 120),
]


# =============================================================================
# 6. Derived room indexes (ROOM_DATA, ROOM_IDS, ROOM_TO_IDX, IDX_TO_ROOM)
# =============================================================================
ALL_ROOM_ROWS = (
    SECOND_FLOOR_ROOMS
    + FIRST_FLOOR_B_ROOMS
    + FIRST_FLOOR_C_ROOMS
    + FIRST_FLOOR_D_ROOMS
    + FIRST_FLOOR_E_ROOMS
    + SPECIAL_NODES
)

# Dict keyed by room_id with all the column attributes.
ROOM_DATA: Dict[str, Dict] = {
    row[0]: {
        "name": row[1],
        "section": row[2],
        "importance": float(row[3]),
        "magnetism": float(row[4]),
        "capacity": float(row[5]),
    }
    for row in ALL_ROOM_ROWS
}

# Ordered list of room IDs and bijective index maps. Every component of
# the system (graph, observations, density matrix) uses this exact order.
ROOM_IDS: List[str] = [row[0] for row in ALL_ROOM_ROWS]
N_ROOMS = len(ROOM_IDS)
ROOM_TO_IDX = {room: i for i, room in enumerate(ROOM_IDS)}
IDX_TO_ROOM = {i: room for room, i in ROOM_TO_IDX.items()}


# =============================================================================
# 7. Graph topology (119 edges between rooms)
# =============================================================================
# Topology mapped from the official 2023 Uffizi floor plan [MAP].
# All edges are undirected (visitors can move in either direction).
# The list below is the single source of truth for which rooms are
# adjacent. museum_graph.py uses it to construct the NetworkX graph.

EDGES: List[Tuple[str, str]] = [
    # ===== SECOND FLOOR (A block) =====
    # Entry
    ("ENTRY", "A1"),
    ("A1", "A2"),      # vestibule to Western Corridor
    ("A1", "A3"),      # vestibule to Medieval Painting (right turn)
    # Giotto loop: A4 <-> A5 <-> A6 <-> A7 <-> A4
    ("A2", "A4"),      # corridor to Giotto (first entrance on left)
    ("A4", "A5"),
    ("A5", "A6"),
    ("A6", "A7"),
    ("A7", "A4"),      # completes the loop
    # Sequential off A4: A4 -> A8 -> A9
    ("A4", "A8"),
    ("A8", "A9"),
    ("A9", "A2"),      # Piero della Francesca back to corridor
    # Botticelli forced chain
    ("A9", "A10"),
    ("A10", "A11"),    # enter Botticelli Spring
    ("A11", "A12"),    # Spring to Venus
    ("A12", "A13"),    # exit Botticelli
    # Post-Botticelli
    ("A13", "A2"),     # Hugo to corridor
    ("A13", "A14"),    # Maps Terrace (dead end)
    ("A13", "A15"),    # Room of Mathematics
    ("A15", "A2"),     # Mathematics to corridor
    # Tribune
    ("A15", "A16"),    # Mathematics to Tribune
    ("A16", "A2"),     # Tribune to corridor
    # Forced chain A17-A21 (no corridor exits from A18-A21)
    ("A16", "A17"),
    ("A17", "A2"),     # last corridor exit before the chain
    ("A17", "A18"),    # into the forced chain
    ("A18", "A19"),
    ("A19", "A20"),
    ("A20", "A21"),
    ("A21", "A2"),     # exit chain to corridor
    ("A21", "A22"),    # Boudoir of Miniatures (dead end)
    # Southern Corridor
    ("A21", "A23"),    # to Southern Corridor
    ("A23", "A24"),    # Southern to Eastern Corridor
    # U-loop A25-A34 (no cross-connections between rows)
    ("A24", "A25"),    # Eastern Corridor to Ghirlandaio
    ("A25", "A26"),
    ("A26", "A27"),
    ("A27", "A28"),
    ("A28", "A29"),
    ("A29", "A30"),    # Doryphoros (dead end)
    ("A29", "A31"),    # the turn between rows
    ("A31", "A32"),
    ("A32", "A33"),
    ("A33", "A34"),
    ("A34", "A24"),    # back to Eastern Corridor
    # Leonardo-Michelangelo group
    ("A24", "A35"),    # corridor to Leonardo
    ("A35", "A36"),    # Leonardo to Hall of Inscriptions
    ("A36", "A24"),    # Hall back to corridor (direct)
    ("A36", "A37"),    # Hall to unnamed room
    ("A36", "A38"),    # Hall to Raphael/Michelangelo
    ("A38", "A24"),    # Raphael back to corridor
    # Dead-end branches off Eastern Corridor
    ("A24", "A39"),    # Niobe Room
    ("A24", "A40"),    # Hermaphrodite
    ("A24", "A42"),    # Dürer
    # Panoramic Terrace and Lanzi Staircase
    ("A24", "PANORAMIC_TERRACE"),
    ("PANORAMIC_TERRACE", "LANZI_STAIRCASE"),
    # Buontalenti Staircase (connects 2nd floor A36 to 1st floor D9)
    ("A36", "BUONTALENTI_STAIRCASE"),
    ("BUONTALENTI_STAIRCASE", "D9"),

    # ===== FIRST FLOOR: B block (Contini Bonacossi) =====
    # All rooms branch off B1 corridor. U-shaped visiting pattern.
    ("LANZI_STAIRCASE", "B1"),
    ("B1", "B2"),      # left side
    ("B1", "B4"),
    ("B1", "B6"),
    ("B1", "B8"),      # turnaround
    ("B1", "B7"),      # right side (return)
    ("B1", "B5"),
    ("B1", "B3"),

    # ===== FIRST FLOOR: C block (Self-Portraits) =====
    # Forced chain with optional C7-C8 branch.
    ("LANZI_STAIRCASE", "C1"),
    ("C1", "C2"),
    ("C2", "C3"),
    ("C3", "C4"),
    ("C4", "C5"),
    ("C5", "C6"),
    ("C6", "C7"),      # optional branch
    ("C7", "C8"),      # dead end
    ("C6", "C9"),      # continue chain
    ("C9", "C10"),
    ("C10", "C11"),
    ("C11", "C12"),
    ("C12", "D1"),     # into D block

    # ===== FIRST FLOOR: D block (16th-17th century) =====
    # Corridor spine: D1 -> D6 -> D13 -> D14 -> D15 -> D26 -> D28
    ("D1", "D6"),      # corridor
    ("D6", "D13"),
    ("D13", "D14"),
    ("D14", "D15"),
    ("D15", "D26"),
    ("D26", "D28"),
    # Branch: D1 -> D2 -> D3 (dead end), D2 -> D4
    ("D1", "D2"),
    ("D2", "D3"),      # dead end
    ("D2", "D4"),
    # D4 -> D5
    ("D4", "D5"),
    # D5 -> D6 (back to corridor) or D5 -> D7
    ("D5", "D6"),
    ("D5", "D7"),
    # D7 -> D8 (dead end), D7 -> D9
    ("D7", "D8"),      # dead end
    ("D7", "D9"),      # Andrea del Sarto / Buontalenti staircase hub
    # D9 connections
    ("D9", "D13"),     # back to corridor
    ("D9", "D12"),
    ("D9", "D10"),
    # D10 branches
    ("D10", "D11"),    # dead end
    ("D10", "D12"),    # also accessible from D9
    ("D10", "D16"),    # perpendicular to corridor
    # D15 -> D16 (also accessible from D10)
    ("D15", "D16"),
    # D16 branches
    ("D16", "D17"),
    ("D16", "D18"),
    ("D17", "D18"),    # tiny room connection between D17 and D18
    # D16 -> D19, D15 -> D19
    ("D16", "D19"),
    ("D15", "D19"),
    # D19 branches
    ("D19", "D20"),
    ("D19", "D21"),
    ("D20", "D21"),    # loop: D19 -> D20 -> D21
    # D21 connections
    ("D21", "D27"),    # to Veronese Corridor
    ("D27", "D26"),    # connects to corridor
    ("D21", "D22"),
    # D22-D23-D24 triangle
    ("D22", "D23"),    # Titian
    ("D22", "D24"),
    ("D23", "D24"),
    # D24 -> D25 -> D26
    ("D24", "D25"),
    ("D25", "D26"),

    # ===== FIRST FLOOR: E block (Caravaggio) =====
    # Corridor runs from D28 past 4 gallery rooms to EXIT.
    # E4-E7 are side rooms (enter-and-exit spurs) off the corridor.
    # Visitors walk the corridor, popping into rooms as they choose.
    ("D28", "E4"),      # corridor reaches E4 first
    ("D28", "E5"),      # corridor continues past E5
    ("E4", "E5"),       # adjacent along corridor
    ("E5", "E6"),       # adjacent along corridor
    ("E6", "E7"),       # adjacent along corridor
    ("E7", "EXIT"),     # corridor ends at Magliabecchiana staircase
]


# =============================================================================
# 8. Routes and named room sets
# =============================================================================
# Recommended visit route (the "Standard Path" suggested to most visitors):
# enter, walk through the famous rooms in chronological order, then exit.
# Used by the default-path baseline policy and as the "route bias" target
# inside the simulator.
RECOMMENDED_ROUTE = [
    "ENTRY", "A1", "A2",
    "A4", "A5", "A6", "A7",
    "A8", "A9",
    "A10", "A11", "A12", "A13",
    "A15", "A16",
    "A17", "A18", "A19", "A20", "A21",
    "A23", "A24",
    "A25", "A26", "A27", "A28", "A29", "A31", "A32", "A33", "A34",
    "A35", "A36", "A38",
    "A39",
    "PANORAMIC_TERRACE", "LANZI_STAIRCASE",
    "D1", "D6", "D13", "D14", "D15",
    "D26", "D28",
    "E4", "E5", "E7",
    "EXIT",
]

# Vasari narration route: an alternative chronological itinerary that
# highlights Giorgio Vasari's lives of the artists, weaving through the
# B and C blocks before joining the main D-block sequence.
VASARI_ROUTE = [
    "ENTRY", "A1", "A2",
    "A3", "A4",
    "A8", "A9", "A11", "A12",
    "A15", "A16",
    "A24", "A35", "A36", "A38",
    "PANORAMIC_TERRACE", "LANZI_STAIRCASE",
    "B1", "B2", "B4", "B8",
    "C1", "C6", "C12",
    "D1", "D6", "D13", "D14", "D23", "D25", "D26", "D28",
    "E4", "E5", "E7",
    "EXIT",
]

# Magnet rooms that the Type B "checkbox tourist" profile concentrates on.
# These are the canonical "must-see" works that appear in every Uffizi
# guidebook and top-10 list. [assumption]
TYPE_B_MAGNET_ROOMS = {"A11", "A12", "A35", "A38", "A16", "E4", "E5"}

# Rooms with windows or open courtyards. Affect the weather-routing
# intervention (rainy weather discourages outdoor rooms).
OUTDOOR_ROOMS = {"PANORAMIC_TERRACE"}

# West-wing rooms. Used by the staggered-wing-opening intervention.
WEST_WING_ROOMS = {"A24", "A25", "A26", "A27", "A28", "A29", "A30", "A31",
                    "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39",
                    "A40", "A42"}

# Corridor junctions where real-time crowd displays ("kiosks") provide
# full museum density information. Placed at decision points where
# visitors choose between routes. [assumption]
KIOSK_ROOMS = {"A2", "A15", "A24", "D9"}


# =============================================================================
# 9. Reward shaping: experience quality weights
# =============================================================================
# Beyond the absence of congestion, the project tracks four "experience
# quality" components that describe what makes a visit memorable. [assumption]
INTIMACY_THRESHOLD = 15          # max people in room for intimacy bonus
INTIMACY_BONUS_WEIGHT = 2.0
NARRATIVE_COHERENCE_WEIGHT = 1.5  # bonus for following a coherent trail
SURPRISE_WEIGHT = 1.0             # bonus for discovering rooms not on checklist
ENGAGEMENT_DEPTH_WEIGHT = 1.5     # bonus for long uncrowded dwell


# =============================================================================
# 10. Extended parameters
# =============================================================================
# The constants below configure features that are NOT exercised by the
# value-based baseline. They are kept here because the simulator code
# references them when intervention flags are activated. A reader
# focused on the basic visitor flow can skip this section.

# Dynamic pricing: price multiplier per time window. [assumption]
PRICE_WINDOWS = [
    (0, 90, 0.50), (90, 165, 0.75), (165, 285, 1.00),
    (285, 405, 1.25), (405, 555, 0.85),
]
MEAN_WILLINGNESS_TO_PAY = 1.0
WTP_SPREAD = 0.4

# Tour groups
TOUR_GROUP_FRACTION = 0.15
TOUR_GROUP_SIZE_DEFAULT = 30
TOUR_GROUP_SIZE_CAPPED = 15

# Temporary exhibits
TEMPORARY_EXHIBIT_ROOMS = ["A20", "A21", "A30", "A33"]
TEMPORARY_EXHIBIT_IMPORTANCE_BOOST = 7.0

# Evening session
EVENING_SESSION_START = 630
EVENING_SESSION_DURATION = 120
EVENING_SESSION_CAPACITY = 200

# Photography ban
PHOTO_BAN_ROOMS = {"A11", "A12"}
PHOTO_BAN_MAGNETISM_FACTOR = 0.6

# Adaptive audio guide
AUDIO_GUIDE_ADOPTION = 0.60
AUDIO_GUIDE_CROWD_SENSITIVITY = 0.8

# Queue-to-content
QUEUE_TO_CONTENT_DENSITY_THRESHOLD = 0.7
QUEUE_TO_CONTENT_BUFFER_ROOM = "A10"
QUEUE_TO_CONTENT_MAGNETISM_BOOST = 3.0

# Crowd forecast and micro-events
CROWD_FORECAST_RESPONSE_RATE = 0.3
MICRO_EVENT_ROOMS = ["A18", "A27", "A39"]
MICRO_EVENT_PEAK_WINDOW = (105, 285)
MICRO_EVENT_IMPORTANCE_BOOST = 4.0

# Room nobody knows, conservation theater, seating, sound design, photo spots
ROOM_NOBODY_KNOWS_CANDIDATES = ["A15", "A17", "A19", "A20", "A21", "A22",
                                 "A29", "A30", "A32", "A33"]
ROOM_NOBODY_KNOWS_IMPORTANCE_BOOST = 5.0
CONSERVATION_THEATER_MAGNETISM_BOOST = 4.0
SEATING_BOOST_ROOMS = ["A5", "A7", "A18", "A27", "A39"]
SEATING_BOOST_FACTOR = 1.5
SEATING_REMOVE_ROOMS = {"A11", "A12"}
SEATING_REMOVE_FACTOR = 0.7
SOUND_DESIGN_ROOMS = ["A5", "A7", "A9", "A18", "A28"]
SOUND_DESIGN_IMPORTANCE_BOOST = 3.0
SOCIAL_MEDIA_IMPORTANCE_BOOST = 5.0
COURTYARD_ABSORPTION_FRACTION = 0.08

# Buffer cascade and magnet windows
BUFFER_CASCADE_ROOMS = ["A9", "A10", "A13"]
BUFFER_CASCADE_MAGNETISM_BOOST = 2.5
ALL_MAGNET_WINDOW_ROOMS = {"A11", "A12", "A35", "A38", "E4"}
MAGNET_WINDOW_DURATION = 30
STAGGERED_WEST_DELAY = 60
TREASURE_HUNT_ROOMS = ["A6", "A15", "A19", "A22", "A30", "A34"]

# Last-hour locals, lunch free entry, resident pass
LAST_HOUR_LOCALS_WINDOW = (450, 555)
LAST_HOUR_LOCALS_EXTRA_VISITORS = 300
LAST_HOUR_LOCALS_PRICE = 5.0
LAST_HOUR_LOCALS_BOOST = 2.0
LUNCH_FREE_WINDOW = (210, 270)
LUNCH_FREE_ARRIVAL_BOOST = 1.5
RESIDENT_PASS_DAILY_VISITORS = 100
TWO_VISIT_TICKET_FRACTION = 0.15
BREATHING_PAUSE_START = 180
BREATHING_PAUSE_DURATION = 30

# Pricing innovations
GROUP_BOOKING_AUCTION_PRICE = 800.0
CROSS_VENUE_PASS_BOOST = 1.2
CROSS_VENUE_AFTERNOON_REDUCTION = 0.7
TIME_SPENT_REBATE_HOURS = 2.0
TIME_SPENT_REBATE_AMOUNT = 5.0
OCCUPANCY_PRICE_LOW = 0.85
OCCUPANCY_PRICE_HIGH = 1.20

# Behavioral nudges
ACHIEVEMENT_IMPORTANCE_BOOST = 2.0
ACHIEVEMENT_UNDERUSED_ROOMS = ["A6", "A15", "A17", "A19", "A20", "A21",
                                "A22", "A29", "A30", "A32", "A33"]
COUNTERFLOW_REVERSAL_BOOST = 0.3
QUIET_HOURS_WINDOW = (0, 60)
QUIET_HOURS_GROUP_BAN = True
PHOTO_SPOT_ROOMS = ["A16", "A39", "A9"]
PHOTO_SPOT_IMPORTANCE_BOOST = 4.0

# Content and experience
PAINTING_TALKS_ROOMS = ["A4", "A9", "A11", "A35", "E4"]
PAINTING_TALKS_IMPORTANCE_BOOST = 3.5
COMPARATIVE_DISPLAY_ROOMS = ["A5", "A7", "A28", "A31"]
COMPARATIVE_DISPLAY_IMPORTANCE_BOOST = 2.5
TACTILE_STATION_MAGNETISM_BOOST = 3.5
THEMED_WEEKS = {
    "caravaggio": ["E4", "E5"],
    "women_of_uffizi": ["A5", "A9", "A12", "D12"],
    "renaissance_masters": ["A4", "A8", "A11", "A35", "A38"],
}
THEMED_WEEK_IMPORTANCE_BOOST = 3.0
ARTIST_RESIDENCE_MAGNETISM_BOOST = 4.0

# Real-time data-driven adjustments
REALTIME_BOOST_DENSITY_THRESHOLD = 0.2
REALTIME_BOOST_FACTOR = 2.0
LIGHTING_OVERCROWDED_FACTOR = 0.8
LIGHTING_EMPTY_FACTOR = 1.3
LIGHTING_DENSITY_THRESHOLD_HIGH = 1.2
LIGHTING_DENSITY_THRESHOLD_LOW = 0.2
LIGHTING_DENSITY_THRESHOLD = 0.7
ADAPTIVE_TRAIL_REACTIVITY = 0.5
PREDICTIVE_ROUTING_LEAD_MINUTES = 20
SOCIAL_PROOF_THRESHOLD = 1.0
SOCIAL_PROOF_PENALTY = 0.7

# Other miscellaneous
PROGRESS_BAR_UNVISITED_BOOST = 1.5
WALK_IN_DETERRENCE = 0.5
GROUP_SURCHARGE_REFERENCE_PRICE = 5.0
WEATHER_RAIN_IMPORTANCE_PENALTY = 0.5
WEATHER_SUN_IMPORTANCE_BONUS = 1.2
GROUP_SPLITTING_FRACTION = 0.10
SMART_DEFAULT_BOTTICELLI_WINDOW = 30

# Revenue model (used by the ticket-pricing logic; not active in
# the value-based baseline).
TICKET_PRICE_SCHEDULE = [
    (0, 90, 15.0), (90, 165, 20.0), (165, 285, 25.0),
    (285, 405, 30.0), (405, 555, 18.0),
]
WALK_IN_TICKET_PRICE = 35.0
FREE_VISITOR_FRACTION = 0.30
REDUCED_VISITOR_FRACTION = 0.10
REDUCED_TICKET_PRICE = 2.0

# Themed hidden-gem trails for redirecting demand. [assumption]
HIDDEN_GEM_TRAILS = {
    "colors_of_florence": ["A5", "A7", "A9", "A27", "A28", "D12"],
    "sacred_and_profane": ["A4", "A8", "A18", "A38"],
    "faces_of_power": ["A16", "A25", "D14", "D15", "D23", "E7"],
}

# Vasari adoption fraction (used by the vasari_narration intervention).
VASARI_ADOPTION = 0.25


# =============================================================================
# 11. Utility functions
# =============================================================================

def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return the project's seeded NumPy random generator.

    Centralizing RNG construction here ensures every component uses
    the same default seed and the same generator type (PCG64).
    """

    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)


def normalize_weights(weights: Iterable[float]) -> np.ndarray:
    """Convert an iterable of non-negative weights into a probability vector.

    Returns a NumPy array that sums to 1. If all weights are zero or
    the iterable is empty, returns a uniform distribution.
    """

    w = np.asarray(list(weights), dtype=float)
    if len(w) == 0:
        return w
    s = w.sum()
    if s <= 0:
        return np.full_like(w, 1.0 / len(w))
    return w / s


# Visit duration distribution: log-normal targets calibrated so that
# the early-day mean is ~120 minutes and late-day mean is ~65 minutes
# (visitors arriving later have less time before closing). [assumption]
def sample_visit_duration(entry_slot: int, rng: np.random.Generator) -> int:
    """Sample one visitor's planned visit duration (in minutes).

    The mean target depends on the entry slot: visitors who arrive
    early plan longer visits, visitors who arrive late plan shorter
    visits (they have less time before closing).

    Returns a minimum of 20 minutes (no one comes for less than that).
    """

    if entry_slot < 7:
        target = 120.0
    elif entry_slot < 19:
        target = 100.0
    elif entry_slot < 31:
        target = 80.0
    else:
        target = 65.0
    # Log-normal with mode at `target`. sigma=0.35 gives a moderate spread.
    sigma = 0.35
    mu = np.log(target) - 0.5 * sigma ** 2
    duration = int(np.exp(rng.normal(mu, sigma)))
    return max(20, duration)


def capacity_math_check(daily_visitors: int = DAILY_VISITORS_NORMAL) -> Dict[str, float]:
    """Sanity check: does daily volume fit under the museum capacity?

    Computes the average steady-state occupancy from daily visitors and
    the slot-weighted average visit duration. Compares it to the fire-
    safety cap of 900.

    Average duration is the mean of the per-slot duration targets used
    by `sample_visit_duration` (120/100/80/65 minutes across the four
    quartiles of the entry window). This yields ~91 minutes, well below
    the threshold that would push occupancy over 900.
    """

    # Per-slot duration targets, identical to sample_visit_duration above.
    slot_means = [
        120 if i < 7 else 100 if i < 19 else 80 if i < 31 else 65
        for i in range(N_ENTRY_SLOTS)
    ]
    avg_duration = float(np.mean(slot_means))
    visitor_minutes = daily_visitors * avg_duration
    # Divide by the full open window (615 min), not just the entry window:
    # visitors who entered early are still inside after last entry closes.
    avg_occupancy = visitor_minutes / MUSEUM_OPEN_MINUTES
    return {
        "daily_visitors": float(daily_visitors),
        "avg_duration_min": avg_duration,
        "visitor_minutes": float(visitor_minutes),
        "avg_occupancy": float(avg_occupancy),
        "capacity": float(MAX_MUSEUM_CAPACITY),
        "within_capacity_on_average": float(avg_occupancy < MAX_MUSEUM_CAPACITY),
    }
