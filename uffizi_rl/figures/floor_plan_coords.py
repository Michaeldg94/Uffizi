"""Hand-extracted room coordinates on the Uffizi floor-plan PNGs.

Coordinates are in normalized [0, 1] image space (origin top-left, y down)
relative to the high-resolution PNGs in outputs/assets/ (floor2 2000x1154,
floor1 2000x1245). Read off a 0.05 calibration grid overlaid on each image.
Used to overlay visitor paths / heatmaps on the actual museum map.
"""
from __future__ import annotations

# 2nd floor (A1-A42 + special nodes) ----------------------------------------
# Reference image: outputs/assets/uffizi_floor2.png
FLOOR2 = {
    "ENTRY":               (0.275, 0.025),
    "GRANDUCAL_STAIRCASE": (0.275, 0.105),
    # Lower row of the 15th-century wing.
    "A1":  (0.265, 0.30),
    "A3":  (0.165, 0.30),
    "A5":  (0.325, 0.30),
    "A4":  (0.40,  0.30),
    "A9":  (0.475, 0.30),
    "A13": (0.65,  0.30),
    "A15": (0.69,  0.30),
    "A17": (0.765, 0.30),
    "A18": (0.795, 0.30),
    "A19": (0.825, 0.30),
    "A20": (0.855, 0.30),
    "A21": (0.885, 0.30),
    # Upper sub-row.
    "A6":  (0.32,  0.225),
    "A7":  (0.385, 0.215),
    "A8":  (0.445, 0.215),
    "A10": (0.49,  0.215),
    "A11": (0.545, 0.25),   # Botticelli Spring (left half of A11-A12 cell)
    "A12": (0.59,  0.25),   # Botticelli Venus (right half)
    "A14": (0.63,  0.215),
    "A16": (0.73,  0.265),  # Tribune
    "A22": (0.945, 0.235),
    # Corridors (U-shape).
    "A2":  (0.50,  0.375),  # Eastern Corridor (horizontal)
    "A23": (0.955, 0.45),   # Southern Corridor (vertical, right edge)
    "A24": (0.50,  0.525),  # Western Corridor (horizontal)
    # Lower 16th-century wing + U-loop.
    "A25": (0.90,  0.665),
    "A26": (0.90,  0.715),
    "A27": (0.90,  0.825),
    "A28": (0.90,  0.895),
    "A29": (0.845, 0.965),
    "A30": (0.88,  0.965),
    "A31": (0.785, 0.965),
    "A32": (0.815, 0.895),
    "A33": (0.815, 0.825),
    "A34": (0.815, 0.665),
    "A35": (0.755, 0.715),  # Leonardo
    "A36": (0.685, 0.715),
    "A37": (0.685, 0.795),
    "A38": (0.605, 0.715),  # Raphael & Michelangelo
    "A39": (0.49,  0.71),   # Niobe
    "A40": (0.325, 0.65),
    "A41": (0.325, 0.715),
    "A42": (0.275, 0.715),
    "PANORAMIC_TERRACE":     (0.10,  0.70),
    "LANZI_STAIRCASE":       (0.215, 0.70),
    "BUONTALENTI_STAIRCASE": (0.665, 0.74),
    "EXIT": (0.175, 0.715),
}

# 1st floor (B/C/D/E rooms) -------------------------------------------------
# Reference image: outputs/assets/uffizi_floor1.png
FLOOR1 = {
    # E-block (Caravaggio & Baroque), top right (teal block).
    "E7": (0.665, 0.245), "E6": (0.715, 0.245), "E5": (0.76, 0.245), "E4": (0.81, 0.245),
    "MAGLIABECHI_STAIRCASE": (0.60, 0.025),
    # C-block (Self-Portraits, green) on the left.
    "C3": (0.06, 0.595), "C2": (0.055, 0.655), "C1": (0.05, 0.79),
    "C4": (0.115, 0.595), "C5": (0.155, 0.595), "C6": (0.215, 0.595),
    "C7": (0.215, 0.665), "C8": (0.215, 0.73),
    "C9": (0.255, 0.595), "C10": (0.30, 0.595), "C11": (0.35, 0.595), "C12": (0.43, 0.595),
    # D-block (16th century, purple), centre-right.
    "D1": (0.475, 0.595), "D6": (0.535, 0.595), "D13": (0.59, 0.595),
    "D14": (0.645, 0.595), "D15": (0.73, 0.595), "D26": (0.825, 0.595),
    "D2": (0.465, 0.685), "D3": (0.465, 0.71), "D4": (0.515, 0.685), "D5": (0.55, 0.685),
    "D7": (0.575, 0.65), "D8": (0.59, 0.685), "D9": (0.625, 0.685),
    "D12": (0.655, 0.655), "D10": (0.655, 0.71), "D11": (0.655, 0.77),
    "D16": (0.73, 0.71), "D17": (0.69, 0.77), "D18": (0.73, 0.77),
    "D19": (0.755, 0.685), "D20": (0.795, 0.685), "D21": (0.79, 0.745),
    "D22": (0.83, 0.685), "D25": (0.875, 0.685), "D24": (0.875, 0.79), "D23": (0.875, 0.86),
    "D27": (0.915, 0.31),  # Verone corridor (vertical strip, right)
    "D28": (0.875, 0.42),  # Verone passage toward the E-block (not labelled on map)
    # B-block (Contini Bonacossi, teal), bottom-left.
    "B2": (0.195, 0.84), "B4": (0.25, 0.84), "B6": (0.29, 0.84),
    "B1": (0.265, 0.91), "B8": (0.32, 0.91),
    "B3": (0.205, 0.93), "B5": (0.25, 0.93), "B7": (0.29, 0.93),
    "LANZI_STAIRCASE": (0.135, 0.78),
    "BUONTALENTI_STAIRCASE": (0.565, 0.83),
}

# Combine for lookup convenience.
ALL_COORDS = {**FLOOR2, **FLOOR1}
FLOOR2_ROOMS = set(FLOOR2.keys())
FLOOR1_ROOMS = set(FLOOR1.keys())

# Corridors run as straight lines along one axis: (orient, fixed_coord).
CORRIDORS = {
    "A2":  ("h", 0.375),  # Eastern Corridor, horizontal
    "A23": ("v", 0.955),  # Southern Corridor, vertical
    "A24": ("h", 0.525),  # Western Corridor, horizontal
}

# Explicit corner waypoints between consecutive corridor rooms so the drawn
# line follows the physical U-shaped corridor instead of cutting diagonally.
EDGE_WAYPOINTS = {
    ("A2", "A23"):  [(0.955, 0.375)],
    ("A23", "A2"):  [(0.955, 0.375)],
    ("A23", "A24"): [(0.955, 0.525)],
    ("A24", "A23"): [(0.955, 0.525)],
    ("A24", "LANZI_STAIRCASE"):     [(0.215, 0.525)],
    ("LANZI_STAIRCASE", "A24"):     [(0.215, 0.525)],
    ("A24", "PANORAMIC_TERRACE"):   [(0.10, 0.525)],
    ("PANORAMIC_TERRACE", "A24"):   [(0.10, 0.525)],
    ("A1", "A2"):  [(0.30, 0.375)],
    ("A2", "A1"):  [(0.30, 0.375)],
}
