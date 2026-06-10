"""Intervention utilities for standalone analysis and extensions.

The core intervention mechanisms (timed entry, dynamic info, congestion
pricing) are implemented directly in CrowdSimulator via its constructor
parameters (timed_entry, dynamic_info, botticelli_slot_cap). Hidden gem
trail effects are controlled via trail_acceptance_prob and
heterogeneity_scale.

The modules in this package provide finer-grained standalone tools for
detailed intervention analysis (e.g., per-visitor slot assignment,
reservation book management, trail load vectors).
"""
