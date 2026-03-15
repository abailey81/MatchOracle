"""Prediction models package.

Implements a 5-layer deep ensemble pipeline: Dixon-Coles statistical model,
13 diverse ML base learners, 4 meta-learners via stacking, isotonic calibration,
and confidence-weighted final output with Monte Carlo validation.
"""
