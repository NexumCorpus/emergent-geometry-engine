"""Gromov delta-hyperbolicity via the 4-point condition (Constitution sec 2.1).

Convention (SageMath / standard): for every 4-tuple form the three pairwise-distance
sums, sort them M1>=M2>=M3; the tuple's hyperbolicity is (M1-M2)/2. delta(G) is the
max over all 4-tuples. Tree -> 0 (trees are 0-hyperbolic); C_4 -> 1.
"""
import numpy as np
import networkx as nx
import pytest

from geometry.delta import gromov_delta, gromov_delta_from_distances


def test_tree_is_zero_hyperbolic():
    G = nx.balanced_tree(2, 3)  # a binary tree, 15 nodes
    assert abs(gromov_delta(G)) < 1e-12


def test_path_is_zero_hyperbolic():
    G = nx.path_graph(6)
    assert abs(gromov_delta(G)) < 1e-12


def test_complete_graph_is_zero_hyperbolic():
    # All distances 1 => all three sums equal 2 => delta 0.
    G = nx.complete_graph(6)
    assert abs(gromov_delta(G)) < 1e-12


def test_cycle_four_has_delta_one():
    G = nx.cycle_graph(4)
    assert abs(gromov_delta(G) - 1.0) < 1e-12


def test_from_distance_matrix_matches_graph():
    G = nx.cycle_graph(4)
    D = nx.floyd_warshall_numpy(G)
    assert abs(gromov_delta_from_distances(D) - gromov_delta(G)) < 1e-12


def test_delta_is_nonnegative_on_random_graph():
    G = nx.random_regular_graph(3, 12, seed=0)
    assert gromov_delta(G) >= -1e-12
