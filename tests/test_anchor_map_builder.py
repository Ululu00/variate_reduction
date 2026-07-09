import unittest
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_variate_anchor_map import (
    abs_correlation,
    raw_profile_similarity,
    select_tail_coverage_anchors,
)


class AnchorMapBuilderTest(unittest.TestCase):
    def test_raw_profile_similarity_keeps_valid_similarity_matrix(self):
        rng = np.random.default_rng(7)
        base = rng.normal(size=(80, 1)).astype(np.float32)
        train = np.concatenate(
            [
                base,
                base * 0.9 + rng.normal(scale=0.05, size=(80, 1)).astype(np.float32),
                rng.normal(size=(80, 1)).astype(np.float32),
                np.sin(np.arange(80, dtype=np.float32)[:, None] / 4.0),
            ],
            axis=1,
        )

        sim = raw_profile_similarity(train, raw_weight=0.8)

        self.assertEqual(tuple(sim.shape), (4, 4))
        self.assertTrue(np.all(sim >= 0.0))
        self.assertTrue(np.all(sim <= 1.0))
        self.assertTrue(np.allclose(np.diag(sim), 1.0))
        self.assertTrue(np.allclose(sim, sim.T, atol=1e-6))
        self.assertGreater(sim[0, 1], sim[0, 2])

    def test_raw_profile_similarity_weight_one_matches_raw_correlation(self):
        rng = np.random.default_rng(11)
        train = rng.normal(size=(60, 5)).astype(np.float32)

        sim = raw_profile_similarity(train, raw_weight=1.0)

        self.assertTrue(np.allclose(sim, abs_correlation(train), atol=1e-6))

    def test_raw_profile_similarity_rejects_bad_weight(self):
        train = np.ones((16, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            raw_profile_similarity(train, raw_weight=-0.1)
        with self.assertRaises(ValueError):
            raw_profile_similarity(train, raw_weight=1.1)

    def test_tail_coverage_exact_selects_poorly_covered_representative(self):
        corr = np.array(
            [
                [1.0, 0.95, 0.94, 0.35, 0.10],
                [0.95, 1.0, 0.93, 0.36, 0.11],
                [0.94, 0.93, 1.0, 0.34, 0.12],
                [0.35, 0.36, 0.34, 1.0, 0.20],
                [0.10, 0.11, 0.12, 0.20, 1.0],
            ],
            dtype=np.float32,
        )

        anchors = select_tail_coverage_anchors(corr, 2, max_exact_combinations=20)

        self.assertIn(4, anchors.tolist())


if __name__ == "__main__":
    unittest.main()
