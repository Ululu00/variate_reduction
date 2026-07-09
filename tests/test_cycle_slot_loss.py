import unittest

import torch

from layers.VariateReduction import VariateReducer


def make_reducer(**overrides):
    config = {
        "reduction_type": "mlp_slot_attention",
        "num_variates": 6,
        "reduced_k": 2,
        "d_model": 4,
        "expansion_type": "slot_learned_linear",
        "compute_aux_losses": False,
        "use_cycle_slot_loss": False,
        "cycle_loss_weight": 0.0,
        "cycle_topr": 0,
        "cycle_topr_multiplier": 1.0,
        "cycle_b_norm": "row_l1",
        "cycle_eps": 1e-8,
    }
    config.update(overrides)
    config.pop("cycle_loss_weight", None)
    return VariateReducer(**config)


class CycleSlotLossTest(unittest.TestCase):
    def test_flag_off_returns_zero_cycle_loss_and_shapes(self):
        torch.manual_seed(11)
        reducer = make_reducer()
        x = torch.randn(3, 6, 4)

        z, cache, _ = reducer.compress(x)
        y = reducer.decode(z, cache=cache)
        cycle_loss = reducer.get_aux_losses()["cycle_slot"]

        self.assertEqual(tuple(z.shape), (3, 2, 4))
        self.assertEqual(tuple(y.shape), (3, 6, 4))
        self.assertTrue(torch.isfinite(cycle_loss).item())
        self.assertEqual(float(cycle_loss.item()), 0.0)

    def test_flag_on_cycle_loss_is_finite_and_nonnegative(self):
        torch.manual_seed(13)
        reducer = make_reducer(use_cycle_slot_loss=True)
        x = torch.randn(4, 6, 4)

        reducer.compress(x)
        cycle_loss = reducer.get_aux_losses()["cycle_slot"]

        self.assertTrue(torch.isfinite(cycle_loss).item())
        self.assertGreaterEqual(float(cycle_loss.item()), 0.0)

    def test_cycle_loss_backward_reaches_attention_and_expansion(self):
        torch.manual_seed(17)
        reducer = make_reducer(use_cycle_slot_loss=True)
        x = torch.randn(5, 6, 4)

        reducer.compress(x)
        loss = reducer.get_aux_losses()["cycle_slot"]
        loss.backward()

        score_grad = sum(
            float(param.grad.detach().abs().sum().item())
            for param in reducer.score_mlp.parameters()
            if param.grad is not None
        )
        expand_grad = float(reducer.slot_expand_linear.weight.grad.detach().abs().sum().item())
        self.assertGreater(score_grad, 0.0)
        self.assertGreater(expand_grad, 0.0)

    def test_zero_expansion_weight_stays_finite(self):
        torch.manual_seed(19)
        reducer = make_reducer(use_cycle_slot_loss=True, export_slot_diagnostics=True)
        with torch.no_grad():
            reducer.slot_expand_linear.weight.zero_()
        x = torch.randn(2, 6, 4)

        reducer.compress(x)
        cycle_loss = reducer.get_aux_losses()["cycle_slot"]
        stats = reducer.get_cycle_slot_stats()

        self.assertTrue(torch.isfinite(cycle_loss).item())
        for key in (
            "cycle_topr_jaccard",
            "cycle_a_effective_support",
            "cycle_b_effective_support",
            "cycle_slot_overlap",
            "w_eff_density",
        ):
            self.assertTrue(torch.isfinite(torch.tensor(float(stats[key]))).item())

    def test_diagnostic_matrices_are_heatmap_ready(self):
        torch.manual_seed(23)
        reducer = make_reducer(use_cycle_slot_loss=True, export_slot_diagnostics=True)
        x = torch.randn(2, 6, 4)

        reducer.compress(x)
        matrices = reducer.get_slot_diagnostic_matrices()

        self.assertEqual(tuple(matrices["A"].shape), (2, 6))
        self.assertEqual(tuple(matrices["B"].shape), (2, 6))
        self.assertEqual(tuple(matrices["W_eff"].shape), (6, 6))
        for matrix in matrices.values():
            self.assertTrue(torch.isfinite(matrix).all().item())


if __name__ == "__main__":
    unittest.main()
