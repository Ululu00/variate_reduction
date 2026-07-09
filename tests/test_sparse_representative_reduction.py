import unittest

import torch

from layers.VariateReduction import VariateReducer


class SparseRepresentativeReductionTest(unittest.TestCase):
    def test_topk_compress_and_expand_shapes_and_sparsity(self):
        reducer = VariateReducer(
            "mlp_sparse_representative",
            num_variates=21,
            reduced_k=6,
            d_model=8,
            sparse_compress_topk=4,
            sparse_expand_topk=2,
            compute_aux_losses=True,
        )
        x = torch.randn(3, 21, 8)
        z, _, _ = reducer.compress(x)
        y = reducer.decode(z)

        self.assertEqual(tuple(z.shape), (3, 6, 8))
        self.assertEqual(tuple(y.shape), (3, 21, 8))
        matrices = reducer.get_weight_matrices()
        self.assertEqual(int((matrices["compress_weight"] > 0).sum().item()), 6 * 4)
        self.assertEqual(int((matrices["expand_weight"] > 0).sum().item()), 21 * 2)
        self.assertTrue(torch.allclose(matrices["compress_weight"].sum(dim=-1), torch.ones(6)))
        self.assertTrue(torch.allclose(matrices["expand_weight"].sum(dim=-1), torch.ones(21)))

    def test_support_overlap_loss_has_gradient_signal(self):
        reducer = VariateReducer(
            "mlp_sparse_representative",
            num_variates=12,
            reduced_k=4,
            d_model=8,
            sparse_compress_topk=3,
            sparse_expand_topk=2,
            compute_aux_losses=True,
        )
        x = torch.randn(2, 12, 8)
        z, _, _ = reducer.compress(x)
        y = reducer.decode(z)
        loss = y.pow(2).mean() + reducer.get_aux_losses()["support_overlap"]
        loss.backward()

        self.assertGreater(float(reducer.compress_linear.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(reducer.expand_linear.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
