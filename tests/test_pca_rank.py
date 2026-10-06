import unittest
from types import SimpleNamespace

import numpy as np
import torch

from hyperfast.model import HyperFast
from hyperfast.utils import (
    MainNetworkTrainable, TorchPCA, fine_tune_main_network,
    forward_main_network, svd_flip, transform_data_for_main_network,
)


class PCARankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_repeated_rows_do_not_create_prediction_features(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                rows = torch.randn(5, 16, dtype=dtype)
                padded = rows.repeat_interleave(2, dim=0)
                pca = TorchPCA(n_components=8)
                fitted = pca.fit_transform(padded.clone())
                query = pca.transform(torch.randn(7, 16, dtype=dtype))
                self.assertEqual(pca.n_supported_components_, 4)
                self.assertEqual(fitted.shape, (10, 8))
                self.assertEqual(query.shape, (7, 8))
                self.assertEqual(torch.count_nonzero(fitted[:, 4:]).item(), 0)
                self.assertEqual(torch.count_nonzero(query[:, 4:]).item(), 0)
                self.assertGreater(torch.count_nonzero(query[:, :4]).item(), 0)
                torch.testing.assert_close(fitted, pca.transform(padded.clone()))

    def test_constant_context_has_zero_rank(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                rows = torch.randn(1, 16, dtype=dtype).repeat(10, 1)
                pca = TorchPCA(n_components=8)
                fitted = pca.fit_transform(rows)
                self.assertEqual(pca.n_supported_components_, 0)
                self.assertEqual(torch.count_nonzero(fitted).item(), 0)
                self.assertEqual(torch.count_nonzero(
                    pca.transform(torch.randn(3, 16, dtype=dtype))).item(), 0)

    def test_full_rank_context_preserves_previous_projection(self):
        X = torch.randn(20, 12, dtype=torch.float64)
        centered = X - X.mean(dim=0)
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        U, Vt = svd_flip(U, Vt)
        pca = TorchPCA(n_components=8)
        fitted = pca.fit_transform(X.clone())
        self.assertEqual(pca.n_supported_components_, 8)
        torch.testing.assert_close(pca.components_, Vt[:8], rtol=0, atol=0)
        torch.testing.assert_close(fitted, U[:, :8] * S[:8], rtol=0, atol=0)

    def test_numerically_negligible_directions_are_padded(self):
        X = torch.diag(torch.tensor([1., .5, .2, 1e-9], dtype=torch.float32))
        X = torch.cat((X, -X))
        pca = TorchPCA(n_components=4)
        pca.fit_transform(X)
        self.assertEqual(pca.n_supported_components_, 3)
        self.assertEqual(torch.count_nonzero(pca.components_[3:]).item(), 0)

    def check_network(self, torch_pca, constant=False):
        cfg = SimpleNamespace(n_dims=8, rf_size=16, max_categories=2,
                              torch_pca=torch_pca, clip_data_value=27.6041,
                              hn_n_layers=2, hn_hidden_size=8, main_n_layers=3,
                              nn_bias=False, device='cpu', lr=1e-4)
        network = HyperFast(cfg).eval()
        X = torch.randn(5, 7)
        if constant:
            X[:] = X[0].clone()
        y = torch.arange(5) % 2
        with torch.no_grad():
            parts = network(X.repeat_interleave(2, dim=0), y.repeat_interleave(2), 2)
        rf, pca, layers, bias = parts
        rank = pca.n_supported_components_
        self.assertEqual(rank, 0 if constant else 4)
        query = torch.randn(3, 7)
        projected = transform_data_for_main_network(query, cfg, rf, pca)
        self.assertEqual(projected.shape, (3, 8))
        self.assertEqual(torch.count_nonzero(projected[:, rank:]).item(), 0)
        rf, pca, layers, bias = fine_tune_main_network(
            cfg, X, y, 2, rf, pca, layers, bias, 'cpu', 3, 5)
        projected = transform_data_for_main_network(query, cfg, rf, pca)
        self.assertEqual(torch.count_nonzero(projected[:, rank:]).item(), 0)
        trainer = MainNetworkTrainable(cfg, 2, rf, pca, layers, bias).eval()
        expected, _ = forward_main_network(projected, layers)
        actual = trainer(query)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(np.count_nonzero(
            pca.components_.detach().numpy()[:, :] if torch_pca else pca.components_),
            np.count_nonzero(pca.components_.detach().numpy()[:rank] if torch_pca else pca.components_[:rank]))

    def test_torch_network_keeps_padding_zero_after_fine_tuning(self):
        self.check_network(True)

    def test_sklearn_network_keeps_padding_zero_after_fine_tuning(self):
        self.check_network(False)

    def test_torch_network_with_constant_context(self):
        self.check_network(True, constant=True)

    def test_sklearn_network_with_constant_context(self):
        self.check_network(False, constant=True)


if __name__ == '__main__':
    unittest.main()
