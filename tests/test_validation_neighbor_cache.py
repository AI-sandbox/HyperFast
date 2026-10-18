from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import hyperfast.utils as utils


class ValidationNeighborReuseTests(unittest.TestCase):
    def run_fit(self, device, scope, neighbors, classes, reuse, *, nn_bias=True, validation=True,
                validation_rows=7, batch_size=3, duplicate_references=False):
        torch.manual_seed(197)
        torch.set_num_threads(1)
        cfg = SimpleNamespace(device=device, torch_pca=True, clip_data_value=27.,
                              nn_bias=nn_bias, nn_bias_neighbors=neighbors, lr=1e-3)
        X = torch.randn(11, 3)
        y = torch.arange(11) % classes
        if duplicate_references:
            X[1] = X[0]
        X_val = torch.randn(validation_rows, 3) if validation else None
        y_val = torch.arange(validation_rows) % classes if validation else None
        if duplicate_references and validation:
            X_val[0] = X[0]
        rf = nn.Sequential(nn.Linear(3, 8, bias=False), nn.ReLU()).to(device)
        pca = utils.TorchPCA(n_components=4)
        pca.mean_ = torch.zeros(8, device=device)
        pca.components_ = torch.randn(4, 8, device=device)
        pca.n_supported_components_ = 4
        network = [(torch.randn(4, 4, device=device), torch.randn(4, device=device)),
                   (torch.randn(4, 4, device=device), torch.randn(4, device=device)),
                   (torch.randn(4, classes, device=device), torch.randn(classes, device=device))]
        info, calls = {}, []
        original = utils.NN.predict_proba

        def counted(nearest, *args, **kwargs)
            if len(nearest.train_pts) == 6:
                calls.append(len(args[0]))
            return original(nearest, *args, **kwargs)

        control = nullcontext() if reuse else patch.object(utils, '_fixed_validation_neighbors', return_value=None)
        with control, patch.object(utils.NN, 'predict_proba', new=counted):
            parts = utils.fine_tune_main_network(
                cfg, X, y, classes, rf, pca, network, torch.tensor([.3, -.2], device=device),
                device, 8, batch_size, X_val=X_val, y_val=y_val,
                reference_X=X[:6] if validation else None,
                reference_y=y[:6] if validation else None,
                max_epochs=4, patience_epochs=50, validation_metric='log_loss',
                finetuning_dropout=.2, finetuning_parameter_scope=scope,
                loader_generator=torch.Generator().manual_seed(77), training_info=info)
        fitted_rf, fitted_pca, fitted_layers, fitted_bias = parts
        tensors = list(fitted_rf.state_dict().values()) + [fitted_pca.mean_, fitted_pca.components_]
        tensors += [value for layer in fitted_layers for value in layer] + [fitted_bias]
        weights = torch.cat([value.detach().cpu().flatten() for value in tensors])
        return weights, info, len(calls)

    def compare_fits(self, device):
        for scope in ('all', 'network', 'head'):
            for neighbors, classes in ((1, 2), (3, 3)):
                with self.subTest(device=device, scope=scope, neighbors=neighbors, classes=classes):
                    old_weights, old_info, old_calls = self.run_fit(device, scope, neighbors, classes, False)
                    new_weights, new_info, new_calls = self.run_fit(device, scope, neighbors, classes, True)
                    torch.testing.assert_close(old_weights, new_weights, rtol=0, atol=0)
                    for name in ('best_step', 'best_score', 'steps', 'history', 'stop_reason'):
                        self.assertEqual(old_info[name], new_info[name])
                    self.assertLess(new_calls, old_calls)
                    self.assertEqual(new_info['validation_neighbor_reuse'],
                                     'raw_and_hidden' if scope == 'head' else 'raw')

    def test_cpu_weights_checkpoints_and_scores_are_identical(self):
        self.compare_fits('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_weights_checkpoints_and_scores_are_identical(self):
        self.compare_fits('cuda')

    def test_no_neighbor_bias_and_no_validation_keep_original_fits(self):
        for nn_bias, validation in ((False, True), (True, False)):
            with self.subTest(nn_bias=nn_bias, validation=validation):
                old, old_info, _ = self.run_fit('cpu', 'all', 1, 2, False,
                                               nn_bias=nn_bias, validation=validation)
                new, new_info, calls = self.run_fit('cpu', 'all', 1, 2, True,
                                                   nn_bias=nn_bias, validation=validation)
                torch.testing.assert_close(old, new, rtol=0, atol=0)
                self.assertEqual(old_info, new_info)
                self.assertEqual(new_info['validation_neighbor_reuse'], 'none')
                self.assertEqual(calls, 0)

    def test_non_aligned_nn_chunks_and_duplicate_reference_ties(self):
        for scope in ('all', 'head'):
            for neighbors in (1, 3):
                with self.subTest(scope=scope, neighbors=neighbors):
                    options = dict(validation_rows=261, batch_size=129, duplicate_references=True)
                    old, old_info, _ = self.run_fit('cpu', scope, neighbors, 3, False, **options)
                    new, new_info, _ = self.run_fit('cpu', scope, neighbors, 3, True, **options)
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
                    self.assertEqual(old_info['history'], new_info['history'])
                    self.assertEqual(old_info['best_step'], new_info['best_step'])


if __name__ == '__main__':
    unittest.main()
