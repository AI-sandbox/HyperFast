import unittest
from unittest.mock import patch

import numpy as np
import torch

import hyperfast.hyperfast as api
from hyperfast.utils import MainNetworkTrainable, fine_tune_main_network
from test_bias_finetuning import make_config, make_data, make_network_parts
import test_classifier_finetuning as fixtures


class FineTuningScopeTests(unittest.TestCase):
    def test_cached_frozen_projection_matches_uncached_updates(self):
        X, y = make_data()
        for scope in ('network', 'head'):
            with self.subTest(scope=scope):
                torch.manual_seed(4)
                expected = MainNetworkTrainable(make_config(), 2, *make_network_parts(),
                                                torch.tensor([.7, .9])).train()
                expected.requires_grad_(False)
                layers = expected.layers if scope == 'network' else expected.layers[-1:]
                for layer in layers:
                    layer.requires_grad_(True)
                expected.nn_bias.requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [p for p in expected.parameters() if p.requires_grad], lr=.01, weight_decay=.1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=6e-7)
                loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y),
                    batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(5))
                for _ in range(2):
                    for inputs, targets in loader:
                        optimizer.zero_grad()
                        torch.nn.functional.cross_entropy(expected(inputs, targets), targets).backward()
                        optimizer.step()
                    scheduler.step()
                torch.manual_seed(4)
                parts = fine_tune_main_network(
                    make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
                    'cpu', 4, 3, finetuning_dropout=0, gradient_clip_norm=None,
                    finetuning_parameter_scope=scope, finetuning_weight_decay=.1,
                    loader_generator=torch.Generator().manual_seed(5))
                actual = MainNetworkTrainable(make_config(), 2, *parts)
                for name, value in actual.state_dict().items():
                    torch.testing.assert_close(value, expected.state_dict()[name], rtol=1e-6, atol=1e-7)

    def test_scopes_update_only_the_requested_parameters(self):
        X, y = make_data()
        for scope in ('all', 'network', 'head'):
            with self.subTest(scope=scope):
                torch.manual_seed(42)
                before = MainNetworkTrainable(make_config(), 2, *make_network_parts(),
                                               torch.tensor([.7, .9])).state_dict()
                before = {k: v.clone() for k, v in before.items()}
                info = {}
                parts = fine_tune_main_network(
                    make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
                    'cpu', 4, 3, finetuning_dropout=0., gradient_clip_norm=None,
                    finetuning_parameter_scope=scope, finetuning_weight_decay=.1,
                    training_info=info, loader_generator=torch.Generator().manual_seed(42))
                after = MainNetworkTrainable(make_config(), 2, *parts).state_dict()
                for name in ('rf.0.weight', 'pca_mean', 'pca_components.weight'):
                    if scope != 'all':
                        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
                if scope == 'head':
                    for name in ('layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias'):
                        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
                self.assertFalse(torch.equal(before['layers.2.weight'], after['layers.2.weight']))
                self.assertFalse(torch.equal(before['nn_bias'], after['nn_bias']))
                self.assertEqual(info['finetuning_parameter_scope'], scope)

    def test_explicit_weight_decay_matches_manual_adamw(self):
        X, y = make_data()
        for decay in (0., .2):
            with self.subTest(decay=decay):
                torch.manual_seed(9)
                expected = MainNetworkTrainable(make_config(), 2, *make_network_parts(),
                                                torch.tensor([.7, .9])).train()
                opt = torch.optim.AdamW(expected.parameters(), lr=.01, weight_decay=decay)
                inputs, targets = next(iter(torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(X, y), batch_size=4, shuffle=True,
                    generator=torch.Generator().manual_seed(8))))
                torch.nn.functional.cross_entropy(expected(inputs, targets), targets).backward()
                opt.step()
                torch.manual_seed(9)
                actual = fine_tune_main_network(
                    make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
                    'cpu', 1, 4, finetuning_dropout=0, gradient_clip_norm=None,
                    finetuning_weight_decay=decay, loader_generator=torch.Generator().manual_seed(8))
                for name, value in MainNetworkTrainable(make_config(), 2, *actual).state_dict().items():
                    torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)

    def test_invalid_regularization_fails(self):
        X, y = make_data()
        for kwargs in ({'finetuning_weight_decay': -1}, {'finetuning_weight_decay': float('nan')},
                       {'finetuning_weight_decay': float('inf')}, {'finetuning_parameter_scope': 'invalid'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fine_tune_main_network(make_config(), X, y, 2, *make_network_parts(),
                                       torch.ones(2), 'cpu', 1, 4, **kwargs)

    def test_api_separates_context_and_training_batches_and_roundtrips_params(self):
        f = fixtures.ClassifierFineTuningTests()
        f.setUp()
        with patch.object(api, 'config', f.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=f.network('cpu')):
            model = f.classifier(n_ensemble=2, batch_size=6, finetuning_batch_size=2,
                                 optimize_steps=3, finetuning_parameter_scope='head',
                                 finetuning_weight_decay=.02)
            model.fit(f.X, f.y)
            self.assertTrue(all(len(rows) == 6 for rows in model._X_preds))
            self.assertTrue(all(h['finetuning_batch_size'] == 2 and h['steps'] == 3
                                and h['finetuning_parameter_scope'] == 'head'
                                and h['finetuning_weight_decay'] == .02
                                for h in model.finetuning_history_))
            f.check_probabilities(model, f.X)
            from sklearn.base import clone
            copied = clone(model)
            self.assertEqual(copied.get_params(), model.get_params())

    def test_neighbor_strength_is_isolated_and_zero_matches_no_bias(self):
        f = fixtures.ClassifierFineTuningTests()
        f.setUp()
        network = f.network('cpu')
        original = network.nn_bias.detach().clone()
        with patch.object(api, 'config', f.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=network):
            predictions = []
            for enabled, scale in ((False, 1.), (True, 0.), (True, .2)):
                model = f.classifier(optimization=None, nn_bias=enabled, nn_bias_scale=scale)
                model.fit(f.X, f.y)
                predictions.append(model.predict_proba(f.X))
                for bias in model._nnbias:
                    torch.testing.assert_close(bias, original * scale)
                    self.assertNotEqual(bias.data_ptr(), network.nn_bias.data_ptr())
                torch.testing.assert_close(network.nn_bias, original, rtol=0, atol=0)
            np.testing.assert_allclose(predictions[0], predictions[1], rtol=0, atol=0)
            self.assertFalse(np.array_equal(predictions[0], predictions[2]))

    def test_frozen_validation_checkpoints_match_api_predictions(self):
        from sklearn.metrics import roc_auc_score
        f = fixtures.ClassifierFineTuningTests()
        f.setUp()
        for scope in ('all', 'network', 'head'):
            with self.subTest(scope=scope), patch.object(api, 'config', f.config), \
                    patch.object(api.HyperFastClassifier, '_initialize_model', return_value=f.network('cpu')):
                model = f.classifier(n_ensemble=1, finetuning_parameter_scope=scope,
                                     finetuning_batch_size=3, nn_bias_scale=.2,
                                     optimize_steps=None, finetuning_max_epochs=4,
                                     patience_epochs=1, cooldown_checks=0)
                original = {name: value.detach().clone()
                            for name, value in model._model.named_parameters()}
                model.fit(f.X[:12], f.y[:12], eval_set=(f.X[12:], f.y[12:]))
                probabilities = model.predict_proba(f.X[12:])
                self.assertAlmostEqual(roc_auc_score(f.y[12:], probabilities[:, 1]),
                                       model.finetuning_history_[0]['best_score'])
                for name, value in model._model.named_parameters():
                    torch.testing.assert_close(value, original[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
