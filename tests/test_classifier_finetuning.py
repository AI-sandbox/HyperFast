import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import hyperfast.hyperfast as api
from hyperfast.model import HyperFast


class ClassifierFineTuningTests(unittest.TestCase):
    def setUp(self):
        self.config = {**api.config, 'n_dims': 8, 'rf_size': 16,
                       'hn_hidden_size': 8, 'hn_n_layers': 2, 'max_categories': 2}
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(17, 7)).astype(np.float32)
        self.y = np.arange(len(self.X)) % 2

    def network(self, device):
        torch.manual_seed(42)
        return HyperFast(SimpleNamespace(**self.config, torch_pca=True)).to(device).eval()

    def classifier(self, device='cpu', **params):
        return api.HyperFastClassifier(
            **{'device': device, 'n_ensemble': 2, 'batch_size': 32,
               'nn_bias': True, 'optimization': 'ensemble_optimize',
               'optimize_steps': 2, 'seed': 42, **params})

    def check_probabilities(self, classifier, X, classes=2):
        probabilities = classifier.predict_proba(X)
        self.assertEqual(probabilities.shape, (len(X), classes))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)

    def test_padding_is_only_used_for_network_generation(self):
        network = self.network('cpu')
        original = network.nn_bias.detach().clone()
        with patch.object(api, 'config', self.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=network), \
                patch.object(network, 'forward', wraps=network.forward) as forward, \
                patch.object(api, 'fine_tune_main_network', wraps=api.fine_tune_main_network) as fine_tune:
            classifier = self.classifier()
            classifier.fit(self.X[:5], self.y[:5])
            self.assertEqual([len(call.args[0]) for call in forward.call_args_list], [10, 10])
            self.assertEqual([len(call.args[1]) for call in fine_tune.call_args_list], [5, 5])
            self.assertEqual([len(x) for x in classifier._X_preds], [5, 5])
            self.assertEqual([len(torch.unique(x, dim=0)) for x in classifier._X_preds], [5, 5])
            self.check_probabilities(classifier, self.X)
        torch.testing.assert_close(network.nn_bias, original, rtol=0, atol=0)

    def check_stored_bias(self, device):
        network = self.network(device)
        original = network.nn_bias.detach().clone()
        real_fine_tune = api.fine_tune_main_network
        trained = []
        def capture(*args, **kwargs):
            parts = real_fine_tune(*args, **kwargs)
            trained.append(parts[-1].detach().cpu().clone())
            return parts
        with patch.object(api, 'config', self.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=network), \
                patch.object(api, 'fine_tune_main_network', side_effect=capture):
            classifier = self.classifier(device)
            classifier.fit(self.X, self.y)
            for stored, fitted in zip(classifier._nnbias, trained):
                torch.testing.assert_close(stored.detach().cpu(), fitted, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, original.cpu()) for value in trained))
            self.check_probabilities(classifier, self.X)
        torch.testing.assert_close(network.nn_bias, original, rtol=0, atol=0)

    def test_ensemble_fit_stores_learned_bias(self):
        self.check_stored_bias('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_ensemble_fit_stores_learned_bias_cuda(self):
        self.check_stored_bias('cuda')

    def check_fit_independence(self, optimization, n_ensemble):
        network = self.network('cpu')
        checkpoint = {name: value.detach().clone()
                      for name, value in network.state_dict().items()}

        def assert_checkpoint_unchanged():
            for name, value in network.state_dict().items():
                torch.testing.assert_close(value, checkpoint[name], rtol=0, atol=0)

        with patch.object(api, 'config', self.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=network):
            first = self.classifier(optimization=optimization, n_ensemble=n_ensemble)
            first.fit(self.X, self.y)
            expected = first.predict_proba(self.X)
            assert_checkpoint_unchanged()

            intervening = self.classifier(optimization=optimization, n_ensemble=n_ensemble)
            intervening.fit(self.X, 1 - self.y)
            assert_checkpoint_unchanged()
            np.testing.assert_array_equal(first.predict_proba(self.X), expected)

            repeated = self.classifier(optimization=optimization, n_ensemble=n_ensemble)
            repeated.fit(self.X, self.y)
            assert_checkpoint_unchanged()
            np.testing.assert_array_equal(repeated.predict_proba(self.X), expected)
            np.testing.assert_array_equal(first.predict_proba(self.X), expected)

    def test_ensemble_fits_are_independent(self):
        self.check_fit_independence('ensemble_optimize', 2)

    def test_full_optimization_fits_are_independent(self):
        self.check_fit_independence('optimize', 1)

    def check_full_optimization(self, device, feature_bagging):
        network = self.network(device)
        original = network.nn_bias.detach().clone()
        with patch.object(api, 'config', self.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=network), \
                patch.object(api, 'fine_tune_main_network', wraps=api.fine_tune_main_network) as fine_tune:
            classifier = self.classifier(device, n_ensemble=1, optimization='optimize',
                                         batch_size=5, feature_bagging=feature_bagging,
                                         feature_bagging_size=3)
            classifier.fit(self.X, self.y)
            used_X = fine_tune.call_args.args[1]
            transformed_X = classifier._preprocess_test_data(self.X)
            expected = (transformed_X[:, classifier.selected_features[0]]
                        if feature_bagging else transformed_X)
            torch.testing.assert_close(used_X.cpu(), expected)
            self.assertEqual(len(used_X), len(self.X))
            self.assertEqual(used_X.shape[1], 3 if feature_bagging else 7)
            self.assertFalse(torch.equal(classifier._nnbias[0].detach().cpu(), original.cpu()))
            self.check_probabilities(classifier, self.X)
        torch.testing.assert_close(network.nn_bias, original, rtol=0, atol=0)

    def test_full_optimization_respects_feature_bagging(self):
        self.check_full_optimization('cpu', True)

    def test_full_optimization_without_feature_bagging(self):
        self.check_full_optimization('cpu', False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_full_optimization_respects_feature_bagging_cuda(self):
        self.check_full_optimization('cuda', True)

    def test_padding_with_many_classes_and_feature_bagging(self):
        network = self.network('cpu')
        with patch.object(api, 'config', self.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=network), \
                patch.object(api, 'fine_tune_main_network', wraps=api.fine_tune_main_network) as fine_tune:
            classifier = self.classifier(n_ensemble=1, feature_bagging=True,
                                         feature_bagging_size=3)
            classifier.fit(self.X[:6], np.arange(6) % 3)
            self.assertEqual(len(fine_tune.call_args.args[1]), 6)
            self.assertEqual(len(classifier._X_preds[0]), 6)
            self.check_probabilities(classifier, self.X, classes=3)


if __name__ == '__main__':
    unittest.main()
