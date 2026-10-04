import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import hyperfast.hyperfast as api
import hyperfast.model as model
from hyperfast.utils import seed_everything


class ManyClassTests(unittest.TestCase):
    def setUp(self):
        self.config = {**api.config, 'n_dims': 8, 'rf_size': 16,
                       'hn_hidden_size': 8, 'hn_n_layers': 2, 'max_categories': 46}
        torch.manual_seed(42)
        self.network = model.HyperFast(SimpleNamespace(**self.config, torch_pca=True)).eval()
        self.rng = np.random.default_rng(42)

    def classifier(self, optimization=None, batch_size=128, stratify_sampling=False):
        return api.HyperFastClassifier(
            device='cpu', n_ensemble=1, batch_size=batch_size, nn_bias=False,
            optimization=optimization, optimize_steps=2, seed=42,
            stratify_sampling=stratify_sampling)

    def check_probabilities(self, classifier, X, classes):
        probabilities = classifier.predict_proba(X)
        self.assertEqual(probabilities.shape, (len(X), len(classes)))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(classifier.classes_, classes)
        np.testing.assert_array_equal(
            classifier.predict(X), classes[probabilities.argmax(axis=1)])
        return probabilities

    def test_colliding_encodings_keep_separate_output_classes(self):
        for n_classes in (47, 50):
            X = self.rng.normal(size=(2 * n_classes, 7)).astype(np.float32)
            classes = np.arange(n_classes) * 7 - 100
            y = np.tile(classes, 2)
            for optimization in (None, 'ensemble_optimize', 'optimize'):
                with self.subTest(n_classes=n_classes, optimization=optimization), \
                        patch.object(api, 'config', self.config), \
                        patch.object(api.HyperFastClassifier, '_initialize_model', return_value=self.network), \
                        patch.object(api, 'fine_tune_main_network', wraps=api.fine_tune_main_network) as fine_tune:
                    classifier = self.classifier(optimization)
                    classifier.fit(X, y)
                    probabilities = self.check_probabilities(classifier, X[:9], classes)
                    weights, bias = classifier._main_networks[0][-1]
                    self.assertEqual(weights.shape[1], n_classes)
                    self.assertEqual(bias.shape, (n_classes,))
                    self.assertFalse(torch.equal(weights[:, 0], weights[:, 46]))
                    self.assertFalse(np.array_equal(probabilities[:, 0], probabilities[:, 46]))
                    self.assertEqual(fine_tune.call_count, int(optimization is not None))
                    if optimization is not None:
                        torch.testing.assert_close(
                            torch.unique(fine_tune.call_args.args[2]), torch.arange(n_classes))
                        self.assertEqual(fine_tune.call_args.args[3], n_classes)

    def test_supported_class_counts_preserve_original_encoding(self):
        for n_classes in (2, 46):
            with self.subTest(n_classes=n_classes):
                X = torch.from_numpy(self.rng.normal(size=(max(16, n_classes), 7)).astype(np.float32))
                y = torch.arange(len(X)) % n_classes
                original_one_hot = model.F.one_hot

                def original_encoding(labels, num_classes):
                    torch.testing.assert_close(labels, y, rtol=0, atol=0)
                    return original_one_hot(y, num_classes)

                torch.manual_seed(123)
                with torch.no_grad():
                    current = self.network(X, y, n_classes)
                torch.manual_seed(123)
                with torch.no_grad(), patch.object(model.F, 'one_hot', side_effect=original_encoding) as one_hot:
                    original = self.network(X, y, n_classes)
                one_hot.assert_called_once()
                for current_layer, original_layer in zip(current[2], original[2]):
                    for current_parameter, original_parameter in zip(current_layer, original_layer):
                        torch.testing.assert_close(current_parameter, original_parameter, rtol=0, atol=0)
                torch.testing.assert_close(current[3], original[3], rtol=0, atol=0)

    def test_stratified_sampling_when_classes_exceed_batch_size(self):
        X = torch.arange(200, dtype=torch.float32).reshape(100, 2)
        y = torch.arange(100) % 50
        with patch.object(api, 'config', self.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=self.network):
            classifier = self.classifier(batch_size=4, stratify_sampling=True)
            seed_everything(42)
            sampled_X, sampled_y = classifier._sample_data(X, y)
            seed_everything(42)
            repeated_X, repeated_y = classifier._sample_data(X, y)
        self.assertEqual(sampled_X.shape, (4, 2))
        self.assertEqual(len(torch.unique(sampled_X, dim=0)), 4)
        self.assertEqual(len(torch.unique(sampled_y)), 4)
        torch.testing.assert_close(sampled_X, repeated_X, rtol=0, atol=0)
        torch.testing.assert_close(sampled_y, repeated_y, rtol=0, atol=0)
        row_indices = (sampled_X[:, 0] / 2).long()
        torch.testing.assert_close(sampled_y, y[row_indices])

    def test_stratified_fits_with_more_classes_than_batch_size(self):
        X = self.rng.normal(size=(100, 7)).astype(np.float32)
        classes = np.arange(50) * 7 - 100
        y = np.tile(classes, 2)
        for optimization in (None, 'ensemble_optimize', 'optimize'):
            with self.subTest(optimization=optimization), \
                    patch.object(api, 'config', self.config), \
                    patch.object(api.HyperFastClassifier, '_initialize_model', return_value=self.network):
                classifier = self.classifier(optimization, batch_size=4, stratify_sampling=True)
                classifier.fit(X, y)
                self.assertEqual(len(classifier._X_preds[0]), 4)
                self.assertEqual(len(torch.unique(classifier._y_preds[0])), 4)
                self.check_probabilities(classifier, X[:9], classes)


if __name__ == '__main__':
    unittest.main()
