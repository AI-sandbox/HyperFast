import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import hyperfast.hyperfast as api
from hyperfast.model import HyperFast
from hyperfast.utils import (MainNetworkTrainable, fine_tune_main_network,
                             transform_data_for_main_network, forward_main_network,
                             nn_bias_logits, NN, TorchPCA)
from test_bias_finetuning import make_config, make_data, make_network_parts


class ValidationFineTuningTests(unittest.TestCase):
    def fit_parts(self, **kwargs):
        X, y = make_data()
        info = {}
        parts = fine_tune_main_network(
            make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
            'cpu', None, 4, X_val=X + .1, y_val=y, reference_X=X, reference_y=y,
            max_epochs=4, patience_epochs=1, cooldown_checks=0,
            training_info=info, **kwargs)
        return parts, info

    def test_initial_checkpoint_is_restored_when_training_does_not_improve(self):
        with patch('sklearn.metrics.roc_auc_score', side_effect=[.8, .7]):
            parts, info = self.fit_parts()
        self.assertEqual(info['best_step'], 0)
        self.assertEqual(info['steps'], 1)
        self.assertEqual(info['stop_reason'], 'patience')
        rf, pca, layers = make_network_parts()
        torch.testing.assert_close(parts[0][0].weight, rf[0].weight, rtol=0, atol=0)
        torch.testing.assert_close(parts[1].components_, pca.components_, rtol=0, atol=0)
        torch.testing.assert_close(parts[-1], torch.tensor([.7, .9]), rtol=0, atol=0)
        for got, expected in zip(parts[2], layers):
            for a, b in zip(got, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_best_intermediate_checkpoint_is_restored_not_last(self):
        states = []
        original = MainNetworkTrainable.validation_logits

        def capture(model, *args, **kwargs):
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return original(model, *args, **kwargs)

        with patch('sklearn.metrics.roc_auc_score', side_effect=[.5, .8, .7]), \
                patch.object(MainNetworkTrainable, 'validation_logits', capture):
            parts, info = self.fit_parts()
        self.assertEqual(info['best_step'], 1)
        self.assertEqual(info['steps'], 2)
        restored = MainNetworkTrainable(make_config(), 2, *parts)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, states[1][key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(states[1][k], states[2][k]) for k in states[1]))

    def test_validation_neighbors_match_predict_and_never_use_validation_labels(self):
        X, y = make_data()
        query = X + .17
        rf, old_pca, layers = make_network_parts()
        pca = TorchPCA(n_components=2)
        pca.mean_, pca.components_ = old_pca.mean_, old_pca.components_
        model = MainNetworkTrainable(make_config(), 2, rf, pca, layers,
                                     torch.tensor([.7, .9]), finetuning_dropout=.15).eval()
        with torch.no_grad(), patch.object(NN, 'predict_from_training_with_LOO', side_effect=AssertionError):
            actual = model.validation_logits(query, X, y)
            torch.testing.assert_close(actual, model.validation_logits(query, X, y), rtol=0, atol=0)
            rf, pca, layers, bias = model.get_main_network_parts()
            transformed = transform_data_for_main_network(query.clone(), make_config(), rf, pca)
            reference = transform_data_for_main_network(X.clone(), make_config(), rf, pca)
            expected, hidden = forward_main_network(transformed, layers)
            _, reference_hidden = forward_main_network(reference, layers)
            for q, r, b in zip([query, hidden], [X, reference_hidden], bias):
                expected = nn_bias_logits(expected, q, r, y, b, 2, True)
        torch.testing.assert_close(actual, expected)

    def test_dropout_only_runs_in_training(self):
        model = MainNetworkTrainable(make_config(), 2, *make_network_parts(),
                                     torch.tensor([.7, .9]), finetuning_dropout=.15)
        X, y = make_data()
        original = torch.nn.functional.dropout
        with patch('torch.nn.functional.dropout', wraps=original) as dropout:
            model.train()
            model(X, y)
            self.assertTrue(all(c.kwargs['training'] and c.kwargs['p'] == .15
                                for c in dropout.call_args_list))
            dropout.reset_mock()
            model.eval()
            model.validation_logits(X, X, y)
            self.assertTrue(all(not c.kwargs['training'] for c in dropout.call_args_list))

    def test_gradient_clipping_and_cosine_floor(self):
        X, y = make_data()
        rates = []
        original_step = torch.optim.AdamW.step

        def step(optimizer, *a, **k):
            rates.append(optimizer.param_groups[0]['lr'])
            return original_step(optimizer, *a, **k)

        with patch.object(torch.optim.AdamW, 'step', step), \
                patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
            fine_tune_main_network(make_config(), X, y, 2, *make_network_parts(),
                                   torch.tensor([.7, .9]), 'cpu', None, 4,
                                   max_epochs=4, scheduler_min_lr=.001)
        self.assertEqual(clip.call_count, 4)
        self.assertTrue(all(c.args[1] == 1.18 for c in clip.call_args_list))
        self.assertEqual(rates[0], .01)
        self.assertTrue(all(a > b >= .001 for a, b in zip(rates, rates[1:])))

    def test_single_class_validation_falls_back_to_log_loss(self):
        X, y = make_data()
        info = {}
        with self.assertWarnsRegex(UserWarning, 'one class'):
            fine_tune_main_network(make_config(), X, y, 2, *make_network_parts(),
                                   torch.tensor([.7, .9]), 'cpu', 0, 4,
                                   X_val=X[:1], y_val=y[:1], reference_X=X, reference_y=y,
                                   training_info=info)
        self.assertEqual(info['validation_metric'], 'log_loss')
        self.assertTrue(np.isfinite(info['best_score']))

    def test_log_loss_weights_partial_validation_batches_by_rows(self):
        X, y = make_data()
        cfg = make_config()
        cfg.nn_bias = False
        info = {}
        query = torch.cat([X, X[:1] + 3])
        labels = torch.cat([y, y[:1]])
        parts = fine_tune_main_network(cfg, X, y, 2, *make_network_parts(),
                                       torch.tensor([.7, .9]), 'cpu', 0, 2,
                                       X_val=query, y_val=labels,
                                       validation_metric='log_loss', training_info=info)
        model = MainNetworkTrainable(cfg, 2, *parts).eval()
        expected = torch.nn.functional.cross_entropy(model(query), labels).item()
        self.assertAlmostEqual(info['best_score'], expected, places=6)

    def test_epoch_ceiling_is_respected_even_with_a_larger_update_cap(self):
        X, y = make_data()
        info = {}
        fine_tune_main_network(make_config(), X, y, 2, *make_network_parts(),
                               torch.tensor([.7, .9]), 'cpu', 100, 4,
                               max_epochs=2, training_info=info)
        self.assertEqual(info['steps'], 2)
        self.assertEqual(info['stop_reason'], 'max_epochs')


class ClassifierValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = {**api.config, 'n_dims': 2, 'rf_size': 8,
                       'hn_hidden_size': 8, 'hn_n_layers': 2, 'max_categories': 2}
        torch.manual_seed(42)
        self.network = HyperFast(SimpleNamespace(**self.config, torch_pca=True)).eval()
        rng = np.random.default_rng(4)
        self.X = rng.normal(size=(30, 3)).astype(np.float32)
        self.y = np.arange(30) % 2

    def fit(self, X=None, y=None, **kwargs):
        X = self.X if X is None else X
        y = self.y if y is None else y
        eval_set = kwargs.pop('eval_set', (self.X[:4] + 100, self.y[:4]))
        with patch.object(api, 'config', self.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=self.network):
            model = api.HyperFastClassifier(device='cpu', n_ensemble=2, batch_size=30,
                                            finetuning_max_epochs=2, seed=42, **kwargs)
            model.fit(X, y, eval_set=eval_set)
        return model

    def test_validation_is_excluded_from_preprocessing_generation_and_references(self):
        with patch.object(api, 'fine_tune_main_network', wraps=api.fine_tune_main_network) as fit:
            model = self.fit(nn_bias=True, feature_bagging=True, feature_bagging_size=2)
        np.testing.assert_allclose(model._scaler.mean_, self.X.mean(axis=0), atol=1e-7)
        for member, call in enumerate(fit.call_args_list):
            train = model._preprocess_test_data(self.X)[:, model.selected_features[member]]
            validation = model._preprocess_test_data(self.X[:4] + 100)[:, model.selected_features[member]]
            torch.testing.assert_close(call.args[1], train)
            torch.testing.assert_close(call.kwargs['X_val'], validation)
            self.assertEqual(len(call.kwargs['reference_X']), len(train))
            self.assertLess(call.kwargs['reference_X'].abs().max(), 5)
        self.assertEqual(len(model.best_steps_), 2)
        self.assertTrue(np.isfinite(model.predict_proba(self.X)).all())

    def test_members_select_their_own_checkpoints(self):
        with patch('sklearn.metrics.roc_auc_score', side_effect=[.8, .7, .6, .5, .6, .8]):
            model = self.fit(patience_epochs=5)
        self.assertEqual(model.best_steps_, [0, 2])

    def test_internal_holdout_precedes_preprocessing(self):
        model = self.fit(eval_set=None, validation_fraction=.2)
        self.assertEqual(len(model._X_preds[0]), 24)
        self.assertEqual(model._scaler.n_samples_seen_, 24)
        self.assertTrue(all(h['validation_checks'] > 0 for h in model.finetuning_history_))

    def test_single_member_optimize_uses_validation(self):
        with patch.object(api, 'config', self.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=self.network):
            model = api.HyperFastClassifier(device='cpu', n_ensemble=1, batch_size=30,
                                            optimization='optimize', finetuning_max_epochs=2,
                                            nn_bias=True, seed=42)
            model.fit(self.X, self.y, eval_set=(self.X[:4] + .1, self.y[:4]))
        self.assertEqual(len(model.best_steps_), 1)
        self.assertGreater(model.finetuning_history_[0]['validation_checks'], 0)
        self.assertTrue(np.isfinite(model.predict_proba(self.X)).all())

    def test_string_labels_and_unseen_validation_labels(self):
        labels = np.where(self.y, 'case', 'control')
        model = self.fit(y=labels, eval_set=(self.X[:4], labels[:4]))
        self.assertEqual(set(model.predict(self.X)), set(labels))
        with self.assertRaisesRegex(ValueError, 'unseen labels'):
            self.fit(eval_set=(self.X[:4], np.array(['other'] * 4)))

    def test_validation_supports_multiclass_and_training_only_categories(self):
        X = self.X.astype(object)
        X[:, 2] = np.where(self.y, 'red', 'blue')
        query = X[:6].copy()
        query[:, 2] = 'unseen'
        labels = np.arange(len(X)) % 3
        model = self.fit(X, labels, cat_features=[2], nn_bias=True,
                         eval_set=(query, labels[:6]))
        self.assertTrue(all(h['validation_metric'] == 'log_loss' for h in model.finetuning_history_))
        self.assertEqual(model.predict_proba(query).shape, (6, 3))
        encoder = model.one_hot_encoder.named_transformers_['cat']
        self.assertNotIn('unseen', encoder.categories_[0])

    def test_validation_fits_are_reproducible_and_checkpoint_stays_unchanged(self):
        original = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
        first = self.fit(nn_bias=True)
        expected = first.predict_proba(self.X)
        self.fit(y=1 - self.y, nn_bias=True)
        second = self.fit(nn_bias=True)
        np.testing.assert_array_equal(expected, second.predict_proba(self.X))
        np.testing.assert_array_equal(expected, first.predict_proba(self.X))
        for k, v in self.network.state_dict().items():
            torch.testing.assert_close(v, original[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
