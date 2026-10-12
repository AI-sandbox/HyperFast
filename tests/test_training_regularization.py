import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

import hyperfast.hyperfast as api
from hyperfast.utils import MainNetworkTrainable, fine_tune_main_network, isolated_training_rng
from test_bias_finetuning import make_config, make_data, make_network_parts
import test_classifier_finetuning as fixtures


class TrainingRegularizationTests(unittest.TestCase):
    def test_smoothed_updates_match_manual_adamw(self):
        X, y = make_data()
        torch.manual_seed(42)
        expected = MainNetworkTrainable(make_config(), 2, *make_network_parts(),
                                        torch.tensor([.7, .9])).train()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), batch_size=3, shuffle=True,
            generator=torch.Generator().manual_seed(99))
        optimizer = torch.optim.AdamW(expected.parameters(), lr=.01)
        step = 0
        while step < 3:
            for inputs, targets in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    expected(inputs, targets), targets, label_smoothing=.05)
                loss.backward()
                optimizer.step()
                step += 1
                if step == 3:
                    break
        torch.manual_seed(42)
        parts = fine_tune_main_network(
            make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
            'cpu', 3, 3, finetuning_dropout=0, gradient_clip_norm=None,
            finetuning_scheduler='plateau', label_smoothing=.05,
            loader_generator=torch.Generator().manual_seed(99))
        actual = MainNetworkTrainable(make_config(), 2, *parts)
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)

    def test_validation_logloss_remains_unsmoothed(self):
        X, y = make_data()
        model = MainNetworkTrainable(make_config(), 2, *make_network_parts(),
                                     torch.tensor([.7, .9])).eval()
        with torch.no_grad():
            logits = model.validation_logits(X + .1, X, y)
            expected = torch.nn.functional.cross_entropy(logits, y).item()
            smoothed = torch.nn.functional.cross_entropy(logits, y, label_smoothing=.2).item()
        info = {}
        fine_tune_main_network(
            make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
            'cpu', 0, 4, X_val=X + .1, y_val=y, reference_X=X, reference_y=y,
            validation_metric='log_loss', label_smoothing=.2, training_info=info)
        self.assertAlmostEqual(info['history'][0]['score'], expected, places=7)
        self.assertNotAlmostEqual(expected, smoothed, places=6)
        self.assertEqual(info['best_step'], 0)

    def test_invalid_smoothing_fails_before_training(self):
        X, y = make_data()
        for value in (-.1, 1.1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'label_smoothing'):
                fine_tune_main_network(
                    make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
                    'cpu', 1, 4, label_smoothing=value)

    def test_training_rng_restores_state_on_success_and_exception(self):
        for fail in (False, True):
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            try:
                with isolated_training_rng(99) as generator:
                    random.random()
                    np.random.rand(3)
                    torch.rand(3)
                    if torch.cuda.is_available():
                        torch.rand(3, device='cuda')
                    first = torch.randperm(20, generator=generator)
                    if fail:
                        raise RuntimeError('training failed')
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual(random.getstate(), state[0])
            np.testing.assert_equal(np.random.get_state(), state[1])
            torch.testing.assert_close(torch.get_rng_state(), state[2], rtol=0, atol=0)
            if torch.cuda.is_available():
                for actual, original in zip(torch.cuda.get_rng_state_all(), state[3]):
                    torch.testing.assert_close(actual, original, rtol=0, atol=0)
            with isolated_training_rng(99) as generator:
                torch.testing.assert_close(torch.randperm(20, generator=generator), first)


class EnsembleRandomnessTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ClassifierFineTuningTests()
        self.fixture.setUp()

    def check_member_initializations(self, device):
        f = self.fixture
        network = f.network(device)
        original = api.fine_tune_main_network
        results = []
        with patch.object(api, 'config', f.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=network):
            for steps, smoothing, size in ((0, 0., 3), (3, .05, 5)):
                initializations = []

                def capture(*args, **kwargs):
                    initializations.append((
                        kwargs['reference_X'].clone(), kwargs['reference_y'].clone(),
                        args[4][0].weight.detach().cpu().clone(),
                        args[5].components_.detach().cpu().clone(),
                        [v.detach().cpu().clone() for layer in args[6] for v in layer]))
                    return original(*args, **kwargs)

                model = f.classifier(device, n_ensemble=size, optimize_steps=steps,
                                     label_smoothing=smoothing, batch_size=5,
                                     feature_bagging=True, feature_bagging_size=3,
                                     stratify_sampling=True)
                with patch.object(api, 'fine_tune_main_network', side_effect=capture):
                    model.fit(f.X, f.y)
                results.append((initializations, model.selected_features))
                f.check_probabilities(model, f.X)
                self.assertEqual(model.get_params()['label_smoothing'], smoothing)
                self.assertEqual([h['minibatch_seed'] for h in model.finetuning_history_],
                                 [42 + 104729 * (i + 1) for i in range(size)])
            for i in range(3):
                torch.testing.assert_close(results[0][1][i], results[1][1][i], rtol=0, atol=0)
                for a, b in zip(results[0][0][i], results[1][0][i]):
                    if isinstance(a, list):
                        for x, y in zip(a, b):
                            torch.testing.assert_close(x, y, rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_member_initializations_do_not_depend_on_training_or_ensemble_size(self):
        self.check_member_initializations('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_member_initializations_are_independent_on_cuda(self):
        self.check_member_initializations('cuda')

    def test_smoothing_with_more_classes_than_checkpoint(self):
        f = self.fixture
        with patch.object(api, 'config', f.config), \
                patch.object(api.HyperFastClassifier, '_initialize_model', return_value=f.network('cpu')):
            model = f.classifier(label_smoothing=.05, batch_size=5, stratify_sampling=True)
            model.fit(f.X, np.arange(len(f.X)) % 4)
            f.check_probabilities(model, f.X, classes=4)


if __name__ == '__main__':
    unittest.main()
