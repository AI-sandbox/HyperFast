import random
import unittest
import numpy as np
import torch
from hyperfast.utils import NN, nn_bias_logits, isolated_training_rng, MainNetworkTrainable
from test_bias_finetuning import make_config, make_network_parts, make_data

class NeighborSmoothingTests(unittest.TestCase):
    def check_neighbors(self, device):
        x = torch.tensor([[0.], [0.], [1.], [3.]], device=device)
        y = torch.tensor([0, 1, 1, 2], device=device)
        model = NN(x, y)
        for loo in (False, True):
            for chunked in (False, True):
                for k in (1, 2, 3, 9):
                    actual = model.predict_proba(x, 3, k, chunked, loo)
                    d = (x[:, None] - x[None, :]).square().sum(-1)
                    if loo:
                        d.fill_diagonal_(float('inf'))
                    effective = min(k, len(x) - int(loo))
                    indices = (d.argmin(1, keepdim=True) if effective == 1 else
                               d.topk(effective, largest=False).indices)
                    expected = torch.nn.functional.one_hot(y[indices], 3).float().mean(1)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual.sum(1), torch.ones(len(x), device=device))
                    if k == 1:
                        legacy = (model.predict_from_training_with_LOO(chunked) if loo else
                                  model.predict(x, chunked))
                        self.assertTrue(torch.equal(actual.argmax(1), legacy))
        bias = torch.tensor(.5, device=device, requires_grad=True)
        logits = nn_bias_logits(torch.zeros(4, 3, device=device), x, x, y, bias, 3, True, 2)
        logits[:, 1].sum().backward()
        self.assertGreater(bias.grad.item(), 0)

    def test_neighbors_cpu(self):
        self.check_neighbors('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_neighbors_cuda(self):
        self.check_neighbors('cuda')

    def test_invalid_and_empty_context(self):
        x = torch.zeros(1, 1)
        model = NN(x, torch.zeros(1, dtype=torch.long))
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                model.predict_proba(x, 2, value)
        with self.assertRaises(ValueError):
            model.predict_proba(x, 2, leave_one_out=True)
        model = NN(torch.tensor([[0.], [1.]]), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            model.predict_proba(torch.tensor([[1.], [0.]]), 2, leave_one_out=True)

    def test_training_and_validation_use_soft_correction(self):
        cfg = make_config()
        cfg.nn_bias_neighbors = 3
        model = MainNetworkTrainable(cfg, 2, *make_network_parts(), torch.tensor([.7, .9])).eval()
        x, y = make_data()
        plain, representations = model._forward_features(x)
        expected = plain.clone()
        for representation, bias in zip(representations, model.nn_bias):
            expected = expected + bias * NN(representation, y).predict_proba(
                representation, 2, 3, leave_one_out=True)
        actual = model(x, y)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(actual, y).backward()
        self.assertTrue(torch.isfinite(model.nn_bias.grad).all())
        expected = plain.clone()
        for representation, bias in zip(representations, model.nn_bias):
            expected = expected + bias * NN(representation, y).predict_proba(representation, 2, 3)
        torch.testing.assert_close(model.validation_logits(x, x, y), expected, rtol=0, atol=0)

class IsolatedRandomnessTests(unittest.TestCase):
    def draws(self, seed):
        with isolated_training_rng(seed):
            result = [random.random(), np.random.random(), torch.rand(5)]
            if torch.cuda.is_available():
                result.append(torch.rand(5, device='cuda').cpu())
        return result

    def test_seed_controls_every_generator_independently_of_entry_state(self):
        first = self.draws(42)
        random.seed(99); np.random.seed(99); torch.manual_seed(99)
        second = self.draws(42)
        third = self.draws(43)
        self.assertEqual(first[:2], second[:2])
        self.assertNotEqual(first[:2], third[:2])
        for a, b, c in zip(first[2:], second[2:], third[2:]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(torch.equal(a, c))

class ClassifierNeighborTests(unittest.TestCase):
    def check_api(self, device):
        from unittest.mock import patch
        from sklearn.base import clone
        from sklearn.metrics import roc_auc_score, log_loss
        import hyperfast.hyperfast as api
        import test_classifier_finetuning as fixtures
        for n_classes in (2, 3):
            f = fixtures.ClassifierFineTuningTests()
            f.setUp()
            f.config['max_categories'] = 3
            f.y = np.arange(len(f.X)) % n_classes
            network = f.network(device)
            with patch.object(api, 'config', f.config), patch.object(
                    api.HyperFastClassifier, '_initialize_model', return_value=network):
                for neighbors in (1, 4, 32):
                    with self.subTest(device=device, n_classes=n_classes, neighbors=neighbors):
                        model = f.classifier(device, n_ensemble=1, nn_bias_neighbors=neighbors,
                                             finetuning_batch_size=3, optimize_steps=None,
                                             finetuning_max_epochs=3, patience_epochs=1,
                                             cooldown_checks=0, nn_bias_scale=.2)
                        checkpoint = {name: value.detach().clone()
                                      for name, value in network.state_dict().items()}
                        model.fit(f.X[:12], f.y[:12], eval_set=(f.X[12:], f.y[12:]))
                        probabilities = model.predict_proba(f.X[12:])
                        expected = (roc_auc_score(f.y[12:], probabilities[:, 1]) if n_classes == 2
                                    else log_loss(f.y[12:], probabilities, labels=np.arange(n_classes)))
                        self.assertAlmostEqual(expected, model.finetuning_history_[0]['best_score'], places=6)
                        self.assertEqual(clone(model).get_params(), model.get_params())
                        # A fresh fit with the same seed must reproduce every probability.
                        repeated = f.classifier(device, **{k: v for k, v in model.get_params().items()
                                                          if k != 'device'})
                        repeated.fit(f.X[:12], f.y[:12], eval_set=(f.X[12:], f.y[12:]))
                        np.testing.assert_array_equal(repeated.predict_proba(f.X[12:]), probabilities)
                        for name, value in network.state_dict().items():
                            torch.testing.assert_close(value, checkpoint[name], rtol=0, atol=0)

    def test_api_cpu(self):
        self.check_api('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_api_cuda(self):
        self.check_api('cuda')
