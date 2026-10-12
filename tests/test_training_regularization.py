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
            for steps, size in ((0, 3), (3, 5)):
                initializations = []

                def capture(*args, **kwargs):
                    initializations.append((
                        kwargs['reference_X'].clone(), kwargs['reference_y'].clone(),
                        args[4][0].weight.detach().cpu().clone(),
                        args[5].components_.detach().cpu().clone(),
                        [v.detach().cpu().clone() for layer in args[6] for v in layer]))
                    return original(*args, **kwargs)

                model = f.classifier(device, n_ensemble=size, optimize_steps=steps,
                                     batch_size=5,
                                     feature_bagging=True, feature_bagging_size=3,
                                     stratify_sampling=True)
                with patch.object(api, 'fine_tune_main_network', side_effect=capture):
                    model.fit(f.X, f.y)
                results.append((initializations, model.selected_features))
                f.check_probabilities(model, f.X)
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



if __name__ == '__main__':
    unittest.main()
