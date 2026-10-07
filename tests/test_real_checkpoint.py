import gc
import os
import unittest

import numpy as np
import torch

from hyperfast import HyperFastClassifier


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('HYPERFAST_TEST_CHECKPOINT'),
                     'CUDA and HYPERFAST_TEST_CHECKPOINT are required')
class RealCheckpointTests(unittest.TestCase):
    def test_fine_tuning_with_feature_bagging(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(33, 5001)).astype(np.float32)
        y = np.arange(len(X)) % 2
        query = rng.normal(size=(8, X.shape[1])).astype(np.float32)
        for optimization, n_ensemble in [('ensemble_optimize', 2), ('optimize', 1)]:
            with self.subTest(optimization=optimization):
                model = HyperFastClassifier(
                    device='cuda', custom_path=os.environ['HYPERFAST_TEST_CHECKPOINT'],
                    n_ensemble=n_ensemble, batch_size=32, nn_bias=True,
                    feature_bagging=True, feature_bagging_size=3000,
                    optimization=optimization, optimize_steps=2, seed=42)
                original_bias = model._model.nn_bias.detach().clone()
                model.fit(X, y)
                self.assertTrue(all(len(samples) == 32 for samples in model._X_preds))
                self.assertTrue(any(not torch.equal(bias.detach().cpu(), original_bias.cpu())
                                    for bias in model._nnbias))
                probabilities = model.predict_proba(query)
                self.assertEqual(probabilities.shape, (len(query), 2))
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                torch.testing.assert_close(model._model.nn_bias, original_bias, rtol=0, atol=0)
                del model
                gc.collect()
                torch.cuda.empty_cache()

    def test_validation_checkpoint_restoration_with_feature_bagging(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(33, 5001)).astype(np.float32)
        y = np.arange(len(X)) % 3
        validation = rng.normal(size=(9, X.shape[1])).astype(np.float32)
        y_val = np.arange(9) % 3
        query = rng.normal(size=(8, X.shape[1])).astype(np.float32)
        model = HyperFastClassifier(
            device='cuda', custom_path=os.environ['HYPERFAST_TEST_CHECKPOINT'],
            n_ensemble=2, batch_size=32, nn_bias=True,
            feature_bagging=True, feature_bagging_size=3000,
            finetuning_max_epochs=3, patience_epochs=1, cooldown_checks=0, seed=42)
        original_bias = model._model.nn_bias.detach().clone()
        model.fit(X, y, eval_set=(validation, y_val))
        self.assertEqual(len(model.best_steps_), 2)
        self.assertTrue(all(h['validation_metric'] == 'log_loss'
                            and h['validation_checks'] >= 2 for h in model.finetuning_history_))
        probabilities = model.predict_proba(query)
        self.assertEqual(probabilities.shape, (8, 3))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        torch.testing.assert_close(model._model.nn_bias, original_bias, rtol=0, atol=0)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
