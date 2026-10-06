import unittest
from unittest.mock import patch

import torch

from hyperfast import HyperFastClassifier


class SamplingTests(unittest.TestCase):
    def classifier(self, batch_size, stratify=True):
        with patch.object(HyperFastClassifier, '_initialize_model', return_value=None):
            return HyperFastClassifier(device='cpu', batch_size=batch_size,
                                       stratify_sampling=stratify, seed=42)

    def data(self, counts):
        y = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
        ids = torch.arange(len(y), dtype=torch.float32)
        return torch.stack((ids, -ids), dim=1), y

    def check_sample(self, X, y, sampled_X, sampled_y, batch_size):
        ids = sampled_X[:, 0].long()
        self.assertEqual(len(ids), min(len(X), batch_size))
        self.assertEqual(len(torch.unique(ids)), len(ids))
        torch.testing.assert_close(sampled_X, X[ids], rtol=0, atol=0)
        torch.testing.assert_close(sampled_y, y[ids], rtol=0, atol=0)

    def test_small_imbalanced_dataset_uses_every_row_once(self):
        X, y = self.data([990, 10])
        sampled = self.classifier(1024)._sample_data(X, y)
        self.check_sample(X, y, *sampled, 1024)
        torch.testing.assert_close(sampled[0][:, 0].sort().values, X[:, 0])

    def test_rare_class_unused_quota_is_redistributed(self):
        X, y = self.data([2000, 10])
        sampled = self.classifier(1024)._sample_data(X, y)
        self.check_sample(X, y, *sampled, 1024)
        torch.testing.assert_close(torch.bincount(sampled[1]), torch.tensor([1014, 10]))

    def test_multiple_exhausted_classes_redistribute_evenly(self):
        X, y = self.data([1, 4, 100, 100])
        sampled = self.classifier(21)._sample_data(X, y)
        self.check_sample(X, y, *sampled, 21)
        torch.testing.assert_close(torch.bincount(sampled[1]), torch.tensor([1, 4, 8, 8]))

    def test_remainder_is_filled_without_favoring_fixed_classes(self):
        X, y = self.data([100, 100, 100])
        classifier = self.classifier(8)
        smaller_classes = set()
        for seed in range(10):
            torch.manual_seed(seed)
            sampled = classifier._sample_data(X, y)
            self.check_sample(X, y, *sampled, 8)
            counts = torch.bincount(sampled[1])
            torch.testing.assert_close(counts.sort().values, torch.tensor([2, 3, 3]))
            smaller_classes.add(int(counts.argmin()))
        self.assertEqual(smaller_classes, {0, 1, 2})

    def test_batch_sizes_preserve_capacity_and_class_coverage(self):
        X, y = self.data([2, 1, 5, 20, 3])
        for batch_size in (1, 4, 5, 6, 17, 31, 40):
            with self.subTest(batch_size=batch_size):
                sampled = self.classifier(batch_size)._sample_data(X, y)
                self.check_sample(X, y, *sampled, batch_size)
                self.assertEqual(len(torch.unique(sampled[1])), min(batch_size, 5))

    def test_stratified_samples_are_reproducible(self):
        X, y = self.data([1, 8, 100, 100])
        classifier = self.classifier(24)
        torch.manual_seed(13)
        first = classifier._sample_data(X, y)
        torch.manual_seed(13)
        repeated = classifier._sample_data(X, y)
        for actual, expected in zip(repeated, first):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unstratified_sampling_preserves_original_random_selection(self):
        X, y = self.data([100, 2])
        for batch_size in (12, 128):
            with self.subTest(batch_size=batch_size):
                classifier = self.classifier(batch_size, stratify=False)
                torch.manual_seed(13)
                indices = torch.randperm(len(X))[:batch_size]
                expected_rng_state = torch.random.get_rng_state()
                torch.manual_seed(13)
                sampled = classifier._sample_data(X, y)
                torch.testing.assert_close(sampled[0], X[indices], rtol=0, atol=0)
                torch.testing.assert_close(sampled[1], y[indices], rtol=0, atol=0)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng_state))


if __name__ == '__main__':
    unittest.main()
