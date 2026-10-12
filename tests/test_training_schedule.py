import unittest
from unittest.mock import patch

import torch

from hyperfast.utils import MainNetworkTrainable, fine_tune_main_network
from test_bias_finetuning import make_config, make_data, make_network_parts


class TrainingScheduleTests(unittest.TestCase):
    def test_plateau_without_clipping_reproduces_original_optimizer_updates(self):
        X, y = make_data()

        def original_loop():
            model = MainNetworkTrainable(make_config(), 2, *make_network_parts(),
                                         torch.tensor([.7, .9])).train()
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, y), batch_size=3, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=.1, patience=10)
            step = 0
            while step < 7:
                for inputs, targets in loader:
                    optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(inputs, targets), targets)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss.item())
                    step += 1
                    if step == 7:
                        break
            return model

        torch.manual_seed(42)
        expected = original_loop()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        with patch('torch.nn.utils.clip_grad_norm_', side_effect=AssertionError):
            parts = fine_tune_main_network(
                make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
                'cpu', 7, 3, finetuning_dropout=0, gradient_clip_norm=None,
                finetuning_scheduler='plateau')
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        actual = MainNetworkTrainable(make_config(), 2, *parts)
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)

    def test_plateau_still_restores_initial_validation_checkpoint(self):
        X, y = make_data()
        info = {}
        with patch('sklearn.metrics.roc_auc_score', side_effect=[.8, .7]):
            parts = fine_tune_main_network(
                make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
                'cpu', 64, 4, X_val=X + .1, y_val=y, reference_X=X, reference_y=y,
                finetuning_scheduler='plateau', finetuning_dropout=0,
                gradient_clip_norm=None, patience_epochs=1, cooldown_checks=0,
                training_info=info)
        self.assertEqual(info['best_step'], 0)
        self.assertEqual(info['stop_reason'], 'patience')
        self.assertEqual(info['steps'], 1)
        torch.testing.assert_close(parts[0][0].weight, torch.eye(2), rtol=0, atol=0)

    def test_unknown_scheduler_fails(self):
        X, y = make_data()
        with self.assertRaisesRegex(ValueError, 'finetuning_scheduler'):
            fine_tune_main_network(
                make_config(), X, y, 2, *make_network_parts(), torch.tensor([.7, .9]),
                'cpu', 1, 4, finetuning_scheduler='unknown')


if __name__ == '__main__':
    unittest.main()
