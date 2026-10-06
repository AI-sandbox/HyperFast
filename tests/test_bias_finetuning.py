import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from hyperfast.utils import MainNetworkTrainable, NN, fine_tune_main_network


def make_network_parts(device="cpu"):
    rf = nn.Sequential(nn.Linear(2, 2, bias=False)).to(device)
    with torch.no_grad():
        rf[0].weight.copy_(torch.eye(2, device=device))
    pca = SimpleNamespace(
        mean_=torch.zeros(2, device=device),
        components_=torch.eye(2, device=device),
    )
    layers = [
        (torch.eye(2, device=device), torch.tensor([0.1, 0.2], device=device)),
        (torch.tensor([[0.3, 0.2], [0.1, 0.4]], device=device), torch.zeros(2, device=device)),
        (torch.tensor([[0.1, 0.3], [-0.2, 0.2]], device=device), torch.zeros(2, device=device)),
    ]
    return rf, pca, layers


def make_config(device="cpu"):
    return SimpleNamespace(
        clip_data_value=10.0, nn_bias=True, device=device, torch_pca=True, lr=0.01
    )


def make_data(device="cpu"):
    return (
        torch.tensor([[0.1, 0.2], [0.2, 0.3], [1.0, 1.1], [1.1, 1.2]], device=device),
        torch.tensor([0, 1, 0, 1], device=device),
    )


class FineTuningTests(unittest.TestCase):
    def test_singleton_minibatch_skips_nearest_neighbor_bias(self):
        model = MainNetworkTrainable(
            make_config(), 2, *make_network_parts(), torch.tensor([0.7, 0.9])
        )
        X, y = make_data()
        with patch.object(NN, "predict_from_training_with_LOO", side_effect=AssertionError):
            actual = model(X[:1], y[:1])
        model.use_nn_bias = False
        expected = model(X[:1], y[:1])
        torch.testing.assert_close(actual, expected)
        nn.functional.cross_entropy(actual, y[:1]).backward()
        self.assertIsNotNone(model.layers[-1].weight.grad)
        self.assertIsNone(model.nn_bias.grad)

    def check_bias_update(self, device, transferred=False):
        cached_bias = nn.Parameter(torch.tensor([0.7, 0.9], device=device))
        original = cached_bias.detach().clone()
        input_bias = cached_bias.cpu() if transferred else cached_bias
        parts_device = "cpu" if transferred else device
        model = MainNetworkTrainable(
            make_config(device), 2, *make_network_parts(parts_device), input_bias
        ).to(device)
        self.assertEqual(model.nn_bias.device.type, torch.device(device).type)
        self.assertIsInstance(model.nn_bias, nn.Parameter)
        self.assertIs(dict(model.named_parameters())["nn_bias"], model.nn_bias)
        self.assertNotEqual(model.nn_bias.data_ptr(), input_bias.data_ptr())
        X, y = make_data(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = model.nn_bias.detach().clone()
        loss = nn.functional.cross_entropy(model(X, y), y)
        loss.backward()
        self.assertIsNotNone(model.nn_bias.grad)
        self.assertTrue(torch.isfinite(model.nn_bias.grad).all().item())
        self.assertGreater(model.nn_bias.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(model.nn_bias.detach(), before))
        torch.testing.assert_close(cached_bias.detach(), original, rtol=0, atol=0)
        self.assertIsNone(cached_bias.grad)

    def test_bias_is_registered_updated_and_isolated(self):
        self.check_bias_update("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_bias_is_registered_updated_and_isolated_cuda(self):
        self.check_bias_update("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_bias_remains_trainable_after_cuda_to_cpu_transfer(self):
        self.check_bias_update("cuda", transferred=True)

    def test_tensor_bias_is_registered_as_parameter(self):
        input_bias = torch.tensor([0.7, 0.9], requires_grad=True) * 2
        model = MainNetworkTrainable(make_config(), 2, *make_network_parts(), input_bias)
        self.assertIsInstance(model.nn_bias, nn.Parameter)
        self.assertTrue(model.nn_bias.is_leaf)
        self.assertIs(dict(model.named_parameters())["nn_bias"], model.nn_bias)
        self.assertNotEqual(model.nn_bias.data_ptr(), input_bias.data_ptr())

    def test_fine_tuning_handles_singleton_final_minibatch(self):
        torch.manual_seed(7)
        X, y = make_data()
        X = torch.cat([X, torch.tensor([[1.4, 1.8]])])
        y = torch.cat([y, torch.tensor([1])])
        input_bias = nn.Parameter(torch.tensor([0.7, 0.9]))
        original = input_bias.detach().clone()
        parts = fine_tune_main_network(
            make_config(), X, y, 2, *make_network_parts(), input_bias,
            device="cpu", optimize_steps=3, batch_size=2,
        )
        trained_bias = parts[-1]
        self.assertIsInstance(trained_bias, nn.Parameter)
        self.assertTrue(torch.isfinite(trained_bias).all().item())
        self.assertFalse(torch.equal(trained_bias.detach(), original))
        torch.testing.assert_close(input_bias.detach(), original, rtol=0, atol=0)
        self.assertIsNone(input_bias.grad)

    def test_fine_tuning_cycles_all_rows_with_fixed_update_count(self):
        torch.manual_seed(7)
        X, y = make_data()
        X = torch.cat([X, torch.tensor([[1.4, 1.8]])])
        y = torch.cat([y, torch.tensor([1])])
        batches = []
        forward = MainNetworkTrainable.forward

        def capture(model, inputs, targets):
            batches.append((inputs.detach().clone(), targets.detach().clone()))
            return forward(model, inputs, targets)

        with patch.object(MainNetworkTrainable, 'forward', new=capture), \
                patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau,
                             'step', autospec=True) as schedule:
            fine_tune_main_network(
                make_config(), X, y, 2, *make_network_parts(), torch.tensor([0.7, 0.9]),
                device='cpu', optimize_steps=7, batch_size=2)
        self.assertEqual([len(inputs) for inputs, _ in batches], [2, 2, 1, 2, 2, 1, 2])
        self.assertEqual(schedule.call_count, 7)
        for start in (0, 3):
            inputs = torch.cat([b[0] for b in batches[start:start + 3]])
            targets = torch.cat([b[1] for b in batches[start:start + 3]])
            order = inputs[:, 0].argsort()
            torch.testing.assert_close(inputs[order], X)
            torch.testing.assert_close(targets[order], y)

    def test_fine_tuning_stops_before_completing_an_epoch(self):
        X, y = make_data()
        forward = MainNetworkTrainable.forward
        seen = []

        def capture(model, inputs, targets):
            seen.append(len(inputs))
            return forward(model, inputs, targets)

        with patch.object(MainNetworkTrainable, 'forward', new=capture):
            fine_tune_main_network(
                make_config(), X, y, 2, *make_network_parts(), torch.tensor([0.7, 0.9]),
                device='cpu', optimize_steps=1, batch_size=2)
        self.assertEqual(seen, [2])


if __name__ == "__main__":
    unittest.main()
