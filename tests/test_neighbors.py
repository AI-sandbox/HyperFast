import unittest

import torch

from hyperfast.utils import NN


class LeaveOneOutTests(unittest.TestCase):
    def check_leave_one_out(self, device):
        for n_rows in (2, 127, 128, 129, 259):
            with self.subTest(device=device, n_rows=n_rows):
                points = torch.arange(n_rows, device=device, dtype=torch.float32)[:, None]
                labels = torch.arange(n_rows, device=device)
                predictor = NN(points, labels)
                batched = predictor.predict_from_training_with_LOO(mini_batches=True)
                full = predictor.predict_from_training_with_LOO(mini_batches=False)
                torch.testing.assert_close(batched, full, rtol=0, atol=0)
                self.assertTrue(torch.all(batched != labels).item())
                self.assertEqual(batched[-1].item(), n_rows - 2)

    def test_batched_leave_one_out_matches_full(self):
        self.check_leave_one_out("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_batched_leave_one_out_matches_full_cuda(self):
        self.check_leave_one_out("cuda")

    def test_leave_one_out_rejects_fewer_than_two_rows(self):
        for n_rows in (0, 1):
            for mini_batches in (False, True):
                with self.subTest(n_rows=n_rows, mini_batches=mini_batches):
                    predictor = NN(torch.zeros(n_rows, 2), torch.zeros(n_rows, dtype=torch.long))
                    with self.assertRaisesRegex(ValueError, "at least two"):
                        predictor.predict_from_training_with_LOO(mini_batches=mini_batches)


if __name__ == "__main__":
    unittest.main()
