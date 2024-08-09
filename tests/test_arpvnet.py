import torch
import unittest
from arpvnet.models import ARPVNet

class TestARPVNet(unittest.TestCase):
    def setUp(self):
        self.model = ARPVNet(num_classes=20)
        self.input = torch.rand(2, 10000, 3)

    def test_forward(self):
        output = self.model(self.input)
        self.assertEqual(output.shape, (2, 10000, 20))

    def test_backward(self):
        output = self.model(self.input)
        loss = output.sum()
        loss.backward()
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

if __name__ == '__main__':
    unittest.main()
