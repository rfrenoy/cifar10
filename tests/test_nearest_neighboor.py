import unittest
import numpy as np
from cifar.NearestNeighbor import NearestNeighbor


class TestNearestNeighbor(unittest.TestCase):
    def test_predict_on_empty_example_list_return_empty_prediction_list(self):
        # Given
        nn = NearestNeighbor()
        nn.train(np.array([1]), np.array([1]))
        y_pred = list(nn.predict(np.array([])))

        # Then
        self.assertListEqual(y_pred, [])

    def test_predict_on_untrained_model_raise_exception(self):
        # Given
        nn = NearestNeighbor()

        # Then
        with self.assertRaises(Exception):
            list(nn.predict(np.array([])))

    def test_predict_on_model_trained_with_empty_list_of_examples_raise_exception(self):
        # Given
        nn = NearestNeighbor()

        # Then
        with self.assertRaises(Exception):
            list(nn.predict(np.array([])))
