import os
import sys
import unittest
import mock

from cifar import load
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestBase(unittest.TestCase):

    @mock.patch('cifar.load.pickle')
    def test_unpickle_should_call_pickle_load(self, mock_pickle):
        with mock.patch('cifar.load.open') as mocked_open:
            # Given
            mocked_file = StringIO('foo')
            mocked_open.return_value = mocked_file
            load.unpickle('file to unpickle')

            # Then
            mock_pickle.load.assert_called_with(mocked_file, encoding='bytes')

    def test_unpickle_with_non_existent_file_should_raise_filenotfounderror(self):
        with self.assertRaises(FileNotFoundError):
            load.unpickle('file that does not exist')

    def test_create_dataset_as_dict_should_return_dict_with_empty_values_when_args_is_an_empty_list(self):
        # Given
        data = load.create_dataset_as_dict([])
        expected_res = {'data': [], 'labels': []}

        # Then
        self.assertDictEqual(data, expected_res)

    def test_create_dataset_as_dict_should_return_dict_with_empty_values_when_args_is_None(self):
        # Given
        data = load.create_dataset_as_dict(None)
        expected_res = {'data': [], 'labels': []}

        # Then
        self.assertDictEqual(data, expected_res)

    @mock.patch('cifar.load.unpickle')
    def test_create_dataset_as_dict_should_unpickle_every_given_files(self, mock_unpickle):
        # Given
        mock_unpickle.return_value = {b'data': [], b'labels': []}
        files = ['data/data_batch_{}'.format(i) for i in range(1, 3)]
        expected_calls = [mock.call(file) for file in files]
        load.create_dataset_as_dict(files)

        # Then
        for file in files:
            mock_unpickle.assert_has_calls(expected_calls)

    def test_create_dataset_as_dict_append_input_files_data_and_labels_values(self):
        # Given
        with mock.patch('cifar.load.unpickle') as mock_unpickle:
            data_len = 10
            mock_unpickle.return_value = {b'data': [1] * data_len,
                                          b'labels': [0] * data_len}
            nb_files = 20
            files = ['sample file'] * 20
            expected_data_len = data_len * nb_files
            expected_labels_len = data_len * nb_files
            data = load.create_dataset_as_dict(files)

            # Then
            self.assertEqual(expected_data_len, len(data['data']))
            self.assertEqual(expected_labels_len, len(data['labels']))


if __name__ == '__main__':
    unittest.main()
