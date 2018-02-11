import pickle


def unpickle(filepath):
    """
    Returns unpickled version of given filepath
    :param filepath: filepath to object to unpickle
    :return: unpicked version of given filepath
    """
    with open(filepath, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def create_dataset_as_dict(files):
    """
    Concat files data in a unique dictionary containing 'data' and 'labels' fields
    :param files: files containing data as pickled object
    :return: A dict containing 'data' and 'labels' fields
    """
    files = files or []
    data = []
    labels = []
    for file in files:
        data_dict = unpickle(file)
        [data.append(img) for img in data_dict[b'data']]
        [labels.append(label) for label in data_dict[b'labels']]
    return {'data': data, 'labels': labels}


def load_training_set():
    """
    Load training data for CIFAR10
    :return: dictionnary with 'data' and 'labels' keys
    """
    train_data_files = ['data/data_batch_{}'.format(i) for i in range(1, 6)]
    return create_dataset_as_dict(train_data_files)


def load_test_set():
    """
    Load test data for CIFAR10
    :return: dictionnary with 'data' and 'labels' keys
    """
    test_data_file = ['data/test_batch']
    return create_dataset_as_dict(test_data_file)


def load_metadata():
    """
    Load metadata for CIFAR10
    :return: dictionnary with 'label_names' key
    """
    return unpickle('data/batches.meta')

