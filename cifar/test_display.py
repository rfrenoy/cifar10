import matplotlib.pyplot as plt
import numpy as np
from cifar import load
from cifar.NearestNeighbor import NearestNeighbor


def printable_format(img):
    printable = np.reshape(img, (3, 1024))
    printable = list(zip(*printable))
    return np.reshape(printable, (32, 32, 3))


def main():
    train_dataset = load.load_training_set()
    test_set = load.load_test_set()
    #img = train_dataset['data'][250]
    #plt.imshow(printable_format(img))
    #plt.show()
    nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
    nn.train(np.array(train_dataset['data']), np.array(train_dataset['labels']))  # train the classifier on the training images and labels
    y_pred = nn.predict(np.array(test_set['data']))  # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print('accuracy: %f' % (np.mean(y_pred == test_set['labels'])))


if __name__ == '__main__':
    main()
