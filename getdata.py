import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os
import gzip
class GetDataSet():
    def __init__(self, dataSetName):
        self.dataSetName = dataSetName

        self.trainData = None
        self.trainLabel = None
        self.trainDataSize = None

        self.testData = None
        self.testLabel = None
        self.testDataSize = None

        if self.dataSetName == 'MNIST' or self.dataSetName == 'mnist':
            self.mnistDataDistribution()
        elif self.dataSetName == 'CIFAR10':
            self.cafar10DataDistribution()

    # def mnistDataDistribution(self, isIID):
    #
    #     trainingData = datasets.CIFAR10(
    #         root="data",
    #         train=True,
    #         download=True,
    #         transform=ToTensor(),
    #     )
    #     trainData = []
    #     trainLabel = []
    #     for X, y in trainingData:
    #         trainData.append(X.tolist())
    #         trainLabel.append(y)
    #     self.trainDataSize = len(trainData)
    #     # ----------------------------------------------------------- #
    #     testingData = datasets.CIFAR10(
    #         root="data",
    #         train=False,
    #         download=True,
    #         transform=ToTensor(),
    #     )
    #     testData = []
    #     testLabel = []
    #     for X, y in testingData:
    #         testData.append(X.tolist())
    #         testLabel.append(y)
    #     self.testDataSize = len(testData)
    #     self.testData = torch.tensor(testData)
    #     self.testLabel = torch.tensor(testLabel)
    #     # ----------------------------------------------------------- #
    #
    #     if isIID == True:
    #         self.trainData = torch.tensor(trainData)
    #         self.trainLabel = torch.tensor(trainLabel)
    #         print(1)
    #
    #     else:
    #         trainDataT = np.array(trainData, dtype='float32')
    #         trainLabelT = np.array(trainLabel, dtype='int64')
    #         self.trainData = trainDataT
    #         self.trainLabel = trainLabelT
    #     print(self.trainData.shape)

    def mnistDataDistribution(self, ):

        data_dir = r'./data/MNIST/raw'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)

        # print(train_images.shape) # 图片的形状 (60000, 28, 28, 1) 60000张 28 * 28 * 1  灰色一个通道
        # print('-' * 22 + "\n")
        train_labels = self.extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape)  # label shape (60000, 10)
        # print('-' * 22 + "\n")
        test_images = self.extract_images(test_images_path)
        test_labels = self.extract_labels(test_labels_path)


        # assert train_images.shape[0] == train_labels.shape[0]
        # assert test_images.shape[0] == test_labels.shape[0]
        #
        #
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #
        # assert train_images.shape[3] == 1
        # assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])

        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        self.trainData = train_images
        self.trainLabel = np.argmax(train_labels == 1, axis = 1)
        self.testData = test_images
        self.testLabel = np.argmax(test_labels == 1, axis = 1)
        print(self.trainData.shape)



    def extract_images(self, filename):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')

        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_labels(self, filename):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return self.dense_to_one_hot(labels)

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    def cafar10DataDistribution():

        return

#
# g = GetDataSet("MNIST")
# print(g.trainData)
# print(g.trainLabel)
