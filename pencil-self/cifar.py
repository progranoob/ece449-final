import numpy as np
import torchvision
from PIL import Image

# Define noised CIFAR
def get_cifar10(root, train_ratio=0.9, asym=True, percent=0., train=True,
                transform_train=None, transform_val=None,
                download=False):
    base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
    # print('base_dataset',dir(base_dataset))

    train_idxs, val_idxs = train_val_split(base_dataset.targets,train_ratio)
    train_dataset = CIFAR10_train(root, train_idxs, percent=percent, train=train, transform=transform_train)
    if asym:
        train_dataset.asymmetric_noise()
    else:
        train_dataset.symmetric_noise()
    val_dataset = CIFAR10_val(root, val_idxs, transform=transform_val)

    print(f"Train: {len(train_idxs)} Val: {len(val_idxs)}")
    return train_dataset, val_dataset

def train_val_split(train_val,train_ratio):
    train_val = np.array(train_val)
    train_n = int(len(train_val) * train_ratio / 10)#5000
    train_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(train_val == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CIFAR10_train(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, percent=0., train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_train, self).__init__(root, train=train,
                                            transform=transform, target_transform=target_transform,
                                            download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            print(type(self.targets))
            self.targets = np.array(self.targets)[indexs]
            print(self.targets)
            self.true_labels= self.targets+1-1
        self.percent=percent
        self.prediction = np.zeros((len(self.data), 10, 10), dtype=np.float32)
        self.count = 0
        self.count_img=0
        self.ch_label=np.zeros(10,dtype=np.float32)

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.data))
        for i, idx in enumerate(indices):
            if i < self.percent * len(self.data):
                self.ch_label[self.targets[idx]]+=1
                self.targets[idx] = np.random.randint(10, dtype=np.int32)
            self.labels_update[idx][self.targets[idx]] = K

    def asymmetric_noise(self):
        for i in range(10):
            indices = np.where(self.true_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.percent * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.targets[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.targets[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.targets[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.targets[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.targets[idx] = 7

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.count_img+=1
        img, target = self.data[index], self.targets[index]
        true_labels = self.true_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # print('target',target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index, true_labels

    
class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        self.data = self.data[indexs]
        self.targets = np.array(self.targets)[indexs]
