import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio

features_save_dir = "features"
features_suffix = '_features_list.pt'

def get_dataloader_by_name(dataset_name, image_size):
    dataset = load_raw_image(dataset_name, image_size)
    if dataset_name == 'cifar100':
        batch_size = 300
    elif dataset_name == 'tiny-imageNet':
        batch_size = 400
    else:
        batch_size = 100
    dl = DataLoader(dataset, batch_size=batch_size)
    return dl



class CustomDataset(Dataset):
    def __init__(self, x, y, transform=None):
        '''
        :param x: if x is tensor then the shape is (N, 3, W, H) if x is ndarray then the shape is (N, W, H, 3)
        :param y:
        :param transform:
        '''
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]

        if self.transform is None:
            return img, label

        img = self.transform(img)
        return img, label


def load_raw_image(name, img_size):
    if name == 'cifar10':
        return cifar_dataset(img_size)
    elif name == 'stl10':
        return stl10_dataset(img_size)
    elif name == 'fashion_mnist':
        return fmnist_dataset(img_size)
    elif name == 'COIL20':
        return coil20_dataset(img_size)
    elif name == 'cifar100':
        return cifar100_dataset(img_size)
    elif name == 'tiny-imageNet':
        return tiny_imageNetdataset(img_size)
    elif name == 'ORL':
        return ORL_dataset(img_size)
    elif name == 'USPS':
        return usps_dataset(img_size)


def tiny_imageNetdataset(img_size):
    # 从 npz 文件加载字典
    loaded_data = np.load("../datasets/tiny_imageNet.npz")
    # 获取加载的数据
    loaded_dict = {key: loaded_data[key] for key in loaded_data.keys()}
    images = loaded_dict["images"]
    labels = loaded_dict["labels"]
    images = (images/255.).astype(np.float32)

    # 定义 ImageNet 标准化参数
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]

    # 创建图像转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为 PyTorch 的张量
        transforms.Resize(img_size),
        transforms.Normalize(mean=mean_values, std=std_values)
    ])

    dt = CustomDataset(images, labels, transform)

    return dt


def usps_dataset(img_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dt = datasets.USPS(
        root="../datasets",
        train=True,
        download=True,
        transform=transform,
    )

    # breakpoint()
    x = dt.data[:1000].reshape(1000, 1, 16, 16) / 255.
    x = np.repeat(x, 3, axis=1)
    x = np.transpose(x, (0, 2, 3, 1)).astype(np.float32)
    y = dt.targets[:1000]

    dt = CustomDataset(x, y, transform)


    return dt

def coil20_dataset(img_size):
    data = sio.loadmat('../datasets/COIL20.mat')
    x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
    x = np.transpose(x, (0, 2, 3, 1)) # numpy image is (H, W, C)
    x = np.repeat(x, 3, axis=-1)
    x = x.astype(np.float32)
    y = np.squeeze(y - 1)  # y in [0, 1, ...,

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dt = CustomDataset(x, y, transform)

    return dt

def ORL_dataset(img_size):
    data = sio.loadmat('../datasets/ORL_32x32.mat')
    x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
    x = np.transpose(x, (0, 2, 3, 1))  # numpy image is (H, W, C)
    x = np.repeat(x, 3, axis=-1)
    # breakpoint()
    x = x.astype(np.float32)/255.
    y = np.squeeze(y - 1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dt = CustomDataset(x, y, transform)

    return dt


def cifar100_dataset(img_size):
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    dt = datasets.CIFAR100(
        root="../datasets",
        download=True,
        train=True,
        transform=norm_transform
    )

    x = dt.data[:3000]/255.
    x = x.astype(np.float32)
    y = dt.targets[:3000]

    dt = CustomDataset(x, y, transform=norm_transform)

    return dt


def fmnist_dataset(img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dt = datasets.FashionMNIST(
        root="../datasets",
        download=True,
        train=True,
        transform=transform
    )

    x = dt.data[:1000].view(1000, 1, 28, 28) / 255.
    x = x.repeat(1, 3, 1, 1)
    y = dt.targets[:1000]
    dt = CustomDataset(x, y, transform)

    return dt


def stl10_dataset(img_size):
    norm_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    training_data = datasets.STL10(
        root="../datasets",
        download=True,
        split='train',
        transform=norm_transform,
    )

    return training_data


def cifar_dataset(img_size):
    norm_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    training_data = datasets.CIFAR10(
        root="../datasets",
        download=True,
        train=False,
        transform=norm_transform,
    )

    return training_data