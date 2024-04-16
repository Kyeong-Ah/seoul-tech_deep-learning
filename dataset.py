import os
import io
import tarfile
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081))
        ])
        # self.train_tar =  tarfile.open(os.path.join(data_dir, 'train.tar'))
        # self.test_tar =  tarfile.open(os.path.join(data_dir, 'test.tar'))
        # self.all_filenames = self.train_tar.getnames() + self.test_tar.getnames()

        # 이미지 불러오기
        self.data = []
        self.labels = []

        with tarfile.open(data_dir, 'r') as tar:
            members = [member for member in tar.getmembers() if member.name.endswith('.png')]

            for member in members:
                img = tar.extractfile(member)
                img = Image.open(img)
                img = self.transform(img)
                label = int(member.name.split('_')[1].split('.')[0])

                self.data.append(img)
                self.labels.append(label)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]
        label = self.labels[idx]
        
        return img, label


if __name__ == '__main__':
    train_data = MNIST(data_dir='./data')
    test_data = MNIST(data_dir='./data')

    for i in range(len(10)):
        img, label = train_data[i]
        print(f"Image shape: {img.shape}, Label: {label}")


