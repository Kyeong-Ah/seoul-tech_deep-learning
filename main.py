from dataset import MNIST
from model import LeNet5, LeNet5_Dropout, CustomMLP
# import some packages you need here
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(trn_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    trn_loss = train_loss / len(trn_loader)
    acc = 100. * correct / total

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tst_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    tst_loss = test_loss / len(tst_loader)
    acc = 100. * correct / total

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    # 데이터셋과 DataLoader 설정
    train_dataset = MNIST(data_dir='./data/train.tar')
    test_dataset = MNIST(data_dir='./data/test.tar')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 모델, 손실 함수, 옵티마이저 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_lenet5 = LeNet5().to(device)
    model_lenet5_dropout = LeNet5_Dropout().to(device)
    model_mlp = CustomMLP().to(device)
    criterion = nn.CrossEntropyLoss() # cost function
    optimizer_lenet5 = optim.SGD(model_lenet5.parameters(), lr=0.01, momentum=0.9)
    optimizer_lenet5_dropout = optim.SGD(model_lenet5_dropout.parameters(), lr=0.01, momentum=0.9)
    optimizer_mlp = optim.SGD(model_mlp.parameters(), lr=0.01, momentum=0.9)

    # plotting 위해 loss, acc 저장할 리스트 선언
    train_LeNet5 = []
    test_LeNet5 = []
    train_LeNet5_dropout = []
    test_LeNet5_dropout = []
    train_MLP = []
    test_MLP = []

    # 학습 및 테스트
    for epoch in range(30):  # 30 에포크 동안 학습
        # LeNet-5
        trn_loss, trn_acc = train(model_lenet5, train_loader, device, criterion, optimizer_lenet5)
        tst_loss, tst_acc = test(model_lenet5, test_loader, device, criterion)
        train_LeNet5.append((trn_loss, trn_acc))
        test_LeNet5.append((tst_loss, tst_acc))
        print(f"LeNet-5 Epoch: {epoch+1}, Train Loss: {trn_loss:.4f}, Train Acc: {trn_acc:.2f}%, Test Loss: {tst_loss:.4f}, Test Acc: {tst_acc:.2f}%", flush=True)

        # LeNet-5 Dropout
        trn_loss, trn_acc = train(model_lenet5_dropout, train_loader, device, criterion, optimizer_lenet5_dropout)
        tst_loss, tst_acc = test(model_lenet5_dropout, test_loader, device, criterion)
        train_LeNet5_dropout.append((trn_loss, trn_acc))
        test_LeNet5_dropout.append((tst_loss, tst_acc))
        print(f"LeNet-5_Dropout Epoch: {epoch+1}, Train Loss: {trn_loss:.4f}, Train Acc: {trn_acc:.2f}%, Test Loss: {tst_loss:.4f}, Test Acc: {tst_acc:.2f}%", flush=True)
        
        # Custom MLP
        trn_loss, trn_acc = train(model_mlp, train_loader, device, criterion, optimizer_mlp)
        tst_loss, tst_acc = test(model_mlp, test_loader, device, criterion)
        train_MLP.append((trn_loss, trn_acc))
        test_MLP.append((tst_loss, tst_acc))
        print(f"Custom MLP Epoch: {epoch+1}, Train Loss: {trn_loss:.4f}, Train Acc: {trn_acc:.2f}%, Test Loss: {tst_loss:.4f}, Test Acc: {tst_acc:.2f}%", flush=True)
    
    # plotting
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    num_epochs = 31

    # LeNet5 plotting
    axs[0, 0].plot(train_LeNet5[0], label='Train Loss')
    axs[0, 0].plot(test_LeNet5[0], label='Test Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('LeNet5 - Loss Curve')
    axs[0,0].set_xticks(range(1,num_epochs))
    axs[0, 0].legend()

    axs[0, 1].plot(train_LeNet5[1], label='Train Accuracy')
    axs[0, 1].plot(test_LeNet5[1], label='Test Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_title('LeNet5 - Accuracy Curve')
    axs[0,1].set_xticks(range(1,num_epochs))
    axs[0, 1].legend()

    # LeNet5WithDropout plotting
    axs[1, 0].plot(train_LeNet5_dropout[0], label='Train Loss')
    axs[1, 0].plot(test_LeNet5_dropout[0], label='Test Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_title('LeNet5WithDropout - Loss Curve')
    axs[1,0].set_xticks(range(1,num_epochs))
    axs[1, 0].legend()

    axs[1, 1].plot(train_LeNet5_dropout[1], label='Train Accuracy')
    axs[1, 1].plot(test_LeNet5_dropout[1], label='Test Accuracy')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].set_title('LeNet5WithDropout - Accuracy Curve')
    axs[1,1].set_xticks(range(1,num_epochs))
    axs[1, 1].legend()

    # CustomMLP plotting
    axs[2, 0].plot(train_MLP[0], label='Train Loss')
    axs[2, 0].plot(test_MLP[0], label='Test Loss')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Loss')
    axs[2, 0].set_title('CustomMLP - Loss Curve')
    axs[2,0].set_xticks(range(1,num_epochs))
    axs[2, 0].legend()

    axs[2, 1].plot(train_MLP[1], label='Train Accuracy')
    axs[2, 1].plot(test_MLP[1], label='Test Accuracy')
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Accuracy')
    axs[2, 1].set_title('CustomMLP - Accuracy Curve')
    axs[2,1].set_xticks(range(1,num_epochs))
    axs[2, 1].legend()

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
    exit()

