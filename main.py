"""
PyTorch-XNOR-GEMM-Extention
Authors: Tairen (tairenpiao@gmail.com)
This code can be used only for research purposes.
For other purposes (e.g., commercial), please contact me.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict

import binarized_modules as bi


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(784,1024)
        self.fc1 = nn.Linear(1024,768)
        self.fc2 = bi.BinarizeLinear_training(768,512)
        # self.fc2 = bi.BinarizeLinear_inference(24,512) # Because we column encode the right matrix

        self.fcc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fcc(x)
        output = nn.functional.log_softmax(x, dim=1)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()

        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        # model.fc2.weight.grad = model.fc2.quantized_weight.grad
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLossW:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss.item()))
        if loss<0.009:
            break


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.view(-1, 784)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.004, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)  

    # num_p = sum(p.numel() for p in model.parameters())
    # print("number of parameters",num_p)

    test(args, model, device, test_loader)


    if args.save_model:
        print('saving...')
        torch.save(model,'./model.pt')
        torch.save(model.state_dict(), "./model_dict.pt")

    # for k,v in model.state_dict().items():
    #     print(k,v.shape)

    # model = torch.load('./mnist.pt')
    # # state_dict_quant = OrderedDict()
    # for k,v in model.items():
    #     v = v.to(dtype=torch.half)
    #     print(v.dtype)
    #     print(k,v)
    #     if k == 'fc1.weight' or k =='fc2.weight' or k=='fc2.bias' or k =='fc1.bias' or k =='fc3.weight' or k =='fc3.bias':
    #         v = Binary(v)
    #     state_dict_quant[k] = v
    #     print(state_dict_quant[k])


if __name__ == '__main__':
    main()