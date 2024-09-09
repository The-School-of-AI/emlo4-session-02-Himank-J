import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Define your model architecture here
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # TODO: Define the forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # TODO: Implement the training loop here
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} '
                  f'({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test_epoch(model, device, data_loader):
    # TODO: Implement the testing loop here
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


def main():
    # Parser to get command line arguments
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    # TODO: Define your command line arguments here
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from checkpoint')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # TODO: Load the MNIST dataset for training and testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = Net().to(device)
    # TODO: Choose and define the optimizer here
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # TODO: Add a way to load the model checkpoint if 'resume' argument is True
    start_epoch = 1
    if args.resume:
        checkpoint_path = 'model_checkpoint.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    # TODO: Implement the training and testing cycles
    # Hint: Save the model after each epoch
    for epoch in range(start_epoch, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)
        accuracy = test_epoch(model, device, test_loader)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
        }, 'model_checkpoint.pth')

if __name__ == "__main__":
    main()
