from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 3, padding=1) #input -? OUtput? RF
#         #nn.BatchNorm2d(16)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv1d_1 =  nn.Conv2d(32,8, 1)
#         self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(16)
#         self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
#         self.bn4 = nn.BatchNorm2d(32)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv1d_2 =  nn.Conv2d(32,8,1,stride=1)
#         self.conv5 = nn.Conv2d(8, 16, 3)
#         self.bn5 = nn.BatchNorm2d(16)
#         self.conv6 = nn.Conv2d(16, 32, 3)
#         self.bn6 = nn.BatchNorm2d(32)
#         self.conv1d_3 =  nn.Conv2d(32,8,1,stride=1)
#         self.bn7 = nn.BatchNorm2d(8)
#         self.fc = nn.Linear(72,10)
        

#     def forward(self, x):
#         x = self.conv1d_1(self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))))
#         #return x
#         x = nn.Dropout(0.001)(x)
#         x = self.conv1d_2(self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x))))))))
#         x = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x))))))
#         x = F.relu(self.bn7(self.conv1d_3(x)))
#         x = x.view(x.size(0), -1) 
#         return F.log_softmax(F.relu(self.fc(x)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) #input -? OUtput? RF
        #nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1d_1 =  nn.Conv2d(32,8, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv1d_2 =  nn.Conv2d(32,8,1,stride=1)
        self.conv5 = nn.Conv2d(8, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 32, 3)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv1d_3 =  nn.Conv2d(32,8,1,stride=1)
        self.bn7 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(72,10)
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x):
        x = self.conv1d_1(self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))))
        #return x
        x = self.dropout(x)
        x = self.conv1d_2(self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x))))))))
        x = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x))))))
        x = F.relu(self.bn7(self.conv1d_3(x)))
        x = x.view(x.size(0), -1) 
        return F.log_softmax(F.relu(self.fc(x)))
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
if __name__=='__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    #summary(model, input_size=(1, 28, 28))
    torch.manual_seed(1)
    batch_size = 128

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    

    # Update the data loaders to include data augmentation
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomRotation(10),  # Rotate by up to 10 degrees
                            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 20):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
