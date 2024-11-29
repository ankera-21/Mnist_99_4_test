# MNIST_99_4_Test

## Objective
- Achieve 99.4 Validation(Test) accuracy on Mnist dataset with below constraints.
1. Total parameters should less than 20k
2. Less than 20 Epochs
3. Have used Batch Normalization and Drop Out

My Architecture:
```py
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
        x = self.dropout(x)
        x = self.conv1d_2(self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x))))))))
        x = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x))))))
        x = F.relu(self.bn7(self.conv1d_3(x)))
        x = x.view(x.size(0), -1) 
        return F.log_softmax(F.relu(self.fc(x)))


```

```py

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))


----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
            Conv2d-3           [-1, 32, 28, 28]           4,640
       BatchNorm2d-4           [-1, 32, 28, 28]              64
         MaxPool2d-5           [-1, 32, 14, 14]               0
            Conv2d-6            [-1, 8, 14, 14]             264
           Dropout-7            [-1, 8, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           1,168
       BatchNorm2d-9           [-1, 16, 14, 14]              32
           Conv2d-10           [-1, 32, 14, 14]           4,640
      BatchNorm2d-11           [-1, 32, 14, 14]              64
        MaxPool2d-12             [-1, 32, 7, 7]               0
           Conv2d-13              [-1, 8, 7, 7]             264
           Conv2d-14             [-1, 16, 5, 5]           1,168
      BatchNorm2d-15             [-1, 16, 5, 5]              32
           Conv2d-16             [-1, 32, 3, 3]           4,640
      BatchNorm2d-17             [-1, 32, 3, 3]              64
           Conv2d-18              [-1, 8, 3, 3]             264
      BatchNorm2d-19              [-1, 8, 3, 3]              16
           Linear-20                   [-1, 10]             730
================================================================
Total params: 18,242
Trainable params: 18,242
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.82
Params size (MB): 0.07
Estimated Total Size (MB): 0.89
----------------------------------------------------------------

```

- Created Github Action to test below test conditions

1. Total Parameter Count Test
2. Use of Batch Normalization
3. Use of DropOut
4. Use of Fully Connected Layer 

## Training Logs

```log
loss=0.20113474130630493 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.64it/s]

Test set: Average loss: 0.0538, Accuracy: 9871/10000 (98.7%)

loss=0.14662954211235046 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.72it/s]

Test set: Average loss: 0.0397, Accuracy: 9870/10000 (98.7%)

loss=0.10597454756498337 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.78it/s]

Test set: Average loss: 0.0316, Accuracy: 9910/10000 (99.1%)

loss=0.14412657916545868 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.82it/s]

Test set: Average loss: 0.0299, Accuracy: 9916/10000 (99.2%)

loss=0.0729445368051529 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.38it/s]

Test set: Average loss: 0.0229, Accuracy: 9928/10000 (99.3%)

loss=0.003632587380707264 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.00it/s]

Test set: Average loss: 0.0267, Accuracy: 9927/10000 (99.3%)

loss=0.08136429637670517 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.42it/s]

Test set: Average loss: 0.0310, Accuracy: 9913/10000 (99.1%)

loss=0.02216625027358532 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.01it/s]

Test set: Average loss: 0.0208, Accuracy: 9925/10000 (99.2%)

loss=0.1248861625790596 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 15.08it/s]

Test set: Average loss: 0.0204, Accuracy: 9935/10000 (99.3%)

loss=0.03354756906628609 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.89it/s]

Test set: Average loss: 0.0214, Accuracy: 9931/10000 (99.3%)

loss=0.01820302940905094 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.36it/s]

Test set: Average loss: 0.0190, Accuracy: 9945/10000 (99.5%)

loss=0.03332187235355377 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.75it/s]

Test set: Average loss: 0.0192, Accuracy: 9939/10000 (99.4%)

loss=0.050290632992982864 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.74it/s]

Test set: Average loss: 0.0309, Accuracy: 9901/10000 (99.0%)

loss=0.006051718723028898 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.44it/s]

Test set: Average loss: 0.0179, Accuracy: 9940/10000 (99.4%)

loss=0.009286673739552498 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.85it/s]

Test set: Average loss: 0.0217, Accuracy: 9932/10000 (99.3%)

loss=0.0931750237941742 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.62it/s]

Test set: Average loss: 0.0208, Accuracy: 9933/10000 (99.3%)

loss=0.026775948703289032 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.66it/s]

Test set: Average loss: 0.0172, Accuracy: 9941/10000 (99.4%)

loss=0.05636361613869667 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.14it/s]

Test set: Average loss: 0.0165, Accuracy: 9949/10000 (99.5%)

loss=0.05541451275348663 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.85it/s]

Test set: Average loss: 0.0152, Accuracy: 9954/10000 (99.5%)

```
