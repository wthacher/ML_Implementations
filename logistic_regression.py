'''
Docstring for ML_Practice.logistic_regression


implement logistic regression model
Data: X n by d features
Y is n by 1 , for k class classification

model: p_k(x) = 


'''
import os

# Add these BEFORE your other imports and code
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


torch.set_num_threads(10)

class LR_model(nn.Module):
    def __init__(self,d,k):
        super(LR_model,self).__init__()
        self.layer1 = nn.Linear(d,k)

    def forward(self, X):
        X = X.view(X.size(0), -1)
        return self.layer1(X)
if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=64, 
        shuffle=True,
        num_workers=8,      # Parallelize data loading
        pin_memory=False,              # Not needed for CPU
        persistent_workers=True        # Keep workers alive between epochs
    )
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    model = LR_model(28*28, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=.01)


    for epoch in range(30):
        running_loss = 0.0
        total=0
        correct=0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images, labels
    
            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total
        
        print("train_acc: ",train_accuracy)
        
        print("loss: ",running_loss)


