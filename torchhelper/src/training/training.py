import torch
from torch import nn, optim

import matplotlib.pyplot as plt

def classification(model, train, test, criterion=nn.CrossEntropyLoss(), opt=None, epochs=25, log=True, graph=False):
    if opt is None:
        opt = optim.Adam(model.parameters(), lr=0.01)
    
    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for imgs, labs in train:
            opt.zero_grad()
            logps = model.forward(imgs)
            loss = criterion(logps, labs)
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for images, labels in test:
                    logps = model.forward(images)
                    test_loss += criterion(logps, labels)
                    
                    _, top_class = logps.topk(1, dim=1)
                    equals = ((top_class == labels.view(*top_class.shape)))
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                model.train()
            
            train_losses.append(running_loss/len(train))
            test_losses.append(test_loss/len(test))
                    
            if log:
                print(f'Epoch: {e+1}/{epochs}')
                print(f'Training loss: {train_losses[-1]}')
                print(f'Testing loss: {test_losses[-1]}')
                print(f'Accuracy: {accuracy/len(test)}')
        
    if graph:
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Testing Loss')
        plt.legend(frameon=False)
        plt.show()
            
    return (train_losses, test_losses)

def regression(model, train, test, criterion=nn.MSELoss(), opt=None, epochs=25, log=True, graph=False):
    if opt is None:
        opt = optim.Adam(model.parameters(), lr=0.01)
    
    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for imgs, labs in train:
            opt.zero_grad()
            logps = model.forward(imgs)
            loss = criterion(logps, labs)
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
        else:
            test_loss = 0
            with torch.no_grad():
                model.eval()
                for images, labels in test:
                    logps = model.forward(images)
                    test_loss += criterion(logps, labels)
                model.train()
            
            train_losses.append(running_loss/len(train))
            test_losses.append(test_loss/len(test))
                    
            if log:
                print(f'Epoch: {e+1}/{epochs}')
                print(f'Training loss: {train_losses[-1]}')
                print(f'Testing loss: {test_losses[-1]}')
        
    if graph:
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Testing Loss')
        plt.legend(frameon=False)
        plt.show()
            
    return (train_losses, test_losses)
