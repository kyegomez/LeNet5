from torch import optim
from torch import nn
from lenet.model import LeNet5 as model
from lenet.model import device

loss = nn.CrossEntropyLoss() # init cross entropy 
optim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # init stochastic gradient descent with the model parameters, don't forget to init the () after parameters, learning rate = 0.001, momentum factor = 0.9 

EPOCHS = 10 # iterations of the whole dataset

for epoch in range(EPOCHS): # for loop
    epoch_loss = 0.0 #initial loss
    
    for inputs, labels in trainloader: #for the inputs and labels in trainloader
        inputs = inputs.to(device) # set the inputs to the device
        labels = labels.to(device) # set the labels to device

        optim.zero_grad() # init the optimizer with 0 gradients

        outputs = model(inputs) #input the tensors into the model
        loss = loss(outputs, labels) #apply the loss to the models outputs with the labels
        loss.backward() #then backtrack the loss
        optim.step # activate optim step

        epoch_loss += loss.item() # the epoch loss += the loss reported by the model
    print(f" Epoch: {epoch} LOSS: {epoch_loss / len(trainloader)}")

