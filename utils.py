from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim

def imsave(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('test.png')

def get_model(state_dict=None):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1000, bias=True),
        nn.Sigmoid(),
        nn.Linear(1000, 10, bias=False)
    )

    if state_dict is None:
        return model

    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict)

    model.load_state_dict(state_dict)
    model.eval()
    return model

def copy_model(model):
    tmp_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1000, bias=True),
        nn.Sigmoid(),
        nn.Linear(1000, 10, bias=False)
    )

    tmp_model.load_state_dict(model.state_dict())
    return tmp_model

def train(model, trainloader, writer, max_epoch, device, verbose=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(max_epoch):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(
                device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i % 1000 == 999) and verbose:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                writer.add_scalar('training loss',
                                  running_loss / 1000,
                                  epoch * len(trainloader) + i)
                running_loss = 0.0

def eval(model, classes, testloader, device, verbose=False):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels.squeeze())
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if verbose:
        print('Accuracy of the network on the 10000 test images: %d %%' %
                (100 * correct / total))

        for i in range(len(classes)):
            print('Accuray of %5s : %2d %%' %
                (classes[i], 100 * class_correct[i]/class_total[i]))

    accuracy = 100 * correct / total
    return accuracy
