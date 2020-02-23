#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
from collections import OrderedDict
import copy
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset ')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--checkpointx', type=str, help='Save trained model checkpoint to file')

args, _ = parser.parse_known_args()


def load_train_checkpoint(arch = 'vgg19',num_labels = 102, hidden_units = 4096):
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    features = list(model.classifier.children())[:-1]

    num_filters = model.classifier[len(features)].in_features
    features.extend([
        nn.Dropout(),
        nn.Linear(num_filters, hidden_units),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, num_labels)
        #,nn.Softmax(dim=1)
    ])
    #Adding softmax
    model.classifier = nn.Sequential(*features)
    return model

def do_deep_learning(image_datasets, arch='vgg19', hidden_units=4096, epochs=3, learning_rate=0.001, gpu=True, checkpointx=''):
    if args.arch:
        arch = args.arch

    if args.hidden_units:
        hidden_units = args.hidden_units

    if args.epochs:
        epochs = args.epochs

    if args.learning_rate:
        learning_rate = args.learning_rate

    if args.gpu:
        gpu = args.gpu

    if args.checkpointx:
        checkpointx = args.checkpointx

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
        for x in list(image_datasets.keys())
    }

    # Calculate dataset sizes.
    dataset_sizes = {
        x: len(dataloaders[x].dataset)
        for x in list(image_datasets.keys())
    }


    print('Network architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)

    # Load the model
    num_labels = len(image_datasets['train'].classes)
    #model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)
    model = load_train_checkpoint(arch = arch, num_labels = num_labels, hidden_units = hidden_units)

    """if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        model.to('cuda')
    else:
        print('Using CPU for training')
        model.to('cpu')
    """
    """classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs = epochs
    print_every = 40
    steps = 0
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Using CPU for training')
        device = torch.device("cpu")


    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0
    #Validation  phase
    print("In validation phase")
    steps = 0
    for ve in range(epochs):
        model.eval()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders['valid']):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0
        if ve == 2:
            best_model_wts = copy.deepcopy(model.state_dict())

    model.to('cpu')
    model.load_state_dict(best_model_wts)
    model.class_to_idx = image_datasets['train'].class_to_idx
    print("Checking accuracy")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaders['valid']:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the valid images: %d %%' % (100 * correct / total))

    checkpoint = {'arch':'vgg19',
                  'input_size': 25088,
                  'output_size': 102,
                  'hidden_units':4096,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    if args.checkpointx:
        torch.save(checkpoint, args.checkpointx)
    else:
        torch.save(checkpoint, 'mycheckpoint.pth')
    print("Checkpoint created")

    return model

if args.data_dir:

    data_transforms = {
        'train':transforms.Compose([transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])])
    }
    image_datasets = {
            x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
            for x in list(data_transforms.keys())
    }
    do_deep_learning(image_datasets)

    
