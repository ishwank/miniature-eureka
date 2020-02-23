
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


def load_train_checkpoint(arch = 'vgg19',output_labels = 102, hidden_units = 4096):
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        features = list(model.classifier.children())[:-1]
        input_feature = model.classifier[len(features)].in_features
        features.extend([
            nn.Dropout(),
            nn.Linear(input_feature, hidden_units),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_units, output_labels),
            nn.LogSoftmax(dim=1)
        ]) 
        
        model.classifier = nn.Sequential(*features)
    elif arch == 'densenet':
        hidden_units = 1000
        model = models.densenet201(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                       ('fc1', nn.Linear(1920, hidden_units)),
                       ('relu', nn.ReLU()),
                       ('fc2', nn.Linear(hidden_units, output_labels)),
                       ('output', nn.LogSoftmax(dim=1))
                       ]))
        model.classifier = classifier
    else:
        raise ValueError('Architecture not supported. Training in vgg and densenet. ',arch)
    return model

def do_deep_learning(image_datasets, arch='vgg19', hidden_units=4096, epochs=10, learning_rate=0.001, gpu=False, checkpointx=''):
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
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)

    # Load the model
    output_labels = len(image_datasets['train'].classes)
    model = load_train_checkpoint(arch = arch, output_labels = output_labels, hidden_units = hidden_units)
    criterion = nn.CrossEntropyLoss()
    print(criterion)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,momentum = 0.9)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
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
        epochcorrect = 0
        running_loss = 0
        for phase in ['train','valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for ii, (inputs, labels) in enumerate(dataloaders['train']):
                steps += 1

                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()
                # Forward and backward passes
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                epochcorrect += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = epochcorrect.double() / dataset_sizes[phase]
            print("Epoch: {}/{} {} ...Loss: {:.4f} Accuracy: {:.4f}".format(e+1, epochs,phase,epoch_loss,epoch_accuracy))
    
            running_loss = 0
            if phase == 'valid':
                model.to('cpu')
                #print("Validation accuracy")
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in dataloaders['valid']:
                        images, labels = data
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                print('Validation phase accuracy %d %%' % (100 * correct / total))
                model.to('cuda')
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model_wts = copy.deepcopy(model.state_dict())
            
    print('Best accuracy: {:4f}'.format(best_accuracy))
    
    model.load_state_dict(best_model_wts)
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    
    checkpoint = {'arch':arch,
                  'hidden_units':hidden_units,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    if args.checkpointx:
        torch.save(checkpoint, args.checkpointx)
    else:
        torch.save(checkpoint, 'mycheckpoint.pth')
    print("Checkpoint created")
    #print(checkpoint)

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
