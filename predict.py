import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import argparse

from torch.autograd import Variable
from PIL import Image
import numpy as np
from train import load_train_checkpoint
import train
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Image to predict')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--labels', type=str, help='JSON file containing label names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args, _ = parser.parse_known_args()


def predict(image, checkpoint, topk=5, labels='', gpu=False):

    if args.image:
        image = args.image

    if args.checkpoint:
        checkpoint = args.checkpoint

    if args.topk:
        topk = args.topk

    if args.labels:
        labels = args.labels

    if args.gpu:
        gpu = args.gpu


    checkpoint = torch.load(checkpoint)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    output_labels = len(checkpoint['class_to_idx'])
    model = load_train_checkpoint(arch = arch, output_labels = output_labels,hidden_units = hidden_units)
    
    model.load_state_dict(checkpoint['state_dict'])
   
    if gpu and torch.cuda.is_available():
        model.cuda()

    model.eval()

    img_loader = transforms.Compose([

        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()

    image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np.transpose(image, (1, 2, 0)) - mean)/std
    image = np.transpose(image, (2, 0, 1))

    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
    if gpu and torch.cuda.is_available():
        image = image.cuda()

    result = model(image).topk(topk)
    if gpu and torch.cuda.is_available():
        probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        classes = result[1].data.cpu().numpy()[0]
    else:
        probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        classes = result[1].data.numpy()[0]


    if labels:
        with open(labels, 'r') as lls:
            cat_to_name = json.load(lls)
    #labels = 'cat_to_name.json'
        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]

    model.train(mode=model.training)

    print('Predictions and probabilities:', list(zip(classes, probs)))

    return probs, classes

if args.image and args.checkpoint:
    predict(args.image, args.checkpoint)
