#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:54:26 2019

@author: juan
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import torch.onnx
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.box_utils import nms
from vision.transforms import transforms as trans
from tqdm import tqdm
from PIL import Image
import cv2


def load_detection(model_path, net_type="vgg16-ssd", class_names=1):
    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(class_names, is_test=True)
        net.eval()
    elif net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(class_names, is_test=True)
    net.load(model_path)
    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    return predictor, net


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_sample(dataloaders, class_names):
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


def load_model(path, model, classes, percentage_frozen):
    if model == "densenet":
        model_conv = torchvision.models.densenet121(pretrained=False)
        num_ftrs = model_conv.classifier.in_features
        model_conv.classifier = nn.Linear(num_ftrs, classes)
    elif model == "mobilenet":
        model_conv = torchvision.models.mobilenet_v2(pretrained=False)
        num_ftrs = model_conv.classifier[1].in_features
        model_conv.classifier[1] = nn.Linear(num_ftrs, classes)
    elif model == "squeezenet":
        model_conv = torchvision.models.squeezenet1_0(pretrained=False)
        num_channels = model_conv.classifier[1].in_channels
        model_conv.classifier[1] = nn.Conv2d(num_channels, classes,
                                             kernel_size=(1, 1),
                                             stride=(1, 1))
    elif model == "resnet":
        model_conv = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, classes)

    total_param = round(percentage_frozen*len(list(model_conv.parameters())))

    for param in list(model_conv.parameters())[:total_param]:
        param.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                 model_conv.parameters()), lr=0.001,
                          momentum=0.9)

    checkpoint = torch.load(path)
    model_conv.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model_conv.eval()

    return model_conv


def save_model(epoch, model, optimizer, accuracy, loss, classes, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'classes': classes
            }, path)


def export_to_onnx(path, model, classes, percentage_frozen=0.9):

    model_conv = load_model(path, model, classes, percentage_frozen)

    batch_size = 1
    x = torch.randn(batch_size, 3, 300, 300, requires_grad=True)

    torch.onnx.export(model_conv,
                      x,
                      os.path.join("./saved_models/", str(model) + ".onnx"),
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True)


def load_data(data_dir):

    data_transforms = {
        'train_detec': transforms.Compose([
            trans.ConvertColor('BGR', 'RGB'),
            trans.Resize(300),
            trans.SubtractMeans(np.array([123, 117, 104])),
            trans.ToTensor(),


        ]),
        'val_detec': transforms.Compose([
            trans.ConvertColor('BGR', 'RGB'),
            trans.Resize(300),
            trans.SubtractMeans(np.array([123, 117, 104])),
            trans.ToTensor(),

        ]),
        'test_detec': transforms.Compose([
            trans.ConvertColor('BGR', 'RGB'),
            trans.Resize(300),
            trans.SubtractMeans(np.array([123, 117, 104])),
            trans.ToTensor(),
        ]),
        'train': transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {y: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[y])
                      for x, y in zip(
                              ['train', 'val', 'test',
                               'train', 'val', 'test'],
                              ['train', 'val', 'test',
                               'train_detec', 'val_detec', 'test_detec'])}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=4,
                                                  shuffle=False, num_workers=4)
                   for x in ['train', 'val', 'test',
                             'train_detec', 'val_detec', 'test_detec']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val',
                                                         'test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return dataloaders, dataset_sizes, class_names, device, image_datasets


def evaluate_model(model, dataloaders, device):

    since = time.time()
    model = model.to(device)
    model.eval()
    running_corrects = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / dataset_sizes['test']
    time_elapsed = time.time() - since
    mean_inference_time = time_elapsed / dataset_sizes['test']

    print('{} Inference time: {:.4f} Acc: {:.4f} NumParam: {}'.format(
            'test', mean_inference_time, epoch_acc,
            sum(p.numel() for p in model.parameters())))


def get_boxes(scores, boxes, height, width, top_k=1, prob_threshold=0.4):

    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, scores.size(1)):
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs = nms(box_probs,
                        nms_method=None,
                        score_threshold=prob_threshold,
                        iou_threshold=0.45,
                        sigma=0.5,
                        top_k=top_k,
                        candidate_size=150)
        picked_box_probs.append(box_probs)

        picked_labels.extend([class_index] * box_probs.size(0))
    if not picked_box_probs:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    picked_box_probs = torch.cat(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


def box_corrector(box):
    box = box.floor().to(torch.int32)
    if box < 0:
        box = 0
    elif box > 300:
        box = 300

    return box


def train_model(model, criterion, optimizer, scheduler, num_epochs,
                dataloaders, device, dataset_sizes, classes, path):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    predictor, net = load_detection(
            "./vgg16-ssd-Epoch-99-Loss-2.832231903076172.pth")
    net.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for ((inputs, labels), (inputs_detec, labels_detec)) in zip(dataloaders[phase],
                                                                      dataloaders[phase + '_detec']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs_detec = inputs_detec.to(device)

                with torch.no_grad():
                    confidence, locations = net.forward(inputs_detec)
                    for image_num in range(confidence.shape[0]):
                        print("input")
                        plt.imshow(inputs[image_num, :, :, :].cpu().numpy().transpose((1, 2, 0)))
                        plt.show()
                        print("input_detection")
                        plt.imshow(inputs_detec[image_num, :, :, :].cpu().numpy().transpose((1, 2, 0)))
                        plt.show()
                        boxes, _, _ = get_boxes(
                                confidence[image_num, :, :],
                                locations[image_num, :, :],
                                inputs_detec.shape[2],
                                inputs_detec.shape[3])
                        if len(boxes):
                            try:
                                boxes[0] = torch.Tensor([box_corrector(i) for i in boxes[0]])
                                boxes_numpy = boxes[0].cpu().numpy().astype(np.int16)
                                inputs[image_num, :, :, :] = trans.totensor(np.array(
                                        cv2.resize(trans.tocv2image(inputs[image_num,
                                                               :,
                                                               boxes_numpy[0]:
                                                                   boxes_numpy[2],
                                                               boxes_numpy[1]:
                                                                   boxes_numpy[3]]),
                                                   dsize=(300, 300)))).to(device)
                                print("After detection")
                                plt.imshow(inputs[image_num, :, :, :].cpu().numpy().transpose((1, 2, 0)))
                                plt.show()
                            except Exception as e:

                                continue
                            
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model(epoch, model, optimizer, epoch_acc, epoch_loss,
                           classes, path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train(epochs, model, classes, percentage_frozen, device,
          dataloaders, dataset_sizes, path):

    if model == "densenet":
        model_conv = torchvision.models.densenet121(pretrained=True)
        num_ftrs = model_conv.classifier.in_features
        model_conv.classifier = nn.Linear(num_ftrs, classes)
    elif model == "mobilenet":
        model_conv = torchvision.models.mobilenet_v2(pretrained=True)
        num_ftrs = model_conv.classifier[1].in_features
        model_conv.classifier[1] = nn.Linear(num_ftrs, classes)
    elif model == "squeezenet":
        model_conv = torchvision.models.squeezenet1_0(pretrained=True)
        num_channels = model_conv.classifier[1].in_channels
        model_conv.classifier[1] = nn.Conv2d(num_channels, classes,
                                             kernel_size=(1, 1),
                                             stride=(1, 1))
    elif model == "resnet":
        model_conv = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, classes)

    total_param = round(percentage_frozen*len(list(model_conv.parameters())))

    for param in list(model_conv.parameters())[:total_param]:
        param.requires_grad = False

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    # https://github.com/pytorch/pytorch/issues/679
    optimizer_conv = optim.SGD(filter(lambda p: p.requires_grad,
                                      model_conv.parameters()),
                               lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7,
                                           gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, epochs,
                             dataloaders, device, dataset_sizes, classes, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pct_freeze",
                        help="the percentage of layers frozen",
                        type=float, default=0.9, dest="pctf")
    parser.add_argument("--classf_net", help="classification network",
                        default="squeezenet", type=str, dest="clf")
    parser.add_argument("--detect_net", help="detection network",
                        default="vgg", type=str, dest="dtc")
    parser.add_argument("--num_epochs", help="number of epochs",
                        default=5, type=int, dest="epochs")
    parser.add_argument("--train", help="either train or test",
                        dest="train", type=bool, default=True)

    parsed = parser.parse_args()

    dataloaders, dataset_sizes, class_names, device, im_datasets = load_data(
            "./images_app/")

    if parsed.train:
        train(parsed.epochs, parsed.clf, len(class_names), parsed.pctf, device,
              dataloaders, dataset_sizes, "./saved_models/" +
              parsed.clf + "_detection.tar")
    else:
        model = load_model("./saved_models/" + parsed.clf + "_detection.tar",
                           parsed.clf,
                           len(class_names), parsed.pctf)
        evaluate_model(model, dataloaders, device)
