import os
import torch
import torchaudio
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

# from sklearn import metrics
# from sklearn import model_selection
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from utils import *
import argparse
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from audiodataset import *
from model import *

import warnings

warnings.filterwarnings("ignore")

#### ARGPARSE ####
parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--nb_features", type=int, default=32)
parser.add_argument("--run", type=int)
parser.add_argument("--dataset", type=str)

args = parser.parse_args()

RUN = args.run
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
NUMBER_FEATURES = args.nb_features
DATASET = args.dataset

EPOCHES = 1000

early_stopping = EarlyStopping(patience=50)
myRRC = my_RRC()
transforms = torch.nn.Sequential(myRRC, T.TimeMasking(40), T.FrequencyMasking(20))

  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############ DATA PREPARATION ##############

if DATASET == 'esc50':
    FILE_PATH = "/users/local/i21moumm/soundata/esc50/audio"
    CSV_PATH = "/users/local/i21moumm/soundata/esc50/meta/esc50.csv"
    PTH_FOLDER = "/users/local/i21moumm/ESCcode/autonorm_specgram/"
    # 4 folds for train, 1 fold for val/test (200 samples for each)
    train_path = PTH_FOLDER + "fold" + str(1) + ".pth"
    train = torch.load(train_path)
    train_data, train_label = train["data"], train["label"]
    for i in range(2, 5):
        PATH = PTH_FOLDER + "fold" + str(i) + ".pth"
        data_dict = torch.load(PATH)
        train_data_tmp, train_label_tmp = data_dict["data"], data_dict["label"]
        train_data, train_label = torch.cat((train_data, train_data_tmp), 0), torch.cat(
            (train_label, train_label_tmp), 0
        )
    # train_data = torch.load(PTH_FOLDER)
    valtest = torch.load(PTH_FOLDER + "fold5.pth")
    valtest_data, valtest_label = valtest["data"], valtest["label"]
    val_data, val_label = valtest_data[valtest_data.shape[0]//2:,...], valtest_label[valtest_data.shape[0]//2:,...]
    test_data, test_label = valtest_data[:valtest_data.shape[0]//2,...], valtest_label[:valtest_data.shape[0]//2,...]
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_data), shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_data), shuffle=False
    )        
elif DATASET == 'us8k':
    FILE_PATH = "/users/local/i21moumm/soundata/urbansound8k/audio"
    CSV_PATH = "/users/local/i21moumm/soundata/urbansound8k/metadata/UrbanSound8K.csv"  
    PTH_FOLDER = "/users/local/i21moumm/Urbancode/autonorm_specgram"
    # 8 folds for train, 1 fold for val and 1 fold for test
    train_path = PTH_FOLDER + "fold" + str(1) + ".pth"
    train = torch.load(train_path)
    train_data, train_label = train["data"], train["label"]
    for i in range(2, 9):
        PATH = PTH_FOLDER + "fold" + str(i) + ".pth"
        data_dict = torch.load(PATH)
        train_data_tmp, train_label_tmp = data_dict["data"], data_dict["label"]
        train_data, train_label = torch.cat((train_data, train_data_tmp), 0), torch.cat(
            (train_label, train_label_tmp), 0
        )
    # train_data = torch.load(PTH_FOLDER)
    val = torch.load(PTH_FOLDER + "fold9.pth")
    val_data, val_label = val["data"], val["label"]

    test = torch.load(PTH_FOLDER + "fold10.pth")
    test_data, test_label = test["data"], test["label"]

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

############ TRAINING ##############

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []
epoches_counter = 0


def train(model, train_dataloader, val_dataloader, optimizer, device, epochs):
    best_val_acc = 0
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_loss = 0.0
        current_train_acc = 0.0
        total = 0.0
        correct = 0.0
        mix_up = False
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for input, target in train_dataloader:
            input, target = input.to(device), target.to(device)
            # input = RRC(input)
            # input = (input - input.mean(0)) / input.std(0)
            input = transforms(input)

            optimizer.zero_grad()

            mix_up = False  # comment this to activate mix_up

            if mix_up:
                mixed_data, y_a, y_b, lam = mixup_data(input, target, device)
                prediction = cnn(input)
                loss = mix_criterion(loss_fn, prediction, y_a, y_b, lam)

            else:
                prediction = cnn(input)
                loss = loss_fn(prediction, target)

            # backpropagate error and update weights
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, labels_predicted = torch.max(prediction, dim=1)
            if mix_up:
                predicted_a = labels_predicted == y_a  # .sum().item()
                predicted_b = labels_predicted == y_b  # .sum().item()
                predicted = lam * predicted_a + (1 - lam) * predicted_b
                target_mixup = lam * y_a + (1 - lam) * y_b

                correct += (predicted == target_mixup).sum().item()
                mix_up = False

            else:
                correct += (labels_predicted == target).sum().item()
                mix_up = True

            total += target.size(0)

        global train_acc_history
        current_train_acc = 100 * (correct / total)
        train_acc_history.append(current_train_acc)

        val_loss = 0.0
        current_val_acc = 0.0
        
        total = 0.0
        correct = 0.0
        with torch.no_grad():
            model.eval()
            for input, target in val_dataloader:
                input, target = input.to(device), target.to(device)

                prediction = model(input)
                loss = loss_fn(prediction, target)

                val_loss += loss.item()

                _, labels_predicted = torch.max(prediction, dim=1)

                total += target.size(0)
                correct += (labels_predicted == target).sum().item()

            global val_acc_history
            current_val_acc = 100 * (correct / total)
            val_acc_history.append(current_val_acc)

            current_train_loss = train_loss / len(train_dataloader)
            current_val_loss = val_loss / len(val_dataloader)

            # Reduce LR on Plateau
            # scheduler.step(current_val_loss)

            global train_loss_history
            train_loss_history.append(current_train_loss)

            global val_loss_history
            val_loss_history.append(current_val_loss)

            if best_val_acc <= current_val_acc:
                best_epoch = i
                state = {"model": cnn.state_dict(), "val_acc": current_val_acc, "epoch": best_epoch}
                torch.save(state, PATH)
                best_val_acc = current_val_acc

        print(
            f"Training Loss: {current_train_loss} \t\t Validation Loss: {current_val_loss} \nTraining Acc: {current_train_acc} \t\t Validation Acc: {current_val_acc}"
        )
        print("---------------------------")
        global epoches_counter
        epoches_counter += 1
        """
        early_stopping(-1 * current_val_acc)
        if early_stopping.early_stop:
            break
        """
    print("Finished training")
    print(f"best validation acc is {best_val_acc} at epoch {best_epoch}")

if DATASET == 'esc50':
    nb_classes = 50
elif DATASET == 'us8k' :
    nb_classes = 10

cnn = ResNet12(number_features=args.nb_features, num_classes=nb_classes).to(DEVICE)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

PATH = (
    "deleteme/cnn"
    + "_lr"
    + str(LEARNING_RATE)
    + "_bs"
    + str(BATCH_SIZE)
    + "_nf"
    + str(NUMBER_FEATURES)
    + "_run"
    + str(RUN)
    + ".pth"
)
train(cnn, train_dataloader, val_dataloader, optimizer, DEVICE, EPOCHES)

epoches = range(epoches_counter)

plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(epoches, train_loss_history, label="train")
plt.plot(epoches, val_loss_history, label="val")
plt.xlabel("Epoches")
plt.ylabel("Loss")
plt.legend()
plt.savefig(
    "deleteme/trainval_loss"
    + "_lr"
    + str(LEARNING_RATE)
    + "_bs"
    + str(BATCH_SIZE)
    + "_nf"
    + str(NUMBER_FEATURES)
    + "_run"
    + str(RUN)
    + ".png"
)

plt.figure(figsize=(10, 5))
plt.title("Training and Validation Accuracy")
plt.plot(epoches, train_acc_history, label="train")
plt.plot(epoches, val_acc_history, label="val")
plt.xlabel("Epoches")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(
    "deleteme/trainval_acc"
    + "_lr"
    + str(LEARNING_RATE)
    + "_bs"
    + str(BATCH_SIZE)
    + "_nf"
    + str(NUMBER_FEATURES)
    + "_run"
    + str(RUN)
    + ".png"
)