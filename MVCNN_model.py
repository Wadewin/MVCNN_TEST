# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:45:07 2023

@author: wade
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
# from torchsummary import summary
from sklearn.model_selection import train_test_split
from PIL import Image
# from PIL import Image, ImageOps, ImageEnhance
import os

# from imblearn.over_sampling import SMOTE
torch.cuda.empty_cache()
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
from copy import deepcopy
import torchvision
# from torchview import draw_graph
import torch, gc

gc.collect()
torch.cuda.empty_cache()
# 檢查CUDA是否可用
use_gpu = torch.cuda.is_available()


class HeatmapDataset_new(Dataset):
    def __init__(self, root_dir_acc, root_dir_gyro, label, transform=None):
        self.root_dir_acc = root_dir_acc
        self.root_dir_gyro = root_dir_gyro

        self.label = label
        self.image_paths_acc = [os.path.join(root_dir_acc, f"label {label}", filename) for filename in
                                os.listdir(os.path.join(root_dir_acc, f"label {label}"))]
        self.image_paths_gyro = [os.path.join(root_dir_gyro, f"label {label}", filename) for filename in
                                 os.listdir(os.path.join(root_dir_gyro, f"label {label}"))]

        self.transform = transform

    def __len__(self):
        return len(self.image_paths_acc)
        return len(self.image_paths_gyro)

    def __getitem__(self, idx):
        image_path_acc = self.image_paths_acc[idx]
        image_path_gyro = self.image_paths_gyro[idx]

        image_acc = Image.open(image_path_acc)
        image_gyro = Image.open(image_path_gyro)

        if self.transform:
            image_acc = self.transform(image_acc)
            image_gyro = self.transform(image_gyro)

        label = torch.tensor(self.label)  # 將整數標籤轉換為 PyTorch 張量
        return image_acc, image_gyro, label


# 資料預處理與增強
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 設定資料集根目錄

root_dir = r':/Label_unnormal_8s_1208/'

root_dir_acc = r':/Label_unnormal_8s_1208/acc/TrainValid/'

root_dir_gyro = r':/Label_unnormal_8s_1208/gyro/TrainValid/'

# new HEAT

datasets = [HeatmapDataset_new(root_dir_acc, root_dir_gyro, label, transform=transform) for label in range(5)]

# 資料集分割比例
train_split = 0.8
train_datasets = []
test_datasets = []

for dataset in datasets:
    if len(dataset) < 2:
        print(f"Skipping class {dataset.label} due to insufficient samples.")
        continue

    labels = [dataset.label] * len(dataset)
    train_data, test_data = train_test_split(dataset, test_size=1 - train_split, random_state=42, stratify=labels)

    train_datasets.append(train_data)
    test_datasets.append(test_data)

# 合併資料集
batch_size = 16

train_dataset = torch.utils.data.ConcatDataset(train_datasets)
test_dataset = torch.utils.data.ConcatDataset(test_datasets)

# 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 設定驗證數據的 DataLoader
valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


test_dir_acc = r':/Label_unnormal_8s_1208/acc/Test/'
test_dir_gyro = r':/Label_unnormal_8s_1208/gyro/Test/'

datasets = [HeatmapDataset_new(test_dir_acc, test_dir_gyro, label, transform=transform) for label in range(5)]

test_loader = DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=batch_size, shuffle=True)


# 建立 CNN 模型


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolution 1 , input_shape=(3,204,320)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,
                               padding=1)  # Output_shape=(16,204,320)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output_shape=(16,102,160)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                               padding=1)  # Output_shape=(32,102,160)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output_shape=(32,51,80)

        self.fc1 = nn.Linear(64 * 80 * 51, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 5)  # 5 labels

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        # print(f"Conv1 output shape: {x.shape}")
        x = self.pool2(self.relu2(self.conv2(x)))
        # print(f"Conv2 output shape: {x.shape}")

        x = x.view(-1, 64 * 80 * 51)  # Reshape for fully connected layer
        # print(f"Flatten output shape: {x.shape}")
        x = self.relu3(self.fc1(x))
        # print(f"FC1 output shape: {x.shape}")
        x = self.dropout(x)
        x = self.fc2(x)
        # print(f"FC2 output shape: {x.shape}")
        return x


model = CNNModel()


class MultiViewCNN(nn.Module):
    def __init__(self, cnn_model):
        super(MultiViewCNN, self).__init__()

        self.cnn_model_a = deepcopy(cnn_model)
        self.cnn_model_b = deepcopy(cnn_model)

        self.fc1 = torch.nn.Linear(in_features=10, out_features=5)
        self.dropout = nn.Dropout(0.05)  # 添加一个 20% 的 Dropout

    def forward(self, x_a, x_b):
        x_a = self.cnn_model_a(x_a)
        x_b = self.cnn_model_b(x_b)

        x_a = x_a.view(-1, 5)
        x_b = x_b.view(-1, 5)
        x = torch.cat((x_a, x_b), dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 将 Dropout 应用于全连接层输出
        return x



# 設置 L2 正则化参数
l2_lambda = 0.001  # 根据需要调整

# 初始化多视角 CNN 模型
multi_view_model = MultiViewCNN(model)
multi_view_model.to('cuda')
criterion = nn.CrossEntropyLoss()

# 定义优化器并加入 L2 正则化
# optimizer = optim.Adam(multi_view_model.parameters(), lr=0.001, weight_decay=l2_lambda)
optimizer = optim.SGD(multi_view_model.parameters(), lr=0.005, weight_decay=l2_lambda, momentum=0.9)
# optimizer = optim.SGD(multi_view_model.parameters(), lr=0.002, weight_decay=l2_lambda, momentum=0.9)

# multi_view_model = MultiViewCNN(model)
# multi_view_model.to('cuda')
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(multi_view_model.parameters(), lr=0.001)


# 如果有GPU可用，將模型和數據移動到GPU
if use_gpu:
    multi_view_model.to('cuda')  # 將模型移動到GPU
    print("GPU is available and model is moved to GPU.")
else:
    print("GPU is not available. Using CPU.")

best_loss = float('inf')


def train_multi_view_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, patience=2):
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []  # 用於儲存每個 epoch 的 training loss
    valid_losses = []  # 用於儲存每個 epoch 的 validation loss
    train_accuracy = []  # 用於儲存每個 epoch 的 training accuracy
    valid_accuracy = []  # 用於儲存每個 epoch 的 valid accuracy

    for epoch in range(num_epochs):
        # 計算訓練資料集的 Loss&Accuracy
        model.train()
        running_loss = 0.0
        loss = 0.0
        correct_train = 0

        total_train = 0
        for batch in train_loader:
            inputs_a_acc, inputs_b_gyro, labels = batch  # 解包每個批次的資料

            if use_gpu:
                inputs_a_acc = inputs_a_acc.to('cuda')
                inputs_b_gyro = inputs_b_gyro.to('cuda')
                labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(inputs_a_acc, inputs_b_gyro)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels.long()).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # train_accuracy_epoch = correct_train / total_train
        train_accuracy_epoch = (correct_train) / (total_train)
        train_accuracy.append(train_accuracy_epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train_Loss: {train_loss}, Train_Accuracy: {train_accuracy_epoch}')

        # 計算驗證資料集的 Loss&Acuuracy
        model.eval()
        valid_loss = 0.0

        correct_valid = 0
        total_valid = 0
        with torch.no_grad():

            for batch in valid_loader:
                inputs_a_acc, inputs_b_gyro, labels = batch  # 解包每個批次的資料

                if use_gpu:
                    inputs_a_acc = inputs_a_acc.to('cuda')
                    inputs_b_gyro = inputs_b_gyro.to('cuda')
                    labels = labels.to('cuda')

                outputs = model(inputs_a_acc, inputs_b_gyro)
                loss = criterion(outputs, labels.long())

                valid_loss += loss.item() * inputs_a_acc.size(0)

                _, predicted_valid = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted_valid == labels.long()).sum().item()  # 計算驗證準確度

        valid_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        valid_accuracy_epoch = correct_valid / (total_valid)
        valid_accuracy.append(valid_accuracy_epoch)

        # print(f'Epoch [{epoch+1}/{num_epochs}], Valid_Loss: {valid_loss}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Valid_Loss: {valid_loss}, Valid_Accuracy: {valid_accuracy_epoch}')

        # 早停機制
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: Validation loss did not improve for {patience} consecutive epochs.")
                break
    # 繪製 Training Loss 和 Validation Loss 圖
    plt.figure(figsize=(18, 6))

    # 子圖1：Training Loss 和 Validation Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='red')
    plt.plot(valid_losses, label='Validation Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # 子圖2：Training Accuracy 和 Validation Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracy, label='Training Accuracy', color='green')
    plt.plot(valid_accuracy, label='Validation Accuracy', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # 子圖3：Training Accuracy 和 Training Loss
    plt.subplot(1, 3, 3)
    plt.plot(train_accuracy, label='Training Accuracy', color='green')
    plt.plot(train_losses, label='Training Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy and Loss')
    plt.legend()
    plt.title('Training Accuracy and Training Loss')

    # 保存圖
    # curve_path = '/root/workspace/myenv/multiview cnn dataset/'
    curve_path = root_dir
    accuracy_loss_curve_file = os.path.join(curve_path, 'AccuracyLoss_curve.png')

    plt.savefig(accuracy_loss_curve_file)
    print(f"AccuracyLoss curve saved at {accuracy_loss_curve_file}")


# 訓練 Multi-View CNN 模型''迭代
num_epochs = 30
patience = 6  # 早停的容忍次數
train_multi_view_model(multi_view_model, train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs,
                       patience=patience)
# train_multi_view_model(multi_view_model, train_loader,test_loader, criterion, optimizer, num_epochs=num_epochs, patience=patience)

print("Multi-View Training complete")

path = r':/Label_unnormal_8s_1208/TRAIN_TEST_lr0.005_b32_model.pt'

torch.save(multi_view_model, path)

path_1 = r':/Label_normalized_v3_CropDcre/training_model/model_checkpoint.tar'

torch.save({
    'epoch': num_epochs,
    'model_state_dict': multi_view_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': best_loss,
}, path_1)

# 測試
path = r':/Label_unnormal_8s_1208/lr0.001_b16_model.pt'

multi_view_model = torch.load(path)

multi_view_model.eval()


def test_multi_view_model(model, test_loader):
    model.eval()
    correct_combined = 0
    total_combined = 0
    total_test = 0
    correct_test = 0
    # total_combined_a_acc = 0
    # total_combined_b_gyro = 0
    # correct_combined_a_acc = 0
    # correct_combined_b_gyro = 0
    all_predicted_combined = []
    all_labels_combined = []

    with torch.no_grad():

        for batch in test_loader:
            inputs_a_acc, inputs_b_gyro, labels = batch  # 解包每個批次的資料

            if use_gpu:
                inputs_a_acc = inputs_a_acc.to('cuda')
                inputs_b_gyro = inputs_b_gyro.to('cuda')
                labels = labels.to('cuda')

            outputs = model(inputs_a_acc, inputs_b_gyro)

            _, predicted_test = torch.max(outputs, 1)

            total_test += labels.size(0)
            correct_test += (predicted_test == labels.long()).sum().item()

            all_predicted_combined.extend(predicted_test.cpu().numpy())
            all_labels_combined.extend(labels.cpu().numpy())

    # 計算合併的混淆矩陣
    cm_combined = confusion_matrix(all_labels_combined, all_predicted_combined)

    # 混淆矩陣
    label_names_combined = [f"Label {i}" for i in range(5)]
    df_cm_combined_Num = pd.DataFrame(cm_combined, index=label_names_combined, columns=label_names_combined)
    df_cm_combined_Probability = pd.DataFrame(cm_combined / cm_combined.sum(axis=1)[:, None],
                                              index=label_names_combined, columns=label_names_combined)

    # 印混淆矩阵
    print("Confusion Matrix:")
    print(df_cm_combined_Num)

    # 打印每个标签的 precision、recall 和 f1 分数
    for i in range(5):
        precision_i = precision_score(all_labels_combined, all_predicted_combined, labels=[i], average='micro')
        recall_i = recall_score(all_labels_combined, all_predicted_combined, labels=[i], average='micro')
        f1_i = f1_score(all_labels_combined, all_predicted_combined, labels=[i], average='micro')

        print(f'\nLabel {i}:')
        print(f'  Precision: {precision_i:.4f}')
        print(f'  Recall: {recall_i:.4f}')
        print(f'  F1 Score: {f1_i:.4f}')

    # Number of Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm_combined_Num, annot=True, fmt=".2f", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix (Number)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    number_confusion_matrix_path = root_dir
    number_confusion_matrix_file = os.path.join(number_confusion_matrix_path, 'Number_confusion_matrix.png')
    plt.savefig(number_confusion_matrix_file)
    print(f"Number Confusion matrix saved at {number_confusion_matrix_file}")

    # Probability of Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm_combined_Probability, annot=True, fmt=".2f", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix (Probabilities)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    probability_confusion_matrix_path = root_dir
    probability_confusion_matrix_file = os.path.join(probability_confusion_matrix_path,
                                                     'Probability_confusion_matrix.png')
    plt.savefig(probability_confusion_matrix_file)
    print(f"Probability Confusion matrix saved at {probability_confusion_matrix_file}")

    # 計算精確度、召回率和F1分數
    precision = precision_score(all_labels_combined, all_predicted_combined, average='weighted')
    recall = recall_score(all_labels_combined, all_predicted_combined, average='weighted')
    f1 = f1_score(all_labels_combined, all_predicted_combined, average='weighted')
    accuracy = correct_test / total_test
    # accuracy =( correct_combined_a_acc+correct_combined_b_gyro)/(total_combined_a_acc+total_combined_b_gyro)
    print(f'\nAccuracy: {accuracy:.4f}')
    print(f'Weighted Precision: {precision:.4f}')
    print(f'Weighted Recall: {recall:.4f}')
    print(f'Weighted F1 Score: {f1:.4f}')

    return accuracy


accuracy = test_multi_view_model(multi_view_model, valid_loader)
print(f"Test Multi-View Accuracy: {accuracy:.4f}")

# 真實資料
accuracy = test_multi_view_model(multi_view_model, test_loader)
print(f"Test Multi-View Accuracy: {accuracy:.4f}")