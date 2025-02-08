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
from torchvision import datasets, models, transforms
import shutil
import torch, gc

gc.collect()
torch.cuda.empty_cache()

# 檢查CUDA是否可用
use_gpu = torch.cuda.is_available()

# 設定資料集根目錄

root_dir = r'D:/Label_unnormal_8s_1208/'


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


root_dir_acc = r'://Label_unnormal_8s_1208/acc/Test/'


root_dir_gyro = r':/Label_unnormal_8s_1208/gyro/Test/'



datasets = [HeatmapDataset_new(root_dir_acc, root_dir_gyro, label, transform=transform) for label in range(5)]

batch_size = 16
test_loader = DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=batch_size, shuffle=True)


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


class MultiViewCNN(nn.Module):
    def __init__(self, cnn_model):
        super(MultiViewCNN, self).__init__()

        self.cnn_model_a = deepcopy(cnn_model)
        self.cnn_model_b = deepcopy(cnn_model)

        self.fc1 = torch.nn.Linear(in_features=10, out_features=5)

    def forward(self, x_a, x_b):
        x_a = self.cnn_model_a(x_a)
        # print(f"CNN Model A output shape: {x_a.shape}")
        x_b = self.cnn_model_b(x_b)
        # print(f"CNN Model B output shape: {x_b.shape}")
        x_a = x_a.view(-1, 5)
        x_b = x_b.view(-1, 5)
        x = torch.cat((x_a, x_b), dim=1)
        # print(f"Concatenated output shape: {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"FC1 output shape: {x.shape}")
        return x



path_all = r'D:/Label_8s_new/training_model/model/multi_model_all.pt'

path_02 = r'D:/Label_8s_new/training_model/model/multi_model_0_2.pt'

multi_view_model_all = torch.load(path_all)

multi_view_model_all.eval()
multi_view_model_02 = torch.load(path_02)

multi_view_model_02.eval()

model1 = multi_view_model_all
model2 = multi_view_model_02


# 計算測試集準確度
def test_multi_view_model(model1, model2, test_loader):
    model1.eval()
    model2.eval()

    correct_combined = 0
    total_combined = 0
    total_test = 0
    correct_test = 0

    all_predicted_combined = []
    all_labels_combined = []

    with torch.no_grad():

        for batch in test_loader:
            inputs_a_acc, inputs_b_gyro, labels = batch  # 解包每個批次的資料

            if use_gpu:
                inputs_a_acc = inputs_a_acc.to('cuda')
                inputs_b_gyro = inputs_b_gyro.to('cuda')
                labels = labels.to('cuda')

            outputs = model1(inputs_a_acc, inputs_b_gyro)

            _, predicted_test = torch.max(outputs, 1)

            outputs_02 = model2(inputs_a_acc, inputs_b_gyro)

            _, predicted_test_02 = torch.max(outputs_02, 1)

            # 找到 predicted_test 中值為 0 或 2 的索引位置
            indices_0 = torch.nonzero(predicted_test == 0).squeeze()
            indices_2 = torch.nonzero(predicted_test == 2).squeeze()

            # 用 predicted_test_02 中相應位置的元素替換 predicted_test 中的 0 和 2
            predicted_test[indices_0] = predicted_test_02[indices_0]
            predicted_test[indices_2] = predicted_test_02[indices_2]

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
        precision_i = precision_score(all_labels_combined, all_predicted_combined, labels=[i], average='macro')
        recall_i = recall_score(all_labels_combined, all_predicted_combined, labels=[i], average='macro')
        f1_i = f1_score(all_labels_combined, all_predicted_combined, labels=[i], average='macro')

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


# 測試 Multi-View CNN 模型
accuracy = test_multi_view_model(multi_view_model_all, multi_view_model_02, test_loader)
print(f"Test Multi-View Accuracy: {accuracy:.4f}")









