import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import dill
from sklearn import metrics
import numpy as np

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device:{torch.cuda.get_device_name(0)}")


def model_metrics(labels, pred):
    '''Calculate evaluation metrics.'''
    tn, fp, fn, tp = metrics.confusion_matrix(labels, pred).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    return acc, sn, sp, mcc


def load_dataloader(type, species, cut=None):
    if cut == None:
        with open('./dataloder/' + species + '_' + type + '.pkl', 'rb') as f_dataloader:
            dataloader = dill.load(f_dataloader)
    else:
        with open('./dataloder/' + species + '_' + type + '_' + str(cut) + '.pkl', 'rb') as f_dataloader:
            dataloader = dill.load(f_dataloader)
    return dataloader


def acc_count(list1, list2):
    num_of_right = 0
    num_all = len(list1)
    if len(list1) == len(list2):
        for i in range(len(list1)):
            if list1[i] == list2[i]:
                num_of_right += 1
    else:
        print('Wrong wrong wrong!!!')
    return num_of_right, num_all


############################################################################
#config_lr = 0.0003549
class lncRNAdeep87(nn.Module):
    """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for one_epoch in range(100):
    """

    def __init__(self):
        super(lncRNAdeep, self).__init__()
        self.sequence_lenth = 3000
        self.input_channel = 19
        self.out_channel1 = 64
        self.out_channel2 = 32
        self.filter_size = 10
        self.filter_size2 = 10
        self.stride = 1
        self.fc1_size = 64
        self.fc2_size = 32
        self.pool_1 = False
        self.pool_2 = False
        self.fc2 = False
        self.feature1= 10
        self.feature2 = 2
        self.ENCODING_TYPE = ENCODING_TYPE
        self.hidden_size = 64
        flatten_size = (self.sequence_lenth - self.filter_size) // self.stride + 1

        flatten_size = (flatten_size - self.filter_size2) // self.stride + 1

        self.flatten_size = flatten_size

        self.conv1 = nn.Conv1d(in_channels=self.input_channel, out_channels=self.out_channel1,
                               kernel_size=self.filter_size, stride=self.stride)
        # self.max_pool1 = nn.MaxPool1d(kernel_size=self.filter_size2, stride=self.stride)
        self.conv2 = nn.Conv1d(self.out_channel1, self.out_channel2, self.filter_size2, self.stride)
        # self.max_pool2 = nn.MaxPool1d(self.filter_size2, self.stride)
        self.liner1 = nn.Linear(self.out_channel2 * self.flatten_size, self.fc1_size)
        self.liner2 = nn.Linear(self.fc1_size, self.fc2_size)
        if self.fc2:
            self.liner3 = nn.Linear(self.fc2_size, self.feature1)
        else:
            self.liner3 = nn.Linear(self.fc1_size, self.feature1)
        self.lstm = nn.LSTM(self.filter_size, self.hidden_size, 2)
        self.linear_1 = nn.Linear(6, 6)
        self.linear_2 = nn.Linear(6, self.feature2)

        self.linear_a1 = nn.Linear(self.feature1 + self.feature2, 8)
        self.linear_a2 = nn.Linear(8, 1)
        self.rnn = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            # input_size=self.out_channel1,  # 图片每行的数据28像素点
            input_size=self.out_channel1,
            hidden_size=self.hidden_size,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.rnn_linear=nn.Linear(self.hidden_size, self.feature1)

    def forward(self, x):
        if ENCODING_TYPE == 1:
            x1 = x[:, 0:4, 0:3000]
        elif ENCODING_TYPE == 2:
            x1 = x[:, 4:7, 0:3000]
        elif ENCODING_TYPE == 3:
            x1 = x[:, 7:13, 0:3000]
        elif ENCODING_TYPE == 4:
            x1 = x[:, 13:19, 0:3000]
        elif ENCODING_TYPE == 5:
            x1 = x[:, 0:19, 0:3000]
        elif ENCODING_TYPE == 7:
            x1 = x[:, 0:19, 0:3000]
        x2 = x[:, 0:6, 3000]

        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        # print(x1.size())
        x1 = x1.view(x1.size(0), -1)
        x1 = F.relu(self.liner1(x1))
        x1 = F.relu(self.liner3(x1))
        x2 = F.relu(self.linear_1(x2))
        x2 = F.relu(self.linear_2(x2))
        inputs = [x1, x2]
        x3 = torch.cat(inputs, dim=1)
        x3 = F.relu(self.linear_a1(x3))
        x3 = self.linear_a2(x3)
        return torch.sigmoid(x3)

############################################################################
#config_lr = 0.0003549
class lncRNAdeep(nn.Module):
    """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for one_epoch in range(100):
    """

    def __init__(self):
        super(lncRNAdeep, self).__init__()
        self.sequence_lenth = 3000
        self.ENCODING_TYPE = ENCODING_TYPE
        if self.ENCODING_TYPE == 1:
            self.input_channel = 4
        elif self.ENCODING_TYPE == 2:
            self.input_channel = 3
        elif self.ENCODING_TYPE == 3:
            self.input_channel = 6
        elif self.ENCODING_TYPE == 4:
            self.input_channel = 6
        elif self.ENCODING_TYPE == 5:
            self.input_channel = 19
        elif self.ENCODING_TYPE == 6:
            self.input_channel = 19
        elif self.ENCODING_TYPE == 7:
            self.input_channel = 19
        self.out_channel1 = 64
        self.out_channel2 = 32
        self.filter_size = 10
        self.filter_size2 = 10
        self.stride = 1
        self.fc1_size = 64
        self.fc2_size = 32
        self.pool_1 = False
        self.pool_2 = False
        self.fc2 = False
        self.feature1= 10
        self.feature2 = 2

        self.hidden_size = 64
        flatten_size = (self.sequence_lenth - self.filter_size) // self.stride + 1

        flatten_size = (flatten_size - self.filter_size2) // self.stride + 1

        self.flatten_size = flatten_size

        self.conv1 = nn.Conv1d(in_channels=self.input_channel, out_channels=self.out_channel1,
                               kernel_size=self.filter_size, stride=self.stride)
        self.conv2 = nn.Conv1d(self.out_channel1, self.out_channel2, self.filter_size2, self.stride)
        self.liner1 = nn.Linear(self.out_channel2 * self.flatten_size, self.fc1_size)
        self.liner2 = nn.Linear(self.fc1_size, self.fc2_size)
        if self.fc2:
            self.liner3 = nn.Linear(self.fc2_size, self.feature1)
        else:
            self.liner3 = nn.Linear(self.fc1_size, self.feature1)
        self.lstm = nn.LSTM(self.filter_size, self.hidden_size, 2)
        self.linear_1 = nn.Linear(6, 6)
        self.linear_2 = nn.Linear(6, self.feature2)

        self.linear_a1 = nn.Linear(self.feature1 + self.feature2, 8)
        self.linear_only_x1 = nn.Linear(self.feature1, 8)
        self.linear_only_x2 = nn.Linear(self.feature2, 8)
        self.linear_a2 = nn.Linear(8, 1)
        self.rnn = nn.LSTM(
            input_size=self.out_channel1,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.rnn_linear=nn.Linear(self.hidden_size, self.feature1)

    def forward(self, x):
        if  self.ENCODING_TYPE in [1,2,3,4,5]:
            if self.ENCODING_TYPE == 1:
                x1 = x[:, 0:4, 0:3000]
            elif self.ENCODING_TYPE == 2:
                x1 = x[:, 4:7, 0:3000]
            elif self.ENCODING_TYPE == 3:
                x1 = x[:, 7:13, 0:3000]
            elif self.ENCODING_TYPE == 4:
                x1 = x[:, 13:19, 0:3000]
            elif self.ENCODING_TYPE == 5:
                x1 = x[:, 0:19, 0:3000]
            x2 = x[:, 0:6, 3000]

            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            # print(x1.size())
            x1 = x1.view(x1.size(0), -1)
            x1 = F.relu(self.liner1(x1))
            x1 = F.relu(self.liner3(x1))
            # x2 = F.relu(self.linear_1(x2))
            # x2 = F.relu(self.linear_2(x2))
            # inputs = [x1, x2]
            # x3 = torch.cat(inputs, dim=1)
            x3 = F.relu(self.linear_only_x1(x1))
            x3 = self.linear_a2(x3)
        elif self.ENCODING_TYPE == 6:
            x1 = x[:, 0:19, 0:3000]
            x2 = x[:, 0:6, 3000]

            # x1 = F.relu(self.conv1(x1))
            # x1 = F.relu(self.conv2(x1))
            # # print(x1.size())
            # x1 = x1.view(x1.size(0), -1)
            # x1 = F.relu(self.liner1(x1))
            # x1 = F.relu(self.liner3(x1))
            x2 = F.relu(self.linear_1(x2))
            x2 = F.relu(self.linear_2(x2))
            # inputs = [x1, x2]
            # x3 = torch.cat(inputs, dim=1)
            x3 = F.relu(self.linear_only_x2(x2))
            x3 = self.linear_a2(x3)
        elif self.ENCODING_TYPE == 7:
            x1 = x[:, 0:19, 0:3000]
            x2 = x[:, 0:6, 3000]

            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            # print(x1.size())
            x1 = x1.view(x1.size(0), -1)
            x1 = F.relu(self.liner1(x1))
            x1 = F.relu(self.liner3(x1))
            x2 = F.relu(self.linear_1(x2))
            x2 = F.relu(self.linear_2(x2))
            inputs = [x1, x2]
            x3 = torch.cat(inputs, dim=1)
            x3 = F.relu(self.linear_a1(x3))
            x3 = self.linear_a2(x3)

        return torch.sigmoid(x3)



best_validation_acc = 0
config_lr = 0.0004
epoch = 20
species = 'Human'
cut = 10
# 1:one-hot 2:NCP 3:DPCP 4:TPCP :5ORF+GC 6:1~4 7:all
# ENCODING_TYPE = 6


model= torch.load('./best_model.pkl').to(device)
test_dataloader_h = load_dataloader('test', 'Human')

print('Evaluate in test dataset...')
first=True
for i, (data, target) in enumerate(test_dataloader_h):
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    pred = output.detach().cpu().numpy().reshape(output.shape[0])
    labels = target.cpu().numpy().reshape(output.shape[0])
    if first:
        all_labels=labels
        all_pred = pred
        first=False
    else:
        all_labels =np.hstack((all_labels,labels))
        all_pred = np.hstack((all_pred, pred))
t_acc, t_sn, t_sp, t_mcc = model_metrics(all_labels, all_pred.round())
t_auc = metrics.roc_auc_score(all_labels, all_pred)
print('t_acc, t_sn, t_sp, t_mcc,t_auc', t_acc, t_sn, t_sp, t_mcc, t_auc)


