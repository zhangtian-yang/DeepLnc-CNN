import warnings
import torch
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
import datetime

DPCP = np.load('./bin/DPCP.npy', allow_pickle=True).item()
TPCP = np.load('./bin/TPCP.npy', allow_pickle=True).item()
warnings.filterwarnings("ignore")


def ORF_encode(seq):
    l = len(seq)
    stat = 3000
    end = 0
    for i in range(l - (l % 3) - 4):
        if (seq[i] + seq[i + 1] + seq[i + 2]) == 'ATG':
            if i < stat:
                stat = i
        if (seq[i] + seq[i + 1] + seq[i + 2]) in ['TAA', 'TAG', 'TGA']:
            if i > end:
                end = i
    orf_longest_lenth = end - stat
    stat = 3000
    end = 0
    for i in range(1, l - (l % 3) - 3):
        if (seq[i] + seq[i + 1] + seq[i + 2]) == 'ATG':
            if i < stat:
                stat = i
        if (seq[i] + seq[i + 1] + seq[i + 2]) in ['TAA', 'TAG', 'TGA']:
            if i > end:
                end = i
    if (end - stat) > orf_longest_lenth:
        orf_longest_lenth = end - stat
    stat = 3000
    end = 0
    for i in range(2, l - (l % 3) - 2):
        if (seq[i] + seq[i + 1] + seq[i + 2]) == 'ATG':
            if i < stat:
                stat = i
        if (seq[i] + seq[i + 1] + seq[i + 2]) in ['TAA', 'TAG', 'TGA']:
            if i > end:
                end = i
    stat = 3000
    end = 0
    for i in range(l - (l % 3) - 4):
        if (seq[i] + seq[i + 1] + seq[i + 2]) in ['AAT', 'GAT', 'AGT']:
            if i < stat:
                stat = i
        if (seq[i] + seq[i + 1] + seq[i + 2]) == 'GTA':
            if i > end:
                end = i
    orf_longest_lenth = end - stat
    stat = 3000
    end = 0
    for i in range(1, l - (l % 3) - 3):
        if (seq[i] + seq[i + 1] + seq[i + 2]) in ['AAT', 'GAT', 'AGT']:
            if i < stat:
                stat = i
        if (seq[i] + seq[i + 1] + seq[i + 2]) == 'GTA':
            if i > end:
                end = i
    if (end - stat) > orf_longest_lenth:
        orf_longest_lenth = end - stat
    stat = 3000
    end = 0
    for i in range(2, l - (l % 3) - 2):
        if (seq[i] + seq[i + 1] + seq[i + 2]) in ['AAT', 'GAT', 'AGT']:
            if i < stat:
                stat = i
        if (seq[i] + seq[i + 1] + seq[i + 2]) == 'GTA':
            if i > end:
                end = i
    if (end - stat) > orf_longest_lenth:
        orf_longest_lenth = end - stat
    if orf_longest_lenth > 0:
        orf_longest_coverage = orf_longest_lenth / l
    else:
        orf_longest_lenth = 0
        orf_longest_coverage = 0
    return orf_longest_coverage, orf_longest_lenth


def GC_content(seq):
    l = len(seq)
    A_content = 0
    T_content = 0
    G_content = 0
    C_content = 0
    for base in seq:
        if base == 'A':
            A_content += 1
        elif base == 'T':
            T_content += 1
        elif base == 'G':
            G_content += 1
        elif base == 'C':
            C_content += 1
        else:
            print('Wrong:the str out of range[A,T,G,C].')
            sys.exit()
    A_content = A_content / l
    T_content = T_content / l
    G_content = G_content / l
    C_content = C_content / l
    GandC = G_content + C_content
    AandT = A_content + T_content
    AthanT = A_content - T_content
    GthanC = G_content - C_content
    return GandC, AandT, AthanT, GthanC


def discrete_encoding(seq):
    matrix = np.zeros((19, 1), dtype=np.float32)
    # def  ORF_encode(seq):
    # return orf_coverage, orf_lenth
    matrix[0, 0], matrix[1, 0] = ORF_encode(seq)

    # def GC_content(seq):
    # return GandC, AandT, AthanT, GthanC
    matrix[2, 0], matrix[3, 0], matrix[4, 0], matrix[5, 0] = GC_content(seq)

    return matrix


def onehot_NCP_DPCP_TPCP_encoding(sequence):
    '''One-hot+NCP+DPCP encoding.'''
    base = 'ATGC'
    matrix_lenth = 3000
    matrix = np.zeros([matrix_lenth, 19], dtype=np.float32)
    for i in range(len(sequence)):
        for j in range(4):
            if sequence[i] == base[j]:
                matrix[i, j] = np.float32(1)
            else:
                matrix[i, j] = np.float32(0)
        if sequence[i] == base[0]:
            matrix[i, 4] = np.float32(1)
            matrix[i, 5] = np.float32(1)
            matrix[i, 6] = np.float32(1)
        elif sequence[i] == base[1]:
            matrix[i, 4] = np.float32(0)
            matrix[i, 5] = np.float32(1)
            matrix[i, 6] = np.float32(0)
        elif sequence[i] == base[2]:
            matrix[i, 4] = np.float32(1)
            matrix[i, 5] = np.float32(0)
            matrix[i, 6] = np.float32(0)
        elif sequence[i] == base[3]:
            matrix[i, 4] = np.float32(0)
            matrix[i, 5] = np.float32(0)
            matrix[i, 6] = np.float32(1)
        else:
            matrix[i, 4] = np.float32(0)
            matrix[i, 5] = np.float32(0)
            matrix[i, 6] = np.float32(0)
    for i in range(len(sequence) - 1):
        couple = sequence[i] + sequence[i + 1]
        if couple in DPCP:
            properties = DPCP[couple]
            for m in range(6):
                matrix[i, 7 + m] += np.float32(properties[m] / 2)
                matrix[i + 1, 7 + m] += np.float32(properties[m] / 2)
        else:
            print(couple)
            print('Wrong:no couple in DPCP!')
            sys.exit()
    for i in range(len(sequence) - 2):
        couple = sequence[i] + sequence[i + 1] + sequence[i + 2]
        if couple in TPCP:
            properties = TPCP[couple]
            for m in range(6):
                matrix[i, 13 + m] += np.float32(properties[m] / 3)
                matrix[i + 1, 13 + m] += np.float32(properties[m] / 3)
                matrix[i + 2, 13 + m] += np.float32(properties[m] / 3)
        else:
            print(couple)
            print('Wrong:no couple in TPCP!')
            sys.exit()
    return np.transpose(matrix)


def all_encoding(sequence):
    # input:[seq](len of seq is in 201~3000)
    c_matrix = onehot_NCP_DPCP_TPCP_encoding(sequence)
    d_matrix = discrete_encoding(sequence)
    if c_matrix.shape[0] == d_matrix.shape[0]:
        all_matrix = np.hstack((c_matrix, d_matrix))
    else:
        print('Encoding wrong!')
        sys.exit()
    return all_matrix


def predict(input, model, device):
    '''
    Overloading DeepLncPro for inference
    :return: Model inference results.
    '''
    model.to(device)
    with torch.no_grad():
        input = input.to(device)
        output = model(input)
        pred = output.detach().cpu().numpy().reshape(output.shape[0])

    return pred


def load_dataset(data_path):
    '''
    Load the sample sequence from the input file. Perform the process of encoding, sampling, etc.
    :param data_path:Input file.
    :return:Information used to predict and write to the file.
            Include: sample name, sequence, location information, and matrix.
    '''
    name_list, sequence_list, matrix_list = [], [], []
    with open(data_path, 'rt') as f:
        lines = f.readlines()
        if lines[0][0] != '>':
            print('Please check the input file format!')
            sys.exit()
        k = 0
        sequence = ''
        for line in lines:
            if line[0] == '>':
                name_list.append(line[1:].strip())
                if k == 1:
                    sequence_list.append(sequence)
                    sequence = ''
                k = 1
            else:
                sequence = sequence + line.strip()
        sequence_list.append(sequence)

        name_output, length_output, sequence_output = [], [], []
        matrix_oneseq_all = []
        sequence_output = sequence_list
        name_output = name_list
        for i in range(len(name_list)):
            seq = sequence_list[i]
            length = len(seq)
            length_output.append(str(length))
            if length < 200:
                print('%s is less than 200bp and cannot be predicted!' % (name_list[i]))
                sys.exit()
            elif length > 3000:
                seq = seq[0:3000]
            matrix_list.append(all_encoding(seq))
    matrix_input = np.asarray([i for i in matrix_list], dtype=np.float32)
    matrix_input = torch.from_numpy(matrix_input)
    return name_output, length_output, sequence_output, matrix_input


class lncRNAdeep(nn.Module):
    """
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
        self.feature1 = 10
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
        self.rnn_linear = nn.Linear(self.hidden_size, self.feature1)

    def forward(self, x):
        if self.ENCODING_TYPE in [1, 2, 3, 4, 5]:
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
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            x1 = x1.view(x1.size(0), -1)
            x1 = F.relu(self.liner1(x1))
            x1 = F.relu(self.liner3(x1))
            x3 = F.relu(self.linear_only_x1(x1))
            x3 = self.linear_a2(x3)
        elif self.ENCODING_TYPE == 6:
            x1 = x[:, 0:19, 0:3000]
            x2 = x[:, 0:6, 3000]
            x2 = F.relu(self.linear_1(x2))
            x2 = F.relu(self.linear_2(x2))
            x3 = F.relu(self.linear_only_x2(x2))
            x3 = self.linear_a2(x3)
        elif self.ENCODING_TYPE == 7:
            x1 = x[:, 0:19, 0:3000]
            x2 = x[:, 0:6, 3000]
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
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


def prediction_process(input_file, species):
    '''
    Prediction of input sequences using DeepLncPro model.
    :return: List of predicted results.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('model/best_model.pkl', map_location='cpu')
    name_list, length_list, sequence_list, data_list = load_dataset(input_file)
    pre_list = predict(data_list, model, device)
    out_list = [name_list, length_list, sequence_list, pre_list]
    return out_list


def write_outputFile(output_list, outputFile, threshold):
    '''Write the results to outputFile.'''
    f = open(outputFile, 'w', encoding="utf-8")
    name_list = output_list[0]
    length_list = output_list[1]
    sequence_list = output_list[2]
    pre_list = output_list[3]
    prediction_list = ['LncRNA' if i > threshold else 'Non-lncRNA' for i in pre_list]
    out1 = open("js/out1.txt", "r")
    out2 = open("js/out2.txt", "r")
    s = out1.read()
    f.write(s)
    out1.close()
    for i in range(len(name_list)):
        f.write('<tr>' + '\n')
        f.write('<td>' + str(i + 1) + '</td>' + '\n')
        f.write('<td>' + name_list[i] + '</td>' + '\n')
        f.write('<td>' + length_list[i] + '</td>' + '\n')
        f.write('<td>' + str(round(pre_list[i], 4)) + '</td>' + '\n')
        f.write('<td>' + str(prediction_list[i]) + '</td>' + '\n')
        f.write('<td>' + sequence_list[i] + '</td>' + '\n')
        f.write('</tr>' + '\n')
    s = out2.read()
    f.write(s)
    out2.close()
    f.close()


def preprocess(inputFile, outputFile, species, threshold):
    '''Predicting lncRNA promoters and writing output files.'''
    print('DeepLnc-CNN starts running...', datetime.datetime.now())
    out_list = prediction_process(inputFile, species)
    write_outputFile(out_list, outputFile, threshold)
    print('Mission accomplished.', datetime.datetime.now())
