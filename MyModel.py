import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import re
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

max_word = 10000#保留最高频的10000词构建单词表
max_len = 300 #句子的最大长度
word_count = {}#字典，词和词出现的次数

#对句子进行处理，转小写，整理标点
def solveStr(string):
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", string)#处理非字母数字
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", ' not', string)
    string = re.sub(r"\'re", "are", string)
    string = re.sub(r"i\'m", "i am", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\+", " \+ ", string)
    string = re.sub(r"\-", " \- ", string)
    string = re.sub(r"\=", " \= ", string)
    string = re.sub(r"'", " ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r" e g ", " eg ", string)
    string = re.sub(r" b g ", " bg ", string)
    string = re.sub(r"e - mail", "email", string)
    string = re.sub(r"\s{2,}", " ", string)
    #string = re.sub(r"")

    return string.strip()


#分词方法
def tokenizer(sentence):
    return sentence.split()

#数据预处理
def data_process(txt_path, txt_dir):
    print("----data preprocess----")
    filepro = open(txt_path, 'w', encoding='utf-8')
    for root, dirs, tp in os.walk(txt_dir):#获取txt_dir下各文件夹名称
        #print(root)
        #print(dirs)
        #print(tp)
        for subdirs in dirs:#遍历文件夹,dirs是文件夹列表
            #print(subdirs)
            a_dir = os.path.join(root, subdirs)#获取文件夹完整路径不包括初始的文件夹
            #print(a_dir)
            txt_list = os.listdir(a_dir)#获取路径下的所有文件构成列表
            #print(txt_list)
            tag = os.path.split(a_dir)[-1]#获取路径最后一个文件夹的名字即标签
            if tag == 'pos':
                label = '1'
            if tag == 'neg':
                label = '0'
            if tag == 'unsup':
                continue

            for i in range(len(txt_list)):
                if not txt_list[i].endswith('txt'):#判断文件是否是txt类型
                    continue
                file = open(os.path.join(a_dir, txt_list[i]), 'r', encoding='utf8')
                raw = file.readline()#读取一行
                solve_raw = solveStr(raw)#处理句子
                tokens = tokenizer(solve_raw)#文本分词
                #统计词出现的次数
                for token in tokens:
                    if token in word_count.keys():
                        word_count[token] += 1
                    else:
                        word_count[token] = 1
                filepro.write(label + ' ' + solve_raw +'\n')#处理后的句子和标签写入新文件中
                file.close()
    filepro.close()

    print("----buildvocabulary----")#建立词量表
    vocab = {"<unk>": 0, "<pad>": 1}
    #词排序
    word_count_sort = sorted(word_count.items(), key=lambda item : item[1], reverse=True)

    #词编号,利用one-hot的方式
    word_vocab_length = 0
    for word in word_count_sort:
        if word[0] not in vocab.keys():
            vocab[word[0]] = len(vocab)#编号
            word_vocab_length += 1
        if word_vocab_length >= max_word:
            break
    return vocab

def txt_transform(sentence_list, vocab):
    sentence_index_list = []
    for sentence in sentence_list:#遍历句子列表
        #将句子中的每个单词转换成其在vocab中的id
        sentence_idx = [vocab[token] if token in vocab.keys()
                       else vocab['<unk>'] for token in tokenizer(sentence)]
        if len(sentence_idx) < max_len:#句子长度不够用pad的索引填充
            for i in range(max_len - len(sentence_idx)):
                sentence_idx.append(vocab['<pad>'])
        sentence_idx = sentence_idx[:max_len]#截取前max_len长度
        sentence_index_list.append(sentence_idx)

    return torch.LongTensor(sentence_index_list)#将所有句子的索引转换为tensor

class MyDataset(Dataset):
    def __init__(self, txt_path):
        file = open(txt_path, 'r', encoding='utf-8')
        self.text_tag = file.readlines()
        #print(self.text_tag[0])
        file.close()

    def __getitem__(self, index):
        line = self.text_tag[index]
        label = int(line[0])
        text = line[2:-1]
        return text, label

    def __len__(self):
        return len(self.text_tag)

class MyLSTM(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(MyLSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False)
        #self.decoder = nn.Linear(num_hiddens, 2)
        #self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(num_hiddens, 2),
            nn.ReLU()
        )

    def forward(self, x):
        #x的形状是（批量大小，词数）和参数维度相反因此需要将0，1维度互换过来
        embeddings = self.embedding(x.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        encoding = outputs[-1]
        outs = self.fc(encoding)
        #outs = self.decoder(encoding)
        #outs = self.softmax(outs)
        return outs

class MyGRU(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(MyGRU, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.GRU(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False)
        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(num_hiddens, 2),
            nn.ReLU()
        )
        #self.decoder = nn.Linear(num_hiddens, 2)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x的形状是（批量大小，词数）和参数维度相反因此需要将0，1维度互换过来
        embeddings = self.embedding(x.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        encoding = outputs[-1]
        outs = self.fc(encoding)
        #outs = self.decoder(encoding)
        #outs = self.softmax(outs)
        return outs


train_epoch_loss = []
train_acc = []
test_epoch_loss = []
test_acc = []

def train(model, trainloader, vocab, optimizer, loss_fun):
    print("----train----")
    all_loss = 0.0
    correct = 0.0
    num = 0.0
    model.train()
    for step, (text, label) in enumerate(tqdm(trainloader)):
        text_x = txt_transform(text, vocab).cuda()
        label_y = label.cuda()
        optimizer.zero_grad()
        output = model(text_x)
        _, predicted = torch.max(output, 1)
        #print(label_y)
        correct += (predicted == label_y).sum().item()
        loss = loss_fun(output, label_y)
        print(step, loss.item())
        all_loss += loss.item() * label_y.size(0)
        num += label_y.size(0)
        #print(label_y.size(0))
        loss.backward()
        optimizer.step()

    acc = correct / num
    epoch_loss = all_loss / num
    train_epoch_loss.append(epoch_loss)
    train_acc.append(acc)


def test(model, testloader, vocab, best_acc):
    print('----test----')
    model.eval()
    all_loss = 0.0
    correct = 0.0
    num = 0.0
    class_acc = [[], []]  # 0-对原neg的判断，1-对原pos的判断
    with torch.no_grad():
        for step, (text, label) in enumerate(tqdm(testloader)):
            text_x = txt_transform(text, vocab).cuda()
            label_y = label.cuda()
            output = model(text_x)

            _, predicted = torch.max(output, 1)
            tp = predicted.data
            ty = label_y.data
            for i in range(len(ty)):
                class_acc[ty[i]].append(tp[i])
            # print(label_y)
            correct += (predicted == label_y).sum().item()
            loss = loss_fun(output, label_y)
            #print(step, loss.item())
            all_loss += loss.item() * label_y.size(0)
            num += label_y.size(0)

        acc = correct / num * 100
        epoch_loss = all_loss / num
        test_epoch_loss.append(epoch_loss)
        test_acc.append(acc)
        print("acc is %.4f" % acc)

        if acc > best_acc:
            best_acc = acc
            neg_true_num = 0.0
            pos_true_num = 0.0
            for item in class_acc[0]:
                if item == 0:
                    neg_true_num += 1
            neg_acc = neg_true_num / (len(class_acc[0])) * 100

            for item in class_acc[1]:
                if item == 1:
                    pos_true_num += 1
            pos_acc = pos_true_num / len(class_acc[1]) * 100

            with open('./Data/best_acc.txt', 'w') as f0:
                f0.write('best acc: %.4f' % best_acc)
                f0.write('\n')
                f0.write('neg acc: %.4f' % neg_acc)
                f0.write('\n')
                f0.write('pos acc: %.4f' % pos_acc)
                f0.flush()
                f0.close()
    return best_acc






train_dir = './aclImdb/train'  # 原训练集文件地址
train_path = './train.txt'  # 预处理后的训练集文件地址
test_dir = './aclImdb/test'
test_path = './test.txt'
vocab = data_process(train_path, train_dir)
#np.save('vocab.npy', vocab)
#vocab = np.load('vocab.npy', allow_pickle=True).item()
data_process(test_path, test_dir)

train_data = MyDataset(train_path)
test_data = MyDataset(test_path)
trainloader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)
print(len(trainloader))
model = MyLSTM(vocab=vocab, embed_size=300, num_hiddens=128, num_layers=2)
#model = MyGRU(vocab=vocab, embed_size=300, num_hiddens=128, num_layers=2)
model.cuda()
LR = 5e-3
#LR = 0.001
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_acc = 78
for tepoch in range(0, 30):
    print("epoch is %d" % (tepoch + 1))
    train(model, trainloader, vocab, optimizer, loss_fun)
    best_acc = test(model, testloader, vocab, best_acc)

with open("./Data/LSTM_train_epoch_loss.txt", 'w') as f1:
    f1.write(str(train_epoch_loss))
    f1.flush()
    f1.close()

with open('./Data/LSTM_train_acc.txt', 'w') as f2:
    f2.write(str(train_acc))
    f2.flush()
    f2.close()

with open('./Data/LSTM_test_acc.txt', 'w') as f3:
    f3.write(str(test_acc))
    f3.flush()
    f3.close()

with open('./Data/LSTM_test_epoch_loss.txt', 'w') as f4:
    f4.write(str(train_acc))
    f4.flush()
    f4.close()