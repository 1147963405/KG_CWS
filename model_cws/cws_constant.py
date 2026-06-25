# coding: utf-8
import os

l2i_dic = {"S":0, "B":1, "M":2, "E":3, "<pad>":4, "<start>":5, "<eos>":6}

i2l_dic = {0:"S", 1:"B", 2:"M", 3:"E", 4:"<pad>", 5:"<start>", 6:"<eos>"}

# 超参
max_length = 150
batch_size = 24
epochs = 10
tagset_size = len(l2i_dic)
use_cuda = False

# 路径（相对于项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_file = os.path.join(BASE_DIR, 'data', 'train.txt')
dev_file = os.path.join(BASE_DIR, 'data', 'dev.txt')
test_file = os.path.join(BASE_DIR, 'data', 'test.txt')
medical_bert = os.path.join(BASE_DIR, 'model', 'medical_cws')
vocab_file = os.path.join(BASE_DIR, 'model', 'medical_cws', 'vocab.txt')
save_model_dir = os.path.join(BASE_DIR, 'output', 'model')
medical_tool_model = os.path.join(BASE_DIR, 'model', 'medical_cws', 'pytorch_model.pkl')
