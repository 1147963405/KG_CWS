# coding:utf-8
import codecs
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from utils.utils import load_vocab
from cws_constant import *

from model_cws import BERT_LSTM_CRF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class medical_seg(object):
    def __init__(self):
        self.NEWPATH = medical_tool_model
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
            self.use_cuda = True
        else:
            self.device = torch.device("cpu")
            self.use_cuda = False

        self.vocab = load_vocab(vocab_file)
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        self.model = BERT_LSTM_CRF(medical_bert, tagset_size, 768, 200, 2,
                              dropout_ratio=0.5, dropout1=0.5, use_cuda=use_cuda)

        if use_cuda:
            self.model.cuda()

    def from_input(self, input_str):
        # 单行的输入
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        text = ['[CLS]'] + [x for x in input_str if x in self.vocab] + ['[SEP]']
        raw_text.append(text)
        cur_len = len(text)
        raw_textid = [self.vocab[x] for x in text] + [0] * (max_length - cur_len)
        textid.append(raw_textid)
        raw_textmask = [1] * cur_len + [0] * (max_length - cur_len)
        textmask.append(raw_textmask)
        textlength.append([cur_len])
        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength

    def from_txt(self, input_path):
        # 多行输入
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line) > 148:
                    line = line[:148]
                temptext = ['[CLS]'] + [x for x in line[:-1] if x in self.vocab] + ['[SEP]']
                cur_len = len(temptext)
                raw_text.append(temptext)

                tempid = [self.vocab[x] for x in temptext] + [0] * (max_length - cur_len)
                textid.append(tempid)
                textmask.append([1] * cur_len + [0] * (max_length - cur_len))
                textlength.append([cur_len])

        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength

    def recover_to_text(self, pred, raw_text):
        # 输入[标签list]和[原文list],batch为1
        pred = [i2l_dic[t.item()] for t in pred[0]]
        pred = pred[:len(raw_text)]
        pred = pred[1:-1]
        raw_text = raw_text[1:-1]
        raw = ""
        res = ""
        for tag, char in zip(pred, raw_text):
            res += char
            if tag in ["S", 'E']:
                res += ' '
            raw += char
        return raw, res

    def _predict_segment(self, seg_text):
        """对单个片段进行推理，返回 (结果字符串, 过滤后的字符数)"""
        raw_text, test_ids, test_masks, test_lengths = self.from_input(seg_text)
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

        for batch in test_loader:
            sentence, masks, lengths = batch
            if use_cuda:
                sentence = sentence.cuda()
                masks = masks.cuda()
            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            _, res = self.recover_to_text(predict_tags, raw_text[0])
            char_count = sum(1 for c in res if c != ' ')
            return res, char_count
        return '', 0

    def predict_long_text(self, text, segment_size=140, overlap=20):
        """长文本分词：分段推理 + 拼接，避免截断丢失信息"""
        if not text:
            return ''
        if len(text) <= segment_size:
            return self.predict_sentence(text)

        # 加载模型权重一次，所有片段共用
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location=self.device))
        self.model.eval()

        step = segment_size - overlap
        if step <= 0:
            step = segment_size // 2

        segments = []
        for i in range(0, len(text), step):
            seg = text[i:i + segment_size]
            if len(seg) < 2:
                continue
            segments.append((i, seg))

        results = []
        for offset, seg in segments:
            res, _ = self._predict_segment(seg)
            results.append((offset, res))

        # 拼接：第一段全取，后续每段跳过前半重叠区的字符数
        merged = results[0][1]
        for k in range(1, len(results)):
            prev_start, prev_res = results[k - 1]
            cur_start, cur_res = results[k]
            orig_overlap = segment_size - step  # 原始文本重叠字符数

            # 在 cur_res 中找到第 (orig_overlap + 1) 个非空格字符的位置
            count = 0
            cut = 0
            for j, ch in enumerate(cur_res):
                if ch != ' ':
                    count += 1
                    if count > orig_overlap:
                        cut = j
                        break
                cut = j + 1

            if not merged.endswith(' ') and not cur_res[cut:].startswith(' '):
                merged += ' '
            merged += cur_res[cut:]

        return merged

    def predict_sentence(self, sentence):
        if sentence == '':
            print("输入为空！请重新输入")
            return
        if len(sentence) > 148:
            print("输入句子过长，请输入小于148的长度字符！")
            sentence = sentence[:148]

        # 加载模型权重
        self.model.load_state_dict(torch.load(self.NEWPATH,map_location=self.device))
        self.model.eval()
        res, _ = self._predict_segment(sentence)
        return res

    def predict_file(self, input_file, output_file):
        # raw_text, test_ids, test_masks, test_lengths = self.from_txt("./data/raw_text.txt")
        raw_text, test_ids, test_masks, test_lengths = self.from_txt(input_file)

        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location={'cuda:0': str(self.device)}))
        self.model.eval()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        op_file = codecs.open(output_file, 'w', 'utf-8')
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            if use_cuda:
                sentence = sentence.cuda()
                masks = masks.cuda()

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()

            raw, res = self.recover_to_text(predict_tags, batch_raw_text)
            op_file.write(res + '\n')

        op_file.close()
        print('处理完成！')
        print("results have been stored in {}".format(output_file))


if __name__ == "__main__":

    meg = medical_seg()
    # meg.predict_file('./data/raw_text.txt', './data/out_raw.txt')
    res = meg.predict_sentence("肾上腺由皮质和髓质两个功能不同的内分泌器官组成，皮质分泌肾上腺皮质激素，髓质分泌儿茶酚胺激素。")
    print(res)
