# CWS — 中文分词

基于 BERT-LSTM-CRF 的中文分词工具。

## 模型下载

由于依赖和模型文件较大，预训练模型和分词模型权重通过百度网盘下载：

链接:https://pan.baidu.com/s/1bU3QoaGs2IxI34WBx7ibMQ  密码:yhek

下载后放入 `model/medical_cws/` 目录。

## 目录结构

```
├── model/medical_cws/          # BERT 预训练模型 + 分词模型权重
│   ├── config.json             # BERT 模型配置
│   ├── pytorch_model.bin       # BERT 权重
│   ├── pytorch_model.pkl       # 训练好的 CWS 模型参数
│   ├── vocab.txt               # BERT 词表
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── model_cws/                  # 分词模块代码
│   ├── crf.py                  # CRF 层（前向算法 / Viterbi 解码 / 负对数似然损失）
│   ├── bert_lstm_crf.py        # BERT-LSTM-CRF 模型定义
│   ├── cws_constant.py         # 标签映射（BMES）、超参数、路径常量
│   ├── train_cws.py            # 训练 + 评估脚本
│   ├── medical_cws.py          # 推理封装类（单句 / 文件 / 长文本分段）
│   └── demo.py                 # 交互式 demo
└── model_cws/test/             # 测试样例
    ├── my_test.txt             # 输入文本
    └── outt.txt                # 分词输出示例
```

## 标签方案（BMES）

| 标签 | 含义 |
|------|------|
| B    | Begin — 词首字 |
| M    | Middle — 词中字 |
| E    | End — 词尾字 |
| S    | Single — 单字成词 |

CRF 层约束标签转移（如 B→M/E, E/S→B/S），保证输出合法序列。

## 依赖库

- Python 3.7+
- torch
- transformers（HuggingFace）
- tqdm
- numpy（`utils.utils` 中隐式依赖）

## 模型使用

### 配置

在 `medical_cws.py` 中修改模型路径：

```python
# medical_cws.py line 16-27
self.NEWPATH = 'model/medical_cws/pytorch_model.pkl'  # 分词模型权重
# BERT 路径和词表路径也需要对应调整
```

### 交互式 Demo

```bash
cd model_cws && python demo.py
```

### 代码调用

```python
from model_cws.medical_cws import medical_seg

seg = medical_seg()

# 单句分词
res = seg.predict_sentence("我爱自然语言处理")
print(res)
# 输出: 我 爱 自然 语言 处理

# 长文本分词（自动分段 + 重叠拼接，避免截断损失）
res = seg.predict_long_text("长文本内容...", segment_size=140, overlap=20)

# 文件批处理
seg.predict_file("input.txt", "output.txt")
```

### 长文本分段策略

`predict_long_text()` 将长文本按 `segment_size` 分段，相邻段有 `overlap` 字的重叠，推理后根据重叠加权拼接，避免序列截断导致的分词信息丢失。

## 模型训练

调整超参和路径在 `cws_constant.py` 中：

```python
# cws_constant.py
max_length = 150
batch_size = 24
epochs = 10
```

训练数据格式：每行一个已分词语句，词之间以空格分隔。

```bash
cd model_cws && python train_cws.py
```

训练过程中自动保存验证集上 F1 最优的模型，并输出测试集指标。

## 模型架构

1. **BERT** — HuggingFace `BertModel`，输出 768 维上下文表示
2. **BiLSTM** — 2 层双向 LSTM，捕捉序列依赖
3. **Linear Projection** — 将 LSTM 输出映射到标签空间
4. **CRF** — 条件随机场，建模标签转移约束，Viterbi 解码

## 路径配置

所有路径集中定义在 `cws_constant.py` 中，基于 `BASE_DIR`（项目根目录）自动拼接，无需手动修改。关键路径：

| 常量 | 默认值 |
|------|--------|
| `vocab_file` | `model/medical_cws/vocab.txt` |
| `medical_bert` | `model/medical_cws/` |
| `medical_tool_model` | `model/medical_cws/pytorch_model.pkl` |
| `train_file` | `data/train.txt` |
| `dev_file` | `data/dev.txt` |
| `test_file` | `data/test.txt` |
| `save_model_dir` | `output/model/` |
