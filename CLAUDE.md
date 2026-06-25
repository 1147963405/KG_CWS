# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

医疗领域中文分词工具（Medical KG Chinese Word Segmentation），基于 BERT-LSTM-CRF 序列标注模型。原是 CMeKG 项目的子模块。

## Architecture

```
model_cws/
  __init__.py          # 导出 CRF, BERT_LSTM_CRF
  crf.py               # CRF 层 (forward algorithm / Viterbi decode / loss)
  bert_lstm_crf.py     # BERT-LSTM-CRF 主模型
  cws_constant.py      # 标签映射 (BMES)、超参、路径常量
  train_cws.py         # 训练 + 验证 + 测试
  medical_cws.py       # 推理封装 (单句/文件/长文本)
  demo.py              # 交互式 demo
  test/
    my_test.txt        # 输入测试文本
    outt.txt           # 分词输出示例
model/medical_cws/     # 预训练 BERT-base 权重 + CWS 模型权重
```

### Label Scheme (BMES)
- `B` - Begin (词首), `M` - Middle (词中), `E` - End (词尾), `S` - Single (单字词)
- CRF 约束保证合法序列（如 B→M/E, E/S→B/S 等）

### Model Pipeline
1. **BERT** (HuggingFace `BertModel`) → token-level embeddings (768d)
2. **BiLSTM** (2 layers) → sequence context encoding
3. **Linear projections** → tag scores
4. **CRF** → Viterbi decoding + negative log-likelihood loss

## Key Usage

### 交互式 Demo
```bash
cd model_cws && python demo.py
```

### 训练
```bash
cd model_cws && python train_cws.py
```

### 推理 API
```python
from model_cws.medical_cws import medical_seg
seg = medical_seg()
seg.predict_sentence("高血压病人不可食用阿莫西林等药物")
seg.predict_file("input.txt", "output.txt")
seg.predict_long_text("长文本...", segment_size=140, overlap=20)
```

## Known Issues (hardcoded paths)

`medical_cws.py` and `train_cws.py` contain hardcoded absolute paths (`D:/CMe_KG/CMeKG/...`). When adapting to this repo, update:
- `medical_cws.py:16` → `self.NEWPATH`
- `medical_cws.py:24,27` → vocab path and BERT model path
- `cws_constant.py` → data/model paths (commented out currently)
- `train_cws.py` → data files, BERT config, save dir

## Dependencies

- Python 3.7+
- PyTorch
- transformers (HuggingFace)
- tqdm
- `utils.utils` (from parent CMeKG project) 提供 `load_vocab`, `load_data`, `recover_label`, `get_f1`
