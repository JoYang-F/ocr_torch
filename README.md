# ocr_torch

基于 PyTorch 实现的轻量级文字检测识别项目，支持 **DBNet + CRNN** 端到端推理和 ONNX 导出。

- 文本检测：DBNet（支持 ResNet34 / MobileNetV3 骨干网络）
- 文本识别：CRNN（支持 MobileNetV3 / ResNet 骨干网络 + SequenceEncoder + CTCHead）

---

## 环境依赖

- Python 3.7+
- PyTorch 1.8+
- onnxruntime（ONNX 推理）

## 项目结构

```text
ocr_torch/
├── config/                    # 配置文件
│   ├── train/                 # 训练配置（det.yml, rec.yml）
│   ├── predict/               # 预测配置（det.yml, rec.yml）
│   ├── lite_ocr.yml           # 端到端推理配置
│   └── load_conf.py           # 配置加载器
├── data_loader/               # 数据加载
│   ├── det/                   # 检测数据集（train/val/test + 标签）
│   ├── rec/                   # 识别数据集（train/val/test + 标签 + 字符映射）
│   ├── det_dataset.py         # DBNet 数据集（支持 txt/json 标签）
│   ├── rec_dataset.py         # CRNN 数据集（支持 txt/json 标签）
│   └── img_aug.py             # 图像增强
├── nets/                      # 网络模块
│   ├── det/                   # 检测网络
│   │   ├── dbnet.py           # DBNet 模型
│   │   ├── db_fpn.py          # 特征金字塔网络（FPN）
│   │   ├── db_head.py         # DB Head
│   │   └── mobilenetv3.py     # MobileNetV3 检测骨干
│   ├── rec/                   # 识别网络
│   │   ├── rnn.py             # CRNN 模型
│   │   ├── mobilenet_v3.py    # MobileNetV3 识别骨干
│   │   ├── resnet.py          # ResNet 识别骨干
│   │   ├── sequence_encoder.py # 序列编码器
│   │   └── ctc_head.py        # CTC 预测头
│   └── __init__.py            # 模型构建入口
├── losses/                    # 损失函数
│   ├── det_loss.py            # DBNet 损失（L1BalanceCELoss）
│   └── ctc_loss.py            # CTC Loss
├── metrics/                   # 评估指标
│   ├── det_metric.py          # 检测指标
│   └── rec_metric.py          # 识别指标（acc, norm_edit_dis）
├── postprocess/               # 后处理
│   ├── det_postprocess.py     # DB 后处理
│   └── rec_postprocess.py     # CRNN 后处理
├── optimizer/                 # 优化器（Adam + Cosine Warmup）
├── logger/                    # 日志
├── utils/                     # 工具函数
├── train.py                   # 训练入口
├── predict.py                 # 预测入口（支持 PyTorch / ONNX）
└── lite_ocr.py                # 端到端检测+识别推理
```

## 使用说明

### 1. 文本检测模型训练（DBNet）

```bash
python train.py -c config/train/det.yml
```

骨干网络可选：`resnet34`（默认）、`det_mobilenet_v3`

### 2. 文本识别模型训练（CRNN）

```bash
python train.py -c config/train/rec.yml
```

骨干网络可选：`rec_mobilenet_v3`（默认）、`rec_resnet18`、`rec_resnet34`

### 3. 文本检测推理

```bash
python predict.py -c config/predict/det.yml
```

支持 PyTorch 模型和 ONNX 两种推理方式（通过 `use_infer_model` 配置切换）。

### 4. 文本识别推理

```bash
python predict.py -c config/predict/rec.yml
```

### 5. 端到端检测+识别推理

```bash
python lite_ocr.py -c config/lite_ocr.yml
```

先检测文本位置，再逐个裁剪识别，结果保存为图片可视化 + result.txt。

### ONNX 导出

预测时设置 `use_infer_model: true` 会自动将 PyTorch 模型导出为 ONNX 并推理。

## 配置文件说明

训练配置采用 `Architecture` 格式组织：

```yaml
Architecture:
  model_type: det           # det / rec
  algorithm: DBNet          # DBNet / CRNN
  Backbone:
    name: resnet34          # 骨干网络名称
  Neck:
    name: DBFPN             # FPN 颈部网络
    inner_channel: 96
  Head:
    name: DBHead            # 检测头
    k: 50
```

## 主要参考文献

1. [DBNet: Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)
2. [CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition](https://arxiv.org/abs/1507.05717)
3. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
4. [DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch)
