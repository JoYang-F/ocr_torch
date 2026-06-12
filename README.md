# ocr_torch

基于 PyTorch 的轻量级 OCR 检测识别项目，支持 **DBNet + CRNN** 端到端推理及 ONNX 导出。

- **文本检测** — DBNet（支持 ResNet18/34 及 MobileNetV3 骨干）
- **文本识别** — CRNN（支持 MobileNetV3 / ResNet18/34 骨干 + SequenceEncoder + CTCHead）

---

## 环境依赖

- Python 3.8
- PyTorch 1.8.1
- onnxruntime（ONNX 推理）

## 项目结构

```text
ocr_torch/
├── config/                          # 配置文件
│   ├── train/                       # 训练配置（det.yml, rec.yml）
│   ├── predict/                     # 推理配置（det.yml, rec.yml）
│   ├── lite_ocr.yml                 # 端到端推理配置
│   └── load_conf.py                 # 配置加载器
├── data_loader/                     # 数据加载
│   ├── det/                         # 检测数据集（图像 + 标签）
│   ├── rec/                         # 识别数据集（图像 + 标签 + 字符映射）
│   ├── det_dataset.py               # 检测数据集（支持 txt / json）
│   ├── rec_dataset.py               # 识别数据集（支持 txt / json）
│   └── img_aug/                     # 图像增强
│       ├── operators.py             # 基础算子（归一化、缩放等）
│       ├── random_crop_data.py      # 随机裁剪
│       ├── make_binary_map.py       # 二值图生成
│       ├── make_threshold_map.py    # 阈值图生成
│       ├── rec_img_aug.py           # 识别数据增强
│       └── text_image_aug/          # 文本图像仿射变换
├── nets/                            # 网络模型
│   ├── det/                         # 检测网络
│   │   ├── dbnet.py                 # DBNet
│   │   ├── db_fpn.py                # 特征金字塔（FPN）
│   │   ├── db_head.py               # DB Head
│   │   ├── mobilenetv3.py           # MobileNetV3 骨干
│   │   ├── resnet.py                # ResNet 骨干
│   │   └── params_mapping.py        # 预训练参数映射
│   └── rec/                         # 识别网络
│       ├── rnn.py                   # CRNN
│       ├── mobilenet_v3.py          # MobileNetV3 骨干
│       ├── resnet.py                # ResNet 骨干
│       ├── sequence_encoder.py      # 序列编码器
│       └── ctc_head.py              # CTC 预测头
├── losses/                          # 损失函数
│   ├── loss.py                      # 基础损失（DiceLoss、L1Loss、BCELoss）
│   ├── det_loss.py                  # DBNet 损失（L1BalanceCELoss）
│   └── ctc_loss.py                  # CTC Loss
├── metrics/                         # 评估指标
│   ├── det_metric.py                # 检测评估
│   ├── eval_det_iou.py              # IoU 计算
│   └── rec_metric.py                # 识别评估（acc, norm_edit_dis）
├── postprocess/                     # 后处理
│   ├── det_postprocess.py           # DB 后处理
│   └── rec_postprocess.py           # CRNN 后处理
├── optimizer/                       # 优化器
│   ├── optim.py                     # Adam 优化器
│   └── learning_rate.py             # 学习率调度（Cosine Warmup）
├── logger/                          # 日志
├── utils/                           # 工具函数
├── train.py                         # 训练入口
├── predict.py                       # 预测入口（PyTorch / ONNX）
└── lite_ocr.py                      # 端到端检测 + 识别推理
```

## 使用说明

### 1. 文本检测模型训练（DBNet）

```bash
python train.py -c config/train/det.yml
```

骨干网络可选：`resnet34`（默认）、`resnet18`、`det_mobilenet_v3`

如需从上次中断恢复：

```bash
python train.py -c config/train/det.yml --resume
```

### 2. 文本识别模型训练（CRNN）

```bash
python train.py -c config/train/rec.yml
```

骨干网络可选：`rec_mobilenet_v3`（默认）、`rec_resnet18`、`rec_resnet34`

如需从上次中断恢复：

```bash
python train.py -c config/train/rec.yml --resume
```

### 3. 文本检测推理

```bash
python predict.py -c config/predict/det.yml
```

支持 PyTorch 模型直接推理或导出 ONNX 后推理（通过 `use_infer_model` 切换）。

### 4. 文本识别推理

```bash
python predict.py -c config/predict/rec.yml
```

### 5. 端到端检测 + 识别

```bash
python lite_ocr.py -c config/lite_ocr.yml
```

依次执行文本检测 → 仿射变换裁剪 → 文本识别，结果保存为可视化图像 + result.txt。

### 6. 断点续训

训练过程中每轮 epoch 结束自动保存 checkpoint 至 `save_pth_dir` 目录：

| checkpoint | 说明 |
|---|---|
| `latest.pth` | 每 epoch 自动更新，用于恢复训练 |
| `iter_epoch_N.pth` | 每隔 `save_epoch_iter` 轮定期保存 |
| `best_xxx.pth` | 验证集指标最优时保存 |

**中断恢复**：训练中按 `Ctrl+C` 会触发优雅退出，自动保存 `latest.pth`。下次启动时添加 `--resume` 即可从上次中断的 epoch 继续训练（包括模型权重、优化器状态、学习率调度器、global_step 等完整恢复）：

```bash
python train.py -c config/train/det.yml --resume
```

> `--resume` 等价于在配置中将 `init_pth_path` 指向 `latest.pth`，只是无需手动修改配置文件。

### ONNX 导出

预测时设置 `use_infer_model: true`，自动导出 ONNX 并加载推理。

## 配置说明

训练配置采用 `Architecture` 分层结构：

```yaml
Architecture:
  model_type: det                  # det / rec
  algorithm: DBNet                 # DBNet / CRNN
  Backbone:
    name: resnet34                 # 骨干网络
    pre_trained_dir:               # 预训练权重路径
  Neck:
    name: DBFPN                    # 颈部网络
    inner_channel: 96
  Head:
    name: DBHead                   # 检测头
    k: 50
```

## 参考文献

1. [DBNet: Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)
2. [CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition](https://arxiv.org/abs/1507.05717)
3. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
4. [DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch)

## 开源协议

本项目基于 MIT 协议开源。
