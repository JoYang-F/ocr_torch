#!/usr/bin/env python
"""
独立的 ONNX 模型导出脚本

从训练好的 .pth 检查点中构建模型并导出为 ONNX 格式。
需要 PyTorch 环境和 nets/ 模块。

用法：
    # 导出检测模型 (DBNet + ResNet34)
    python export.py det \
        --pth ./output/model_det/best_hmean.pth \
        --output ./deploy/models/DBNet.onnx \
        --backbone resnet34

    # 导出检测模型 (DBNet + MobileNetV3)
    python export.py det \
        --pth ./output/model_det/best_hmean.pth \
        --output ./deploy/models/DBNet.onnx \
        --backbone det_mobilenet_v3

    # 导出识别模型 (CRNN + MobileNetV3)
    python export.py rec \
        --pth ./output/model_rec/best_acc.pth \
        --output ./deploy/models/CRNN.onnx \
        --backbone rec_mobilenet_v3 \
        --char_json ./data_loader/rec/num_chars_38.json

    # 从训练配置文件读取模型架构
    python export.py rec \
        --config ./config/train/rec.yml \
        --pth ./output/model_rec/best_acc.pth \
        --output ./deploy/models/CRNN.onnx
"""

import os
import sys
import argparse
import json

# 确保能导入 nets/ 模块
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import onnx
import yaml


# ============================================================
# 模型构建
# ============================================================

def _build_det_model(backbone_name: str, inner_channel: int = 96, k: int = 50,
                     pre_trained_dir: str = "", **backbone_kwargs):
    """构建 DBNet 检测模型"""
    from nets.det.dbnet import DBNet

    backbone_conf = {"name": backbone_name, "pre_trained_dir": pre_trained_dir}
    backbone_conf.update(backbone_kwargs)

    model = DBNet(inner_channel=inner_channel, k=k, backbone=backbone_conf)
    return model


def _build_rec_model(backbone_name: str, classes_num: int,
                     rnn_type: str = "GRU", hidden_size: int = 48,
                     num_layers: int = 2, bidirectional: bool = True,
                     pre_trained_dir: str = "", **backbone_kwargs):
    """构建 CRNN 识别模型"""
    from nets.rec.rnn import CRNN

    backbone_conf = {"name": backbone_name, "pre_trained_dir": pre_trained_dir}
    backbone_conf.update(backbone_kwargs)

    model = CRNN(
        classes_num=classes_num,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        backbone=backbone_conf,
    )
    return model


# ============================================================
# 导出逻辑
# ============================================================

def export_det(model: torch.nn.Module, output_path: str, opset: int = 11):
    """导出 DBNet 为 ONNX。

    注意：必须设置 model.eval()，确保 forward 只输出 prob，
    而非训练时的 (prob, thresh, binary_thresh) 三通道。
    """
    model.eval()

    # DBNet 输入：动态 H, W（最小 32 的倍数更稳定，这里用 224 作为示例）
    dummy = torch.randn(1, 3, 224, 224, requires_grad=False)

    # 先用实际数据跑一次，确认输出形状
    with torch.no_grad():
        out = model(dummy)
    print(f"[DET] 输入 shape: {tuple(dummy.shape)} → 输出 shape: {tuple(out.shape)}")

    dynamic_axes = {
        "input":  {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    }

    torch.onnx.export(
        model=model,
        args=dummy,
        f=output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # 校验
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[DET] ✅ ONNX 模型已导出: {output_path}")


def export_rec(model: torch.nn.Module, output_path: str, opset: int = 11):
    """导出 CRNN 为 ONNX。

    输入： (N, 3, 32, W)  — 高度固定 32，宽度可变
    输出： (T, N, classes_num) — T 随输入宽度变化
    """
    model.eval()

    dummy = torch.randn(1, 3, 32, 320, requires_grad=False)

    with torch.no_grad():
        out = model(dummy)
    print(f"[REC] 输入 shape: {tuple(dummy.shape)} → 输出 shape: {tuple(out.shape)}")

    dynamic_axes = {
        "input":  {0: "batch_size", 3: "width"},
        "output": {0: "time_steps", 1: "batch_size"},
    }

    torch.onnx.export(
        model=model,
        args=dummy,
        f=output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[REC] ✅ ONNX 模型已导出: {output_path}")


# ============================================================
# 权重加载
# ============================================================

def load_checkpoint(model: torch.nn.Module, pth_path: str, device: str = "cpu"):
    """从训练检查点加载模型权重。

    兼容 train.py 保存的格式：{"state_dict": ..., "epoch": ..., ...}
    """
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"检查点文件不存在: {pth_path}")

    ckpt = torch.load(pth_path, map_location=torch.device(device))

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print(f"[INFO] 从检查点加载 (epoch={ckpt.get('epoch', '?')}, "
              f"best_epoch={ckpt.get('best_epoch', '?')})")
    else:
        # 可能是裸的 state_dict
        state_dict = ckpt

    # 处理 DistributedDataParallel 前缀 "module."
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] 缺失的键 ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[WARN] 多余的键 ({len(unexpected)}): {unexpected[:5]}...")

    print(f"[INFO] 权重加载完成")
    return model


# ============================================================
# 配置解析
# ============================================================

def parse_config(config_path: str) -> dict:
    """从训练 YAML 配置中提取模型架构参数"""
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    arch = conf.get("Architecture", conf.get("model", {}))
    algo = arch.get("algorithm", arch.get("name", ""))

    backbone = arch.get("Backbone", arch.get("backbone", {}))
    neck = arch.get("Neck", {})
    head = arch.get("Head", {})

    return {
        "algorithm": algo,
        "backbone_name": backbone.get("name", ""),
        "backbone_kwargs": {k: v for k, v in backbone.items() if k != "name"},
        "inner_channel": neck.get("inner_channel", 96),
        "k": head.get("k", 50),
        "rnn_type": neck.get("rnn_type", "GRU"),
        "hidden_size": neck.get("hidden_size", 48),
        "num_layers": neck.get("num_layers", 2),
        "bidirectional": neck.get("bidirectional", True),
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="导出 OCR 模型为 ONNX 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="task", required=True, help="任务类型: det | rec")

    # ---- 公共参数 ----
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pth", required=True, help="训练检查点 .pth 路径")
    common.add_argument("--output", required=True, help="输出 ONNX 文件路径")
    common.add_argument("--config", default=None, help="训练 YAML 配置文件（自动解析模型架构）")
    common.add_argument("--opset", type=int, default=11, help="ONNX opset 版本 (默认: 11)")
    common.add_argument("--device", default="cpu", help="设备 (默认: cpu)")

    # ---- det 子命令 ----
    det_parser = sub.add_parser("det", parents=[common], help="导出 DBNet 检测模型")
    det_parser.add_argument("--backbone", default="resnet34",
                            choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                                     "det_mobilenet_v3"],
                            help="骨干网络 (默认: resnet34)")
    det_parser.add_argument("--inner_channel", type=int, default=96,
                            help="FPN 内部通道数 (默认: 96)")
    det_parser.add_argument("--k", type=int, default=50,
                            help="DB 二值化系数 (默认: 50)")

    # ---- rec 子命令 ----
    rec_parser = sub.add_parser("rec", parents=[common], help="导出 CRNN 识别模型")
    rec_parser.add_argument("--backbone", default="rec_mobilenet_v3",
                            choices=["rec_mobilenet_v3", "rec_resnet18", "rec_resnet34", "rec_resnet50"],
                            help="骨干网络 (默认: rec_mobilenet_v3)")
    rec_parser.add_argument("--char_json", required=True,
                            help="字符映射表 JSON 路径 (用于确定 classes_num)")
    rec_parser.add_argument("--rnn_type", default="GRU", choices=["GRU", "LSTM"],
                            help="RNN 类型 (默认: GRU)")
    rec_parser.add_argument("--hidden_size", type=int, default=48,
                            help="RNN 隐藏层大小 (默认: 48)")
    rec_parser.add_argument("--num_layers", type=int, default=2,
                            help="RNN 层数 (默认: 2)")
    rec_parser.add_argument("--bidirectional", type=bool, default=True,
                            help="是否双向 (默认: True)")

    args = parser.parse_args()

    # 如果提供了训练配置文件，解析架构参数
    if args.config:
        cfg = parse_config(args.config)
        print(f"[INFO] 从配置文件解析: algorithm={cfg['algorithm']}")
        # 用配置文件参数作为默认值（CLI 参数优先）
        if args.task == "det":
            if not getattr(args, "_backbone_set", False):
                args.backbone = cfg["backbone_name"] or args.backbone
            args.inner_channel = cfg.get("inner_channel", args.inner_channel)
            args.k = cfg.get("k", args.k)
        elif args.task == "rec":
            if not getattr(args, "_backbone_set", False):
                args.backbone = cfg["backbone_name"] or args.backbone
            args.rnn_type = cfg.get("rnn_type", args.rnn_type)
            args.hidden_size = cfg.get("hidden_size", args.hidden_size)
            args.num_layers = cfg.get("num_layers", args.num_layers)
            args.bidirectional = cfg.get("bidirectional", args.bidirectional)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 构建模型
    if args.task == "det":
        print(f"[INFO] 构建 DBNet(backbone={args.backbone}, inner_channel={args.inner_channel}, k={args.k})")
        model = _build_det_model(
            backbone_name=args.backbone,
            inner_channel=args.inner_channel,
            k=args.k,
        )
        model = load_checkpoint(model, args.pth, args.device)
        export_det(model, args.output, args.opset)

    elif args.task == "rec":
        # 从字符集 JSON 确定 classes_num
        if not os.path.exists(args.char_json):
            raise FileNotFoundError(f"字符集文件不存在: {args.char_json}")
        with open(args.char_json, "r", encoding="utf-8") as f:
            char_map = json.load(f)
        classes_num = len(char_map)
        print(f"[INFO] 字符集大小: {classes_num}")

        print(f"[INFO] 构建 CRNN(backbone={args.backbone}, classes_num={classes_num}, "
              f"rnn_type={args.rnn_type}, hidden_size={args.hidden_size}, "
              f"num_layers={args.num_layers}, bidirectional={args.bidirectional})")
        model = _build_rec_model(
            backbone_name=args.backbone,
            classes_num=classes_num,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
        )
        model = load_checkpoint(model, args.pth, args.device)
        export_rec(model, args.output, args.opset)


if __name__ == "__main__":
    main()
