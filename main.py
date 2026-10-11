# train_fusion_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os
import sys
from ultralytics import YOLO
from ultralytics.nn import tasks

warnings.filterwarnings('ignore')

# ================= 定义 MFEblock =================
class oneConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes,
                              padding=paddings, dilation=dilations, bias=False)
    def forward(self, x):
        return self.conv(x)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super().__init__(*modules)

class MFEblock(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = atrous_rates
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE3 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE4 = oneConv(in_channels, in_channels, 1, 0, 1)

    def forward(self, x):
        y0 = self.layer1(x)
        y1 = self.layer2(y0 + x)
        y2 = self.layer3(y1 + x)
        y3 = self.layer4(y2 + x)
        y0_w = self.SE1(self.gap(y0))
        y1_w = self.SE2(self.gap(y1))
        y2_w = self.SE3(self.gap(y2))
        y3_w = self.SE4(self.gap(y3))
        w = torch.cat([y0_w, y1_w, y2_w, y3_w], 2)
        w = self.softmax(self.sigmoid(w))
        w0 = w[:, :, 0:1].unsqueeze(2)
        w1 = w[:, :, 1:2].unsqueeze(2)
        w2 = w[:, :, 2:3].unsqueeze(2)
        w3 = w[:, :, 3:4].unsqueeze(2)
        x_att = w0 * y0 + w1 * y1 + w2 * y2 + w3 * y3
        return self.project(x_att + x)

# ================= 定义 MSEFBlock =================
class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(input_channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(input_channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, input_channels),
            nn.Tanh()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MSEFBlock(nn.Module):
    def __init__(self, ch, reduction_ratio=16):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch)
        self.depthwise_conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.se = SEBlock(ch, reduction_ratio)
        self.fuse_conv = nn.Conv2d(ch * 2, ch, kernel_size=1)
    def forward(self, x):
        x_norm = self.bn(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se(x_norm)
        out = self.fuse_conv(torch.cat([x1, x2], dim=1))
        return out + x

# ================= 动态包装函数（不替换，只附加） =================
def attach_modules(model):
    """在原模块后用Sequential包装附加自定义模块，不改变Detect索引"""
    from ultralytics.nn import tasks
    tasks.MFEblock = MFEblock
    tasks.MSEFBlock = MSEFBlock

    layers = model.model.model
    # 打印前35层确认索引（根据实际输出调整，下面索引适用于您打印的列表）
    print("\n模型层索引及类型（用于确认）:")
    for i, layer in enumerate(layers[:35]):
        print(f"{i:3d}: {layer.__class__.__name__}")

    # 根据您上次打印的索引：
    # 16: A2C2f (P3输出) -> 附加 MSEFBlock
    idx = 16
    old = layers[idx]
    out_ch = old.c2 if hasattr(old, 'c2') else old.cv1.conv.out_channels  # 通用获取输出通道
    print(f"包装层{idx} ({old.__class__.__name__}) out_ch={out_ch} -> + MSEFBlock")
    layers[idx] = nn.Sequential(old, MSEFBlock(out_ch, reduction_ratio=16))

    # 19: A2C2f (P4输出) -> 附加 MFEblock
    idx = 19
    old = layers[idx]
    out_ch = old.c2 if hasattr(old, 'c2') else old.cv1.conv.out_channels
    print(f"包装层{idx} ({old.__class__.__name__}) out_ch={out_ch} -> + MFEblock")
    layers[idx] = nn.Sequential(old, MFEblock(out_ch, atrous_rates=[2, 4, 8]))

    # 22: C3k2 (P5输出) -> 附加 MSEFBlock
    idx = 22
    old = layers[idx]
    out_ch = old.c2 if hasattr(old, 'c2') else old.cv1.conv.out_channels
    print(f"包装层{idx} ({old.__class__.__name__}) out_ch={out_ch} -> + MSEFBlock")
    layers[idx] = nn.Sequential(old, MSEFBlock(out_ch, reduction_ratio=16))

    # 注意：Detect 层的输入列表仍然是 [10,16,19,22]，现在这些位置都变成了 Sequential，
    # 但 Sequential 的输出通道与原来模块的输出通道一致（因为自定义模块保持通道数不变），
    # 所以 Detect 不需要修改。
    return model

# ================= 主程序 =================
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # 路径（请修改为您的实际路径）
    model_yaml_path = r"D:\yolov11\yolov12-main\ultralytics\cfg\models\v12\yolov12_P2.yaml"
    pretrained_weights = r"D:\yolov11\yolov12-main\yolov12n.pt"
    data_yaml_path = r"D:\yolov11\yolov12-main\dataset_new_apple\data.yaml"
    project_name = 'runs/V12train'
    experiment_name = 'exp_Fusion_Attached'

    # 超参数
    IMG_SIZE = 544
    BATCH_SIZE = 16
    EPOCHS = 100
    WORKERS = 0            # Windows 必须 0
    USE_AMP = True
    CACHE = 'disk'

    print("="*60)
    print("🚀 融合 MFEblock + MSEFBlock（包装附加法）")
    print(f"   imgsz={IMG_SIZE}, workers={WORKERS}, amp={USE_AMP}")
    print("="*60)

    # 加载原始模型
    model = YOLO(model_yaml_path)
    print("原始模型加载成功")

    # 附加自定义模块
    model = attach_modules(model)

    # 加载预训练权重（自定义模块会随机初始化，但无关紧要）
    if os.path.exists(pretrained_weights):
        model = model.load(pretrained_weights)
        print("✅ 已加载预训练权重（自定义层自动跳过）")
    else:
        print("⚠️ 从头训练")

    # 开始训练
    results = model.train(
        data=data_yaml_path,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        project=project_name,
        name=experiment_name,
        patience=50,
        verbose=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.0005,
        momentum=0.937,
        amp=USE_AMP,
        workers=WORKERS,
        cache=CACHE,
    )

    # 验证
    best_path = os.path.join(project_name, experiment_name, 'weights', 'best.pt')
    if os.path.exists(best_path):
        best_model = YOLO(best_path)
        metrics = best_model.val(
            data=data_yaml_path, split='val', imgsz=IMG_SIZE,
            batch=BATCH_SIZE, plots=True, verbose=True, workers=0
        )
        print("\n" + "*"*50)
        print("🏆 融合模型评估成绩单 🏆")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50:    {metrics.box.map50:.4f}")
        print(f"精确率:   {metrics.box.mp:.4f}")
        print(f"召回率:   {metrics.box.mr:.4f}")
        print("*"*50)
    else:
        print("❌ 未找到 best.pt")