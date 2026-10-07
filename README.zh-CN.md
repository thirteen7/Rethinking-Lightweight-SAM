# Rethinking Lightweight SAM

[English](README.md) · [项目主页](https://thirteen7.github.io/Rethinking-Lightweight-SAM/) · [交互演示](https://huggingface.co/spaces/thirteen7/Rethinking-Lightweight-SAM) · [模型下载](https://github.com/thirteen7/Rethinking-Lightweight-SAM/releases/tag/v1.0.0)

论文 **Rethinking Lightweight SAM with Prompt-Adaptive Refinement and Efficient Segment Everything Inference** 的提示自适应精化代码与模型。

支持 **TinySAM、MobileSAM、SAM ViT-H**。每个模型只需一份完整 `.pth`，包含官方基座、首点选择、交互反馈、局部残差、框选择、独立框解码器和编辑比较器。

## 1. 安装

Python 3.10+。先安装匹配的 PyTorch/torchvision；训练和数据集评估需要 CUDA，模型加载及交互 API 支持 CPU。

```bash
git clone https://github.com/thirteen7/Rethinking-Lightweight-SAM.git
cd Rethinking-Lightweight-SAM
pip install -r requirements.txt
pip install -e . --no-deps
```

建议复现实验时使用 PyTorch 2.0.0 / torchvision 0.15.1，并固定 CUDA、硬件与依赖版本。不同主机与版本可能产生数值差异。

## 2. 下载一份模型

```bash
python download_models.py --model tinysam
# 其它模型：--model mobilesam / vith；全部：--model all
```

文件保存在 `weights/`。ViT-H 的传输文件自动拼回一份完整 `.pth`，不改变参数精度。模型与每段传输文件的哈希见 [models.json](models.json)。

## 3. 评估

```bash
python evaluate.py --model tinysam --checkpoint weights/tinysam_prompt_adaptive_v1.pth --dataset coco --data-root datasets/coco --output runs/tinysam-coco
```

将 `--dataset` 改为 `lvis` 或 `sa1b` 即可。用 `--gpu 1` 选择显卡。用 `--dry-run` 检查数据覆盖；`--max-images 1` 用于明确标注的小样本检查。

协议：普通 **legacy mIoU**，中心首点，点/框各三轮，分块 64。COCO/LVIS 使用全部有效验证图与全部非 crowd 目标；SA-1B 采用官方读取器的每图最多 64 目标、按原索引播种的 `randperm` 无放回抽样。只保存逐目标数值、汇总及日志。评估使用 GT 生成初始提示、模拟纠错及评分。

结果保存在 `runs/.../summary.json`，检查记录保存在 `verified.json`。不保存预测图片或掩码。

数据布局：

```text
datasets/coco/
  annotations/instances_val2017.json
  annotations/lvis_v1_val.json
  val2017/                  # COCO 验证图
  train2017/                # LVIS 验证也使用部分 COCO train2017 图
datasets/sa1b/
  images/val/*.jpg
  annotations/val/*.json
```

已有 `coco/trainval/` 合并图目录也可直接使用。LVIS 只评估 `lvis_v1_val.json`，它包含来自 COCO train2017 与 val2017 的图像。

## 4. 训练并自动合并

拟合、校准、权重选择全部来自 **SA-1B train**，使用固定的 3500 图与不重叠的 2900/300/300 划分。首点拟合 1500 图；后续解码器/比较器使用 256/64/32 图。外部验证标签不参与训练。

```bash
python prepare_data.py --source datasets/sa1b --output datasets/sa1b_train3500
python download_models.py --model tinysam --base-only
python train.py --model tinysam --base weights/tinysam_official_base.pth --data-root datasets/sa1b_train3500 --output runs/train-tinysam
```

训练完成后自动生成 `runs/train-tinysam/tinysam_prompt_adaptive_v1.pth`。更换模型名即可使用其它骨干。默认训练解码分块为 4；可用 `--batch-size` 调整资源需求，复现时保持协议相同。

添加 `--reuse-first weights/tinysam_prompt_adaptive_v1.pth` 可复用首点头；默认从头训练首点头。`--dry-run` 检查训练输入并打印阶段。

已有完整模块训练目录可单独合并：

```bash
python merge.py --model tinysam --base weights/tinysam_official_base.pth --components runs/train-tinysam --output weights/tinysam_custom.pth
```

## 5. 交互调用

```python
from PIL import Image
import numpy as np
from prompt_adaptive_sam import Predictor

predictor = Predictor("weights/tinysam_prompt_adaptive_v1.pth", device="cuda")
predictor.set_image(np.asarray(Image.open("example.jpg").convert("RGB")))
mask, logits, choice = predictor.predict([[200, 150]], [1])
# 后续点击需传入完整历史，并传入上轮 logits。
mask, logits, choice = predictor.predict([[200, 150], [250, 180]], [1, 1], previous=logits)
# 框使用原图坐标 [x0, y0, x1, y1]。
mask, logits, choice = predictor.predict(box=[100, 80, 350, 300])
```

## Everything 整图分割

无需手动添加点或框，即可自动生成实例掩码。支持 FSD-SAM 的因子化状态、
原生预览和选择性后缀解码，以及 Dense SAM 的完整网格解码。

```bash
python generate.py --checkpoint weights/tinysam_prompt_adaptive_v1.pth --image example.jpg --method fsd --grid 32 --output runs/everything
```

工作台提供 Point、Box、Everything 三种模式。点／框模式仅使用四个带真实标注的精选样例。先运行初始提示，再用左侧按钮逐次加入两个预设前景纠错点；每个阶段并排比较原版与改进输出及真实目标 IoU。使用上一结果／下一结果可以回看已保存的图片、提示和 IoU。展示样例和提示坐标经过筛选，完整数据集平均指标另外列出。

Everything 保留图片上传，固定左侧 SAM ViT-H / Dense、右侧 FSD-SAM / ViT-H，一次运行两条路径并分别显示真实毫秒耗时。流程动画独立于推理计时。

[代码](https://github.com/thirteen7/Rethinking-Lightweight-SAM) · [在线 Demo](https://huggingface.co/spaces/thirteen7/Rethinking-Lightweight-SAM) · [项目网页](https://thirteen7.github.io/Rethinking-Lightweight-SAM/)

本地 PowerShell 运行：

```powershell
pip install -r demo/requirements.txt
$env:SAM_DEMO_DEVICE='cuda'  # 无 CUDA 时使用 cpu
python demo/space/app.py
```

打开 `http://127.0.0.1:7860`。详见 [展示对比协议](docs/showcase.md)、[Everything 协议](docs/everything.md) 和 [部署说明](demo/README.md)。

## 实验结果

以下数值逐项参照所提供的论文 `main.pdf` 表 1、3、5–7（第 11–14 页），交互指标为普通 legacy IoU (%)。初始提示为一个前景中心点或一个 GT 框；+1/+2 指在初始提示上累计增加一次／两次纠错点击。差值直接采用论文中未舍入数值计算的结果。**† 历史参考基线：** SA-11K 的 MobileSAM，以及 LVIS／SA-11K 的 ViT-H 原版数值来自历史图表，不视为本研究新做的配对对照，见论文附录 E。

### 点提示：初始与纠错后

![首点原版与改进 IoU 对比，以及首点、加一次和加两次纠错的提升](site/assets/results/point-comparison.png)

[矢量图 (SVG)](site/assets/results/point-comparison.svg)

左侧柱状图突出初始提示的原版／改进对比；右侧列出首点或初始框、+1 点、+2 点的提升，初始列加框突出。表内数值依次为 **原版 → 改进**；pp 表示百分点，每个差值对应同一提示阶段。

| 数据集 / 骨干 | **首点 IoU (%)** | **首点提升 (pp)** | +1 点 IoU (%) | +1 点提升 (pp) | +2 点 IoU (%) | +2 点提升 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| COCO val2017 / TinySAM | 46.77 → **55.65** | **+8.88** | 59.62 → **64.43** | +4.81 | 63.55 → **69.44** | +5.89 |
| COCO val2017 / MobileSAM | 50.89 → **55.00** | **+4.10** | 59.74 → **64.45** | +4.71 | 62.90 → **69.50** | +6.61 |
| COCO val2017 / ViT-H | 53.57 → **60.83** | **+7.26** | 67.26 → **70.47** | +3.21 | 71.69 → **74.80** | +3.11 |
| LVIS v1 val / TinySAM | 53.65 → **56.26** | **+2.61** | 53.31 → **62.70** | +9.40 | 54.51 → **66.32** | +11.82 |
| LVIS v1 val / MobileSAM | 51.41 → **54.43** | **+3.02** | 52.37 → **60.91** | +8.54 | 54.13 → **65.86** | +11.73 |
| LVIS v1 val / ViT-H † | 60.50 → **62.01** | **+1.51** | 68.10 → **68.55** | +0.45 | 70.70 → **72.62** | +1.92 |
| SA-11K / TinySAM | 66.83 → **70.14** | **+3.31** | 75.88 → **76.27** | +0.39 | 78.59 → **78.73** | +0.14 |
| SA-11K / MobileSAM † | 64.60 → **68.21** | **+3.61** | 73.40 → **74.84** | +1.44 | 76.20 → **79.20** | +3.00 |
| SA-11K / ViT-H † | 76.50 → **78.22** | **+1.72** | 83.40 → **84.32** | +0.92 | 85.10 → **86.52** | +1.42 |

### 框提示：初始与纠错后

![初始框原版与改进 IoU 对比，以及各纠错阶段的提升](site/assets/results/box-comparison.png)

[矢量图 (SVG)](site/assets/results/box-comparison.svg)

左侧柱状图突出初始提示的原版／改进对比；右侧列出首点或初始框、+1 点、+2 点的提升，初始列加框突出。表内数值依次为 **原版 → 改进**；pp 表示百分点，每个差值对应同一提示阶段。

| 数据集 / 骨干 | **初始框 IoU (%)** | **初始框提升 (pp)** | +1 点 IoU (%) | +1 点提升 (pp) | +2 点 IoU (%) | +2 点提升 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| COCO val2017 / TinySAM | 74.99 → **76.52** | **+1.52** | 74.42 → **77.68** | +3.26 | 74.29 → **78.36** | +4.08 |
| COCO val2017 / MobileSAM | 74.45 → **76.03** | **+1.58** | 72.57 → **77.03** | +4.46 | 71.95 → **77.23** | +5.28 |
| COCO val2017 / ViT-H | 77.28 → **78.17** | **+0.89** | 77.76 → **78.56** | +0.80 | 78.00 → **78.70** | +0.70 |
| LVIS v1 val / TinySAM | 73.81 → **75.47** | **+1.66** | 70.37 → **76.57** | +6.20 | 69.29 → **77.13** | +7.84 |
| LVIS v1 val / MobileSAM | 72.81 → **74.45** | **+1.64** | 67.30 → **75.36** | +8.06 | 65.49 → **75.68** | +10.20 |
| LVIS v1 val / ViT-H † | 77.80 → **77.86** | **+0.06** | 78.30 → **78.45** | +0.15 | 78.50 → **78.60** | +0.10 |
| SA-11K / TinySAM | 82.90 → **84.16** | **+1.26** | 83.79 → **84.84** | +1.05 | 84.24 → **85.18** | +0.95 |
| SA-11K / MobileSAM † | 82.00 → **83.50** | **+1.50** | 82.40 → **84.08** | +1.68 | 82.70 → **84.32** | +1.62 |
| SA-11K / ViT-H † | 86.70 → **87.86** | **+1.16** | 86.70 → **88.07** | +1.37 | 87.10 → **88.10** | +1.00 |

### Segment Everything：ViT-H 与 FSD-SAM 时间对比

| 骨干 / 策略 | ms/图 ↓ | 加速比 | AR@300 (%) ↑ | ΔAR (pp) |
|---|---:|---:|---:|---:|
| SAM ViT-H / Dense | 6,067 | 1.00× | 48.322 | — |
| SAM ViT-H / FSD-SAM | 3,043 | 1.99× | 48.237 | -0.085 |

论文表 7：COCO100 开发子集，100 图／709 目标；NVIDIA RTX 5060 Ti，FP32，32 × 32 网格，64 提示／批。每种配置预热 3 次，每图计时 3 次，质量使用第 0 次结果；CUDA 同步计时，包括图像编码和完整掩码生成。排除图片读取、模型加载、编译、预热、写盘及 GT 评估。论文的秒／图乘以 1000 转为毫秒／图。该开发子集平均时间与 Demo 当前图片的实测耗时分别显示。完整结果见 [docs/results.md](docs/results.md)。

## 许可与致谢

许可文件：[LICENSE](LICENSE)、[NOTICE](NOTICE)。

致谢：[SAM](https://github.com/facebookresearch/segment-anything)、[TinySAM](https://github.com/xinghaochen/TinySAM)、[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)。
