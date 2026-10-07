# Rethinking Lightweight SAM

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
mask, logits, choice = predictor.predict([[200, 150], [250, 180]], [1, 0], previous=logits)
# 框使用原图坐标 [x0, y0, x1, y1]。
mask, logits, choice = predictor.predict(box=[100, 80, 350, 300])
```

## 实验结果

下表为发布权重第三轮普通 mIoU (%)。完整三轮结果和覆盖见 [docs/results.md](docs/results.md)。

| 数据集 | TinySAM 点 | TinySAM 框 | MobileSAM 点 | MobileSAM 框 |
|---|---:|---:|---:|---:|
| SA-1B official cap64 | 78.729 | 85.183 | 79.202 | 84.317 |
| COCO val2017 全目标 | 69.438 | 78.364 | 69.502 | 77.228 |
| LVIS v1 val 全目标 | 66.323 | 77.133 | 65.862 | 75.681 |

## 许可与致谢

许可文件：[LICENSE](LICENSE)、[NOTICE](NOTICE)。

致谢：[SAM](https://github.com/facebookresearch/segment-anything)、[TinySAM](https://github.com/xinghaochen/TinySAM)、[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)。
