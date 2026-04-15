# Grokking Baseline 复现

这个目录里的 `grokking_baseline.py` 用来复现两篇 grokking 论文里最常见的 modular arithmetic 实验，并自动生成几类可视化：

- Paper 1: *Towards Understanding Grokking: An Effective Theory of Representation Learning*
- Paper 2: *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*

## 环境配置

下面这套配置以 Windows + NVIDIA GPU 为主，目标是：

- Python `3.11`
- 独立虚拟环境 `.venv`
- 使用 GPU 版 PyTorch

### 1. 安装 Python 3.11

推荐直接从 Python 官网安装 `Python 3.11.x`，安装时勾选：

- `Add python.exe to PATH`

安装完成后，在 PowerShell 里确认：

```powershell
py -3.11 --version
```

### 2. 创建虚拟环境

在项目根目录运行：

```powershell
py -3.11 -m venv .venv
```

激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止脚本执行，可以先临时放开当前窗口的执行策略：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

升级基础工具：

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 3. 安装 GPU 版 PyTorch

先确认机器能看到 NVIDIA 驱动：

```powershell
nvidia-smi
```

如果这条命令都失败，先不要装 PyTorch，先修好显卡驱动。

然后安装 PyTorch。按照 PyTorch 官方安装页，当前 Windows + pip 的稳定版支持多种 CUDA 轮子；如果你不确定选哪个，先从兼容性通常更好的 `cu118` 开始：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果你已经确认自己的驱动支持更新的 CUDA 轮子，也可以把上面的 `cu118` 替换成官方安装页里提供的其他版本。

### 4. 安装项目依赖

```powershell
pip install numpy matplotlib scikit-learn
```

### 5. 验证 PyTorch 是否真的用了 GPU

```powershell
python -c "import torch; print('torch =', torch.__version__); print('cuda available =', torch.cuda.is_available()); print('device count =', torch.cuda.device_count()); print('device 0 =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

如果输出里：

- `cuda available = True`

就说明 PyTorch 已经能用 GPU 了。

### 6. 跑一个最小实验

```powershell
python grokking_baseline.py --device cuda --outdir runs/smoke_test
```

如果你想强制用 CPU，可以改成：

```powershell
python grokking_baseline.py --device cpu --outdir runs/cpu_test
```

### 7. 常见问题

#### `py -3.11` 找不到

说明 Python 3.11 没装好，或者没加到 PATH。回到 Python 安装器重新安装最省事。

#### `nvidia-smi` 找不到

说明 NVIDIA 驱动没装好，或者当前机器没有可用的 NVIDIA GPU。

#### `torch.cuda.is_available()` 是 `False`

优先检查这几项：

- 当前装的是不是 GPU 版 PyTorch，而不是 CPU 版
- NVIDIA 驱动是否正常
- 你有没有在激活的 `.venv` 里安装依赖
- 机器是否真的有 NVIDIA GPU

## 当前版本的默认目标

这份 baseline 现在默认优先追求“先能稳定 grok，再继续逼近 Figure 1 的圆环结构”，而不是一次性把所有设置都改成最激进的 paper-like 版本。

默认配置是：

- 任务：`a + b mod 53`
- 数据划分：`30%` 训练，`70%` 测试
- 模型：2-token decoder-only transformer，`d_model=256`，`n_heads=4`，`n_layers=2`，`d_ff=1024`
- 读出方式：最后一层两个位置的输出拼接后，先做 `LayerNorm`，再接 `Linear(2*d_model -> p)`
- 优化器：AdamW，embedding 和 decoder 分组
- embedding 组：`embed_lr=1e-3`，`weight_decay=0`
- decoder 组：`lr=1e-3`，`weight_decay=1.0`
- 学习率：前 `10` 步 warmup，之后默认 `constant`
- 训练方式：full-batch，`dropout=0.0`
- 梯度裁剪：默认 `grad_clip=1.0`

这套默认值不是最“纯”的论文设置，但它比我上一个版本更稳。上一个版本去掉拼接后 `LayerNorm`、关闭默认梯度裁剪后，训练集能记住，但测试集几乎起不来，说明那样改过头了。

## 运行

### 1. 默认加法实验

```bash
python grokking_baseline.py --outdir runs/paper1_add
```

### 2. 论文里的除法实验

```bash
python grokking_baseline.py --operation div --prime 97 --outdir runs/div_p97
```

### 3. 相图扫描

相图模式会扫描 decoder learning rate 和 decoder weight decay，`embed_lr` 固定为 `1e-3`。

Windows PowerShell:

```powershell
python grokking_baseline.py `
  --phase-diagram `
  --phase-grid-size 8 `
  --phase-max-steps 20000 `
  --outdir runs/phase
```

macOS / Linux:

```bash
python grokking_baseline.py \
  --phase-diagram \
  --phase-grid-size 8 \
  --phase-max-steps 20000 \
  --outdir runs/phase
```

## 产出文件

普通训练会生成：

- `config.json`：完整超参数
- `metrics.csv`：按步记录 train/test loss、acc、effective dimension
- `model.pt`：模型权重
- `curve.png`：训练曲线
- `operation_table.png`：运算表热力图
- `pca_evolution.png`：PCA 演化关键帧
- `pca_evolution_full.png`：所有 PCA 快照
- `pca_evolution_normed.png`：L2 归一化后再做 PCA 的关键帧
- `pca_evolution_normed_full.png`：归一化版全部快照
- `effective_dim.png`：有效维度曲线
- `output_weights_tsne.png`：输出层权重的 t-SNE / PCA 图

相图模式额外生成：

- `phase_diagram.png`

## 关键可视化怎么看

### PCA embedding evolution

- 画的是 `token_embed.weight[0:p]`
- 普通版直接做 PCA
- `normed` 版会先把每个 embedding 做 L2 归一化，再 PCA
- modular addition 真正学到循环结构后，token 往往会沿着数值顺序排成近似圆环

默认快照策略：

- `step 0` 一定保存
- 每隔 `--pca-interval`（默认 `2000`）自动保存
- `--embed-snapshot-steps` 可额外指定关键步数

### Effective dimension

这是 Paper 1 Figure 7 风格的指标。`e^S` 越低，表示 embedding 更集中在少数几个主成分上。出现圆环时，通常会显著下降。

### Output weight t-SNE

这里看的不是 embedding，而是输出头 `head.weight` 的每一行。当前头的形状是：

```text
[prime, 2 * d_model]
```

因为它接收的是两个 token 位置输出拼接后的表示。

## 重要参数

### 训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--prime` | `53` | 模运算素数 |
| `--operation` | `add` | `add/sub/mul/div` |
| `--train-fraction` | `0.3` | 训练集比例 |
| `--batch-size` | `0` | `0` 表示 full-batch |
| `--max-steps` | `100000` | 最大训练步数 |
| `--eval-interval` | `100` | 评估间隔 |
| `--lr` | `1e-3` | decoder 侧学习率：transformer block + 输出头 |
| `--embed-lr` | `1e-3` | embedding 学习率 |
| `--weight-decay` | `1.0` | 只施加在 decoder 侧 |
| `--warmup-steps` | `10` | 线性 warmup 步数 |
| `--lr-schedule` | `constant` | `constant` 或 `cosine` |
| `--lr-min-ratio` | `0.05` | cosine 的最低 LR 比例 |
| `--grad-clip` | `1.0` | 梯度裁剪阈值，`0` 为关闭 |
| `--dropout` | `0.0` | dropout 率 |
| `--seed` | `0` | 随机种子 |

### PCA / 可视化参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--embed-snapshot-steps` | `""` | 额外手动指定 PCA 快照 |
| `--pca-interval` | `2000` | 自动快照间隔，`0` 为关闭 |
| `--stop-test-acc` | `0.99` | 达到该测试准确率后触发“可提前停止”逻辑 |
| `--stop-extra-steps` | `0` | 只有大于 0 才会真的提前停；默认仍跑满 `max_steps` |

### 相图参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--phase-diagram` | `false` | 启用相图模式 |
| `--phase-grid-size` | `8` | 网格大小 |
| `--phase-max-steps` | `20000` | 每个网格点的训练步数 |

## 推荐的实验顺序

### 先验证默认设置

```bash
python grokking_baseline.py --outdir runs/paper1_add
```

### 再扫 decoder 侧超参

```bash
python grokking_baseline.py --lr 3e-4 --outdir runs/add_lr_3e4
python grokking_baseline.py --weight-decay 2.0 --outdir runs/add_wd_2
```

### 如果想测试 embedding 学习率敏感性

```bash
python grokking_baseline.py --embed-lr 3e-4 --outdir runs/add_embedlr_3e4
python grokking_baseline.py --embed-lr 3e-3 --outdir runs/add_embedlr_3e3
```

## 常见问题

### 1. 训练已经 grok，但 PCA 没有圆环

优先排查这几项：

- 任务不是 `add`
- 训练步数不够
- `embed_lr` 和 `lr` 没分开调，或者 embedding 学习率偏离太多
- dropout 太大，或者改成了 `cosine` LR
- 快照没覆盖到关键阶段

建议从默认值开始，只改一项：

```bash
python grokking_baseline.py --outdir runs/baseline_default
python grokking_baseline.py --lr 3e-4 --outdir runs/decoder_lr_3e4
python grokking_baseline.py --weight-decay 2.0 --outdir runs/wd_2
```

### 2. 训练集很快到 1.0，但测试集几乎不涨

这说明模型只是在记忆，没有进入 grokking。优先排查：

- 你是不是改掉了默认的 `LayerNorm` 读出
- 你是不是把 `--grad-clip` 关掉了
- decoder weight decay 是否太小
- seed 是否过于不稳定

### 3. 曲线震荡很厉害

可以尝试：

```bash
python grokking_baseline.py --lr 3e-4
python grokking_baseline.py --warmup-steps 50
python grokking_baseline.py --grad-clip 1.0
```

这些通常会让训练更稳，但不一定更容易出现 Figure 1 那种干净圆环。

### 4. 相图几乎全是 Confusion

- 提高 `--phase-max-steps`
- 缩小扫参范围
- 确认当前任务是否太难

### 5. 相图几乎全是 Comprehension

- `train_fraction` 可能太大
- decoder 可能太弱正则、太容易直接学会

## 备注

这份 README 描述的是当前仓库里的实现，不再沿用旧版文档中“3-token 输入 + 最后一位读出”的说法。当前实现是：

- 输入只有两个 token：`[a, b]`
- 使用 causal mask 的 2-token decoder-only transformer
- 读出时拼接两个位置的输出
- 拼接后先做 `LayerNorm`
- 再接线性分类头
