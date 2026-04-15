# Grokking Baseline 说明

这份文档解释的是当前仓库里的 `grokking_baseline.py`，重点是：

- 代码现在到底实现了什么
- 为什么我最近又调整了一次默认结构
- 哪些设置是为了“更可能 grok”，哪些设置是为了“更接近论文里的圆环表示”

## 1. 当前实验在做什么

任务是学习模运算：

```text
(a, b) -> c
其中 c = a op b mod p
```

默认任务是：

```text
(a + b) mod 53
```

为什么默认还是加法：

- 加法对应循环群 `Z_p`
- 如果模型真的学到群结构，token embedding 更容易在 PCA 里排成圆环
- 这就是 Paper 1 Figure 1 最典型的现象

## 2. 数据是怎么构造的

当前实现直接枚举所有 `(a, b)` 组合：

```python
def make_dataset(prime: int, operation: str, seed: int):
    op_fn = {
        "add": lambda a, b: (a + b) % prime,
        "sub": lambda a, b: (a - b) % prime,
        "mul": lambda a, b: (a * b) % prime,
        "div": lambda a, b: (a * mod_inverse(b, prime)) % prime,
    }[operation]

    samples = []
    b_values = range(1, prime) if operation == "div" else range(prime)
    for a in range(prime):
        for b in b_values:
            y = op_fn(a, b)
            samples.append((a, b, y))
```

默认 `p=53` 时，加法总样本数是：

```text
53 * 53 = 2809
```

默认训练比例是 `0.3`，所以训练集只有 30%。这很重要，因为 grokking 往往发生在“小训练集 + 长时间训练”的组合上。

## 3. 当前模型结构

### 3.1 输入

当前实现不是 3-token 输入，也没有单独的 operation token。

真实输入是：

```python
x = torch.tensor([[a, b] for a, b, _ in samples], dtype=torch.long)
```

也就是说，每个样本只输入两个 token：

```text
[a, b]
```

### 3.2 Transformer 主体

```python
class GrokkingTransformer(nn.Module):
    def __init__(self, prime, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(prime, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(2, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model * 2)
        self.head = nn.Linear(d_model * 2, prime, bias=False)
```

这里有几个关键点：

- 只有两个位置，所以 `pos_embed` 的形状是 `[2, d_model]`
- Transformer 仍然是 causal-mask 的 decoder-only 风格
- 最后一层两个位置的输出会被拼接
- 拼接后会做一次 `LayerNorm`
- 最后再接 `Linear(2*d_model -> p)`

### 3.3 前向传播

```python
def forward(self, x):
    h = self.token_embed(x) + self.pos_embed.unsqueeze(0)
    h = self.encoder(h, mask=self.causal_mask)
    h = torch.cat([h[:, 0, :], h[:, 1, :]], dim=-1)
    h = self.norm(h)
    return self.head(h)
```

## 4. 为什么现在保留拼接后 LayerNorm

我上一轮把拼接后的 `LayerNorm` 去掉了，结果新的 `paper1_add` 产出更差：

- 训练集很快到 1.0
- 测试准确率几乎一直只有 1% 左右
- `eff_dim` 只从 47 左右降到 41 左右

这说明模型在记忆，但根本没有进入 grokking。

所以当前版本把这层加了回来。原因不是“论文一定明确要求它”，而是：

- 这份 baseline 的实际训练动力学更依赖它
- 去掉之后，默认配置会直接退化成纯记忆
- 保留它能让我们回到“先稳定 grok，再继续逼近圆环结构”的状态

换句话说，这里是一个经验性回退，不是理论上宣称“论文必须有这层”。

## 5. 为什么优化器要拆成两组

这是这次保留下来的重要修改。

当前实现用的是：

```python
def build_optimizer(model, decoder_lr, embed_lr, decoder_weight_decay):
    embed_params = list(model.token_embed.parameters()) + [model.pos_embed]
    decoder_params = list(model.encoder.parameters()) + list(model.norm.parameters()) + list(model.head.parameters())
    return torch.optim.AdamW([
        {"params": embed_params, "lr": embed_lr, "weight_decay": 0.0},
        {"params": decoder_params, "lr": decoder_lr, "weight_decay": decoder_weight_decay},
    ])
```

含义是：

- embedding 单独一组
- transformer block + 读出 `LayerNorm` + 输出头归到 decoder 组
- embedding 不做 weight decay
- decoder 做 weight decay
- 两组可以分开设学习率

这点仍然很有价值，因为旧 baseline 最大的问题之一，就是 embedding 和 decoder 共用一个学习率。

## 6. 学习率和调度

当前调度器是：

```python
def build_scheduler(optimizer, warmup_steps, lr_schedule, lr_min_ratio, max_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        if lr_schedule == "cosine":
            progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine_val
        return 1.0
```

默认是：

- `warmup_steps = 10`
- `lr_schedule = "constant"`

也就是说：

- 前 10 步线性升温
- 之后保持常数学习率

`cosine` 仍然保留，因为它有时更稳，但默认值继续是 `constant`。

## 7. 为什么默认梯度裁剪又开回 1.0

当前默认：

```text
--grad-clip 1.0
```

原因很直接：

- 我上一轮把它关掉之后，新的默认 run 根本没有 grok
- 这份 baseline 实际上更依赖轻量的梯度裁剪来稳定训练
- 对我们现在的目标来说，“先稳定泛化”比“纯粹追求 paper-like setting”更重要

如果你想测试更激进的版本，仍然可以手动：

```bash
python grokking_baseline.py --grad-clip 0
```

但它不再是默认值。

## 8. PCA 画的到底是什么

PCA 可视化画的是输入 token embedding：

```python
def get_token_embeddings(model, prime):
    emb = model.token_embed.weight[:prime].cpu().numpy()
    return emb
```

不是：

- 不是输出头权重
- 不是 transformer hidden state
- 不是拼接后的 2-token 表示

所以如果 PCA 没有圆环，最直接的解释仍然是：

- embedding 几何还没有长成论文里的循环结构

而不是“PCA 代码画错了”。

## 9. 为什么要同时保存普通 PCA 和归一化 PCA

PCA 代码支持：

```python
compute_pca_2d(embeddings, normalize=False)
```

当 `normalize=True` 时，会先对每个 embedding 做 L2 归一化。

这么做的意义是：

- 有时 embedding 的方向已经有结构
- 但范数还在变化，普通 PCA 会把结构淹没掉
- 归一化后更容易看见“角度结构”，比如 Fourier-like 的圆环

所以现在会同时输出：

- `pca_evolution.png`
- `pca_evolution_normed.png`
- 以及它们各自的 `full` 版本

## 10. 有效维度为什么重要

有效维度定义为：

```text
e^S
其中 S 是 PCA explained variance ratio 的熵
```

直觉上：

- 如果 embedding 分散在很多维度里，`e^S` 会很大
- 如果 embedding 主要落在很少几个方向上，`e^S` 会下降

所以它是一个很有用的配套指标：

- 如果 `test_acc` 上去了，但 `e^S` 几乎不降，说明模型可能只是用了另一种高维策略
- 如果 `test_acc` 上去，同时 `e^S` 明显下降，说明表示开始变得更有规律

## 11. 这次修正后，当前 baseline 的立场是什么

当前版本不是“最纯论文版”，也不是“最稳工程版”，而是一个折中：

- 保留对圆环结构有帮助的设置：分开 `embed_lr` 和 `lr`
- 保留 empirically 更稳的设置：拼接后 `LayerNorm`、`grad_clip=1.0`
- 保持默认 `constant` LR，避免默认就把训练改成另一种 regime

也就是说，当前 baseline 的策略是：

```text
先保证它能稳定 grok
再在这个基础上继续逼近 Figure 1 的几何结构
```

## 12. 你跑实验时该怎么判断

### 情况 1：train 很快到 1.0，test 也最终上去

这是好现象。接下来重点看：

- `effective_dim.png`
- `pca_evolution_normed.png`

### 情况 2：train 很快到 1.0，但 test 长时间几乎不涨

这是纯记忆。优先怀疑：

- 结构被改坏了
- weight decay 不够
- 梯度裁剪被关掉后训练动力学崩了

### 情况 3：test 上去了，但 PCA 仍不成圆

这说明“会泛化”了，但“不是通过 Figure 1 那种低维循环表示在泛化”。

这时就该继续扫：

- `--lr`
- `--weight-decay`
- `--embed-lr`

而不是继续动 PCA 代码本身。

## 13. 当前建议的实验顺序

### 先跑默认设置

```bash
python grokking_baseline.py --outdir runs/paper1_add
```

### 再只改一项

```bash
python grokking_baseline.py --lr 3e-4 --outdir runs/lr_3e4
python grokking_baseline.py --weight-decay 2.0 --outdir runs/wd_2
python grokking_baseline.py --embed-lr 3e-4 --outdir runs/embedlr_3e4
```

这样最容易分辨：到底是 decoder 学习率、decoder 正则，还是 embedding 学习率在影响几何结构。

## 14. 实验复盘

下面这一节记录的是 2026-03-25 这几次连续实验的现象和结论。它们非常有价值，因为它们不只是告诉我们“哪个配置更好”，还告诉我们“哪些直觉其实是错的”。

### 14.1 历史参考：`runs/add_p53`

这是最早的一版历史 run，可以作为参考，但不要把它和当前 baseline 完全等价看待，因为当时代码里还没有 `embed_lr` 这个参数分组。

观察到的现象：

- `test_acc` 首次超过 `0.99` 的时间大约是 `19800` 步
- `eff_dim` 最低到过大约 `3.54`
- 最终 `eff_dim` 大约是 `5.32`

它说明了一件事：

- 这类任务完全可以 grok，而且 embedding 确实能明显低维化

但它也提醒我们：

- “能 grok” 不等于“圆环一定很完美”
- 不同代码版本之间，单纯比一列数字并不总是公平

### 14.2 过度回退失败：`runs/paper1_add`

这一版我把结构往“更纯论文”方向推得太猛了：

- 去掉了拼接后的 `LayerNorm`
- 默认关闭了梯度裁剪

结果非常差：

- 训练集很快记住，`train_acc` 接近 `1.0`
- 测试集几乎一直只有 `1%` 左右
- 最终 `eff_dim` 还在 `41` 左右

这次实验的结论非常明确：

- 在这份代码里，直接去掉拼接后 `LayerNorm` 并关闭默认 `grad_clip`，会把训练动力学搞坏
- 问题不是“圆环不够完美”，而是模型压根没有进入 grokking

所以它是一个典型的反例：

```text
更 paper-like 的结构改动
不一定会带来更好的 grokking 表示
```

### 14.3 恢复稳定后的基线：`runs/paper1_add_retry`

这版把更稳的部件加回来了：

- 恢复拼接后 `LayerNorm`
- 恢复默认 `grad_clip=1.0`
- 保留 `embed_lr` / `lr` 分组

结果：

- `test_acc` 首次超过 `0.99` 大约在 `23800` 步
- 最终测试准确率回到 `1.0`
- `eff_dim` 最低大约是 `9.44`
- 最终 `eff_dim` 大约是 `11.17`

图像上表现为：

- 已经能看到明显圆环
- 但圆环仍然偏厚、偏椭圆，点距也不够均匀

这版的结论是：

- 代码已经回到“能稳定 grok”的轨道
- 下一步应该调超参，而不是继续大改结构

### 14.4 提高 weight decay：`runs/paper1_add_wd2`

这一版只改了一项：

```text
weight_decay: 1.0 -> 2.0
```

结果有点反直觉：

- 它更早 grok，大约 `17000` 步就到 `test_acc > 0.99`
- `eff_dim` 最低也更低，达到过大约 `8.66`
- 但最终 `eff_dim` 反而回到 `11.52`
- PCA 图的圆环没有变得更漂亮，反而整体更僵、更不稳定

这次实验告诉我们：

- 更大的 decoder weight decay 并不是“圆环更干净”的单调按钮
- 它可能会更早逼出泛化，但不一定得到更好的几何结构

所以对于当前 baseline：

```text
更早 grok
不等于
更漂亮的圆环
```

### 14.5 降低 decoder 学习率：`runs/paper1_add_lr5e4`

这一版只改了一项：

```text
lr: 1e-3 -> 5e-4
embed_lr 仍然保持 1e-3
```

结果非常值得关注：

- 它 grok 得更晚，大约 `75500` 步才到 `test_acc > 0.99`
- 但最终测试准确率仍然是 `1.0`
- `eff_dim` 最低大约到 `7.28`
- 最终 `eff_dim` 大约是 `7.86`
- 目前为止，这一版的归一化 PCA 圆环是最接近“干净圆环”的

这说明：

- 当前实现里，decoder learning rate 比单纯继续加大 `weight_decay` 更可能改善几何结构
- 较小的 decoder lr 虽然会明显推迟 grokking 的发生时间，但会让后期表示收缩得更平滑、更均匀

### 14.6 从这些实验里学到的真正规律

把这几次实验放在一起看，得到的经验不是一条，而是几条。

#### 规律 1：先保证“能 grok”，再谈“圆环有多完美”

`paper1_add` 证明了一件事：

- 如果结构性修改过猛，模型会直接退化成记忆
- 这时讨论 PCA 圆不圆已经没有意义

#### 规律 2：当前实现里，`lr` 比 `weight_decay` 更像“圆环质量旋钮”

从 `paper1_add_retry`、`paper1_add_wd2`、`paper1_add_lr5e4` 三个 run 看：

- `weight_decay=2.0` 更早 grok，但几何没有更漂亮
- `lr=5e-4` grok 更晚，但圆环更干净

所以对这版 baseline 来说，下一阶段更该优先扫：

- `lr`

而不是继续一味增大：

- `weight_decay`

#### 规律 3：不要只盯着“首次 grok 的步数”

如果只看 grok_step：

- `wd2` 似乎最好
- `lr5e4` 似乎最差

但如果看后期的 `eff_dim` 和 PCA 图：

- `lr5e4` 反而最好

所以真正该看的不是单一指标，而是：

- 是否 grok
- 后期 `eff_dim` 最低能降到多少
- 末期归一化 PCA 的圆环厚不厚、均不均匀

#### 规律 4：最佳圆环不一定出现在最后一步

在 `paper1_add_retry` 里，`eff_dim` 最低点并不是最后一步。

这提示我们：

- 训练末尾未必就是几何结构最漂亮的时刻
- 后续如果要把 baseline 做得更完整，可以考虑额外保存“`eff_dim` 最低时刻”的 PCA 快照

### 14.7 当前阶段的工作假设

基于现有结果，我现在对这份 baseline 的工作假设是：

```text
结构已经够用了
主要矛盾在超参数
而且最值得继续扫的是 decoder lr
```

更具体一点：

- `embed_lr=1e-3` 目前先保持不动
- `weight_decay` 不要再盲目往上加
- `lr` 往 `5e-4` 甚至更低扫，可能更有希望得到更圆、更薄的环

这就是为什么这些实验记录值得写进解释文件：它们把“应该往哪边调”这件事，从猜测变成了有证据支持的判断。
