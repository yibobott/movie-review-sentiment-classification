# LSTM Language Model Pretraining — 实现设计文档

> 目标：通过在 unlabeled 数据上预训练 LSTM 语言模型，将权重转移到分类器 BiLSTM，把 Kaggle acc 从当前 ~92% 推过 **94%**。

---

## 1. 总览

### 1.1 现状
- **当前最优**：val 92.35（raw, run 20260510_154607）
- **现状瓶颈**：unlabeled 数据只通过 Word2Vec（co-occurrence within window=8）被利用，其长程依赖、否定结构、篇章信号未被挖掘
- **结构改进 D（多头 attn pool）已加入**，纯架构调参的天花板预计 ≤93.5%

### 1.2 目标
- **主要 KPI**：val acc ≥ 94.0%，对应 Kaggle ≥ 93.5%
- **次要 KPI**：LM perplexity ≤ 80（IMDB 类语料的合理目标）
- **过程 KPI**：从 LM 加载到分类器后，**第 1 个 epoch 的 val acc** 应 ≥ 当前同等 epoch baseline + 1%（确认权重转移有效）

### 1.3 合规性
- ✅ 在**作业提供的 unlabeled 数据**上自训 LSTM，不是外部 pretrained 模型
- ✅ "LSTM-like model" — LM 就是 LSTM
- ✅ 推理时单模型单 forward
- ✅ 规则明确鼓励利用 unlabeled data

---

## 2. 架构决策

### 2.1 LM 方向性：单向（autoregressive）

| 选项 | 优点 | 缺点 | 决定 |
|---|---|---|---|
| **A. 单向前向 LM** | 标准 next-token 预测，loss 干净，1 次训练 | 只能初始化 BiLSTM 前向参数 | ✅ **首选** |
| B. 双向 LM（前向 + 后向 LM 各训一次） | BiLSTM 两个方向都有预训练 | 2× 训练成本，存储两套权重 | 备选（v2 升级） |
| C. Masked LM（BERT 风格） | 真双向、单次训练 | 实现复杂，masking 策略调参敏感，偏离 LSTM 范式 | ❌ |

**结论**：v1 用 **单向前向 LM**。后向 LM 留作 v2 升级。

### 2.2 LM 网络配置

```
LSTMLanguageModel:
  vocab_size: 74058 (与现 vocab 一致)
  embed_dim:  300   (与现 w2v 一致)
  hidden_dim: 192   (与分类器一致 → 关键，便于权重转移)
  num_layers: 2     (与分类器一致)
  bidirectional: False
  dropout: 0.4 (intra-LSTM)
  embed_dropout: 0.3
  weight_tying: True (embedding ↔ output projection 共享 → 减参数 + 正则)
```

**注意**：embedding 投影回 vocab 时，因为 embed_dim=300 ≠ hidden_dim=192，**不能直接 weight tie**。两种方案：

| 方案 | 描述 |
|---|---|
| **a. 加 adapter**：`hidden(192) → linear(192,300) → emb_T(300,vocab)` | 实现 weight tying，减 22M 参数（vocab 74058 × 300 = 22M） |
| **b. 不 tie**：直接 `Linear(192, vocab)` | 参数多 14M，但实现简单 |

**决定**：方案 a（adapter + tying）。adapter 是 1 个小 linear，几乎免费；省下的 22M 显存能开更大 batch。

### 2.3 权重转移策略

分类器 BiLSTM 与 LM 的形状关系：

| 参数 | LM (单向) | BiLSTM (双向) | 是否能转移 |
|---|---|---|---|
| `embedding.weight` | (V, 300) | (V, 300) | ✅ 完全转移 |
| `lstm.weight_ih_l0` | (4·192, 300) | (4·192, 300) | ✅ 转移到前向 |
| `lstm.weight_hh_l0` | (4·192, 192) | (4·192, 192) | ✅ 转移到前向 |
| `lstm.bias_ih/hh_l0` | (4·192,) | (4·192,) | ✅ 转移到前向 |
| `lstm.weight_ih_l1` | **(4·192, 192)** | **(4·192, 384)** | ⚠️ **维度失配**（BiLSTM 第 1 层输入是前后向 concat = 2H） |
| `lstm.weight_hh_l1` | (4·192, 192) | (4·192, 192) | ✅ 转移到前向 |

**Layer 1 weight_ih 失配的处理**（关键）：

```python
# 形状: BiLSTM weight_ih_l1 = (768, 384) = (4·192, 2·192)
# LM   weight_ih_l1         = (768, 192)
#
# 把 LM 权重放到 BiLSTM 的 [:, :192]（对应"接收前向 layer-0 输出"），
# [:, 192:]（接收后向 layer-0 输出）置零。
# 训练初期，layer-1 前向方向忽略 layer-0 后向流，与 LM 行为一致；
# 后续训练逐渐填入第二半部分。

bilstm.weight_ih_l1.data[:, :192] = lm.weight_ih_l1.data
bilstm.weight_ih_l1.data[:, 192:] = 0.0
```

**后向方向**（`*_l0_reverse`, `*_l1_reverse`）：v1 不转移，保持随机初始化。

### 2.4 转移收益拆解

| 组件 | 转移后预期收益 |
|---|---|
| embedding（替换 w2v） | +0.3 ~ +0.6% |
| BiLSTM 前向 layer 0 | +0.3 ~ +0.5% |
| BiLSTM 前向 layer 1 (含 layer1 ih 零填充技巧) | +0.1 ~ +0.3% |
| **合计（v1 单向 LM）** | **+0.7 ~ +1.4%** |
| 双向 LM 升级（v2） | 额外 +0.2 ~ +0.5% |

---

## 3. 文件结构

```
movie-review-sentiment-classification/
├── data/
│   ├── lm_dataset.py           [新增] LMDataset (BPTT 切片)
│   └── ...
├── models/
│   ├── lstm.py                  [既有]
│   └── lstm_lm.py              [新增] LSTMLanguageModel
├── engine/
│   ├── lm_trainer.py           [新增] LM 训练循环
│   ├── trainer.py               [改] train() 函数加 optimizer 参数（允许外部传入 discriminative LR optimizer）
│   └── ...
├── utils/
│   ├── weight_transfer.py      [新增] LM → 分类器权重映射
│   ├── vocab_io.py             [新增] 保存/加载 idx2word.json + vocab_hash
│   ├── config.py                [改] 加 LMConfig + FineTuneConfig
│   └── ...
├── pretrain_lm.py              [新增] LM 预训练入口（独立 CLI）
├── train.py                     [改] 支持 lm_ckpt 加载与 discriminative LR
└── config.yaml                  [改] 新增 lm: 段
```

---

## 4. 数据管道（`data/lm_dataset.py`）

### 4.1 语料构建
- **训练源**：`labeled (25k) + unlabeled (50k) + test (25k)` = **100k** 文本
- 复用现有 tokenize / Vocab
- **不**做 head+tail 截断（LM 不需要）；改为 BPTT 滚动切片

### 4.2 EOS Token：单独占一个 ID（不复用 PAD）

**坑点**：原方案曾考虑「用 `<eos>=PAD` 表示文档结束」，但这是错的：
- `nn.Embedding(padding_idx=0)` 会**强制 PAD 的 embedding 为零向量、不更新梯度**
- LM 训练 loss 用 `ignore_index=PAD` → 模型在「该预测 EOS」的位置**完全无 loss 信号**
- 模型实际学不到「文档结束」这个信号；PAD embedding 是零向量，作为输入语义混乱

**正确做法**：
- 在 LM 阶段扩展 vocab：`EOS_IDX = len(classifier_vocab)`，即 LM vocab 比 classifier vocab 多 1 行
- LM `nn.Embedding` shape = (V_cls + 1, embed_dim)
- LM 训练时 `ignore_index=PAD`，**不** ignore EOS
- 权重转移时只 copy `embedding.weight[:V_cls]`，EOS 那一行随之丢弃（分类器永远看不到 EOS）
- 输出投影层 `proj` 同理：转移给分类器无意义，所以分类器根本不需要 proj，丢弃即可

### 4.3 文档级 train/val Split（必须先 split 再 flatten）

**坑点**：如果先把所有文本拼成长序列再 split train/val，BPTT chunk 边界处会**跨文档**，造成 train/val 上下文泄漏：
- 边界附近的 chunk 可能同时含 train 和 val 的 token
- 同一篇评论被切到两侧时，train 和 val 共享语义上下文
- 经典 temporal data leakage → val PPL 虚低，转移到分类器后真实收益打折

**正确流程**（采用「完全隔离 labeled」的简化方案）：
```
1. 从 unlabeled 50k 中按 5% 抽 LM val docs（≈ 2500 docs）
   ⚠️ 不要从 labeled 25k 抽 LM val！避免与分类器 val 重叠
2. LM train docs = remaining unlabeled (~47.5k) + test (25k) = 72.5k docs
   ⚠️ 完全不用 labeled 25k！避免分类器 val 文本被 LM 预训练看过 (代价仅 0.1-0.2% acc，但换来数字可与旧 baseline 直接比较)
3. 各自分别 tokenize 每篇文档：[tok1, tok2, ..., tokN, EOS]
4. 各自 flatten 成长序列 (LM_train_seq, LM_val_seq)
5. 各自做 BPTT chunk：x[i] = seq[i:i+B], y[i] = seq[i+1:i+B+1]
6. 任何 labeled 文本都不进 LM train/val→分类器 val acc 与历史运行可公平比较
```

**为什么不采用「包含 classifier_train_labeled」的最大化方案**：
- 那需要 pretrain_lm.py 与 train.py 共享同一个 split seed 以复现分类器 train/val 划分，增加耦合
- 额外获得的 LM 训练数据 ≈ 21k docs（0.5× labeled），边际收益较小
- 72.5k docs × 260 tokens ≈ 19M tokens 已足够（AWD-LSTM 在 WikiText-2 / 2M tokens 能训 PPL 60+）
- 实践上 B 方案估计仅比 A 方案差 0.1-0.2% acc，与避免的泄漏量级相当，**净 trade-off 近为 0**

### 4.4 BPTT 切片策略
```
输入 x = seq[i : i+bptt_len]
目标 y = seq[i+1 : i+bptt_len+1]
Loss = cross_entropy(logits, y, ignore_index=PAD)
```

**配置项**：
- `bptt_len: 128` （短序列，加速训练；IMDB 平均长度 260，128 足以学到大部分依赖）
- `batch_size: 64`
- 估算 token 数：72.5k 文本（unlabeled_train + test）× 平均 260 token ≈ 19M tokens
- 一个 epoch 步数：19M / (64 × 128) ≈ 2300 steps（约 4-6 分钟 / epoch on GPU）

### 4.5 不连续 BPTT
为了让一个 batch 内的不同位置看到完全独立的上下文，**不**做跨 batch 的 hidden state 传递（最简实现，loss 略高但代码干净）。如需追求 perplexity 极致，可后续加 truncated BPTT with detach。

---

## 5. LM 模型（`models/lstm_lm.py`）

```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=192,
                 num_layers=2, dropout=0.4, embed_dropout=0.3,
                 tie_weights=True, embedding_init=None, pad_idx=0):
        # NOTE: vocab_size here = classifier_vocab_size + 1 (extra row for EOS).
        # Embedding row [vocab_size-1] is the EOS token, used only during LM training.
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_init is not None:
            # IMPORTANT: embedding_init has V_cls rows (from w2v), but our embedding has V_cls+1 rows.
            # Copy only the first V_cls rows; the trailing EOS row stays at default random init.
            n_init = embedding_init.size(0)
            assert n_init <= self.embedding.weight.size(0), \
                f"embedding_init has {n_init} rows but LM embedding has {self.embedding.weight.size(0)}"
            self.embedding.weight.data[:n_init].copy_(embedding_init)  # warm start from w2v; EOS row random
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        # adapter: hidden_dim -> embed_dim, then tied with embedding.T
        self.adapter = nn.Linear(hidden_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size, bias=False)
        if tie_weights:
            self.proj.weight = self.embedding.weight  # tie
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T) -> logits: (B, T, V)
        emb = self.embed_dropout(self.embedding(x))
        h, _ = self.lstm(emb)
        h = self.dropout(h)
        h = self.adapter(h)
        return self.proj(h)
```

---

## 6. LM 训练（`engine/lm_trainer.py`）

### 6.1 关键超参
```yaml
lm:
  enable: true
  bptt_len: 128
  batch_size: 64
  epochs: 8                  # 早停可能在 5-7
  lr: 1.0e-3                 # 比分类器高，因为 LM 更难拟合
  weight_decay: 1.0e-6       # 几乎不要 wd（LM 本身就是强 implicit regularizer）
  grad_clip: 0.5             # LSTM-LM 梯度容易爆，clip 严
  warmup_ratio: 0.05
  lr_scheduler: warmup_cosine
  embed_dropout: 0.3
  dropout: 0.4
  init_from_w2v: true        # 用现有 w2v 当 embedding 初值
  early_stop_patience: 2     # 看 val perplexity
  val_ratio: 0.05            # 5% 文档当 LM val (不与下游分类 val 重叠)
```

### 6.2 损失与指标
- **Loss**：`F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=PAD)`
- **Perplexity**：`exp(mean_loss)`，每 epoch 在 LM val 上评估
- **早停**：val perplexity 2 个 epoch 不下降即停

### 6.3 优化器
- AdamW，β=(0.9, 0.999)
- gradient clip = 0.5（严，LSTM-LM 出名易爆）
- mixed precision 可选（A100/H100 上）

### 6.4 输出
- `results_lm/<timestamp>/lm_ckpt.pt`：最佳 perplexity 的权重
- 内含：`{"model_state": ..., "vocab_hash": ..., "config": ..., "val_ppl": ...}`
- `vocab_hash` 是关键：分类器加载时验证 vocab 一致性

---

## 7. 权重转移（`utils/weight_transfer.py`）

```python
def transfer_lm_to_classifier(
    lm_state: dict, classifier: LSTMClassifier, logger=None
) -> dict:
    """
    Map a forward LM's state_dict into a classifier with BiLSTM.
    Returns dict with keys 'transferred' / 'skipped' for logging.
    """
    transferred, skipped = [], []
    cs = classifier.state_dict()

    # 1. Embedding (drop the trailing EOS row that LM-only vocab has)
    if "embedding.weight" in lm_state:
        V_cls = cs["embedding.weight"].shape[0]
        cs["embedding.weight"].copy_(lm_state["embedding.weight"][:V_cls])
        transferred.append(f"embedding.weight (clipped {lm_state['embedding.weight'].shape[0]} -> {V_cls})")

    H = classifier.lstm.hidden_size
    num_layers = classifier.lstm.num_layers

    for L in range(num_layers):
        # weight_ih
        lm_ih = lm_state.get(f"lstm.weight_ih_l{L}")
        cls_ih = cs.get(f"lstm.weight_ih_l{L}")
        if lm_ih is not None and cls_ih is not None:
            if lm_ih.shape == cls_ih.shape:
                cls_ih.copy_(lm_ih)
                transferred.append(f"lstm.weight_ih_l{L}")
            elif L > 0 and cls_ih.shape[1] == 2 * lm_ih.shape[1]:
                # layer 1+ in BiLSTM: input dim is 2H (fwd+bwd concat)
                # Place LM weight on the "forward layer-0 output" half, zero the other.
                cls_ih.zero_()
                cls_ih[:, :lm_ih.shape[1]].copy_(lm_ih)
                transferred.append(f"lstm.weight_ih_l{L} (zero-padded)")
            else:
                skipped.append(f"lstm.weight_ih_l{L} (shape mismatch)")

        # weight_hh, bias_ih, bias_hh: shapes always match
        for k in (f"lstm.weight_hh_l{L}", f"lstm.bias_ih_l{L}", f"lstm.bias_hh_l{L}"):
            if k in lm_state and k in cs and lm_state[k].shape == cs[k].shape:
                cs[k].copy_(lm_state[k])
                transferred.append(k)

    # Backward direction (_reverse): leave as random init in v1
    classifier.load_state_dict(cs)
    if logger:
        logger.info(f"LM transfer: {len(transferred)} transferred, {len(skipped)} skipped")
        for n in transferred: logger.info(f"  ✓ {n}")
        for n in skipped:     logger.info(f"  ⨯ {n}")
    return {"transferred": transferred, "skipped": skipped}
```

### 7.1 转移后健全性检查
- forward `model(x)` 前后比较：转移前后输出应有显著差异（不是恒等映射）
- val acc 在加载 LM 权重后立刻评估：应 > 50%（否则转移有 bug）

---

## 8. 微调协议（`train.py` 改动）

### 8.1 Discriminative Learning Rate

```python
# 三组参数，三档 LR：
# - embedding:   1e-5   (LM 预训练已经很好，微调即可)
# - lstm body:   1e-4   (中等强度更新)
# - mhattn pool + classifier head: 5e-4 (随机初始化，需充分训练)

param_groups = [
    {"params": model.embedding.parameters(),   "lr": 1.0e-5},
    {"params": model.lstm.parameters(),         "lr": 1.0e-4},
    {"params": list(model.attn.parameters()) + list(model.classifier.parameters())
                + list(model.feat_norm.parameters()),
                                                "lr": 5.0e-4},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
```

**实现点（trainer.py 要改）**：当前 `engine/trainer.py:train()` 函数在内部构造 optimizer。为了在 LM 模式下使用 discriminative LR，需要添加一个可选参数：

```python
# engine/trainer.py
def train(model, train_loader, val_loader, cfg, device, *,
          ckpt_path, logger, tag,
          optimizer=None,                     # NEW: 允许外部传入
          ...):
    if optimizer is None:
        # 现有默认逻辑，后向兼容
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=cfg.lr, weight_decay=cfg.weight_decay)
    # ... 其余代码不变
```

然后 `train.py` 在 LM 模式下构造三档 optimizer 并传入 `train(...)`。本修改完全向后兼容 (optimizer=None 则走原逻辑)。

### 8.2 Gradual Unfreezing（可选，强化版）

```
Epoch 1-2:   freeze embedding + lstm   (只训 attn + head)
Epoch 3-4:   unfreeze lstm layer 1     (embedding 仍 frozen)
Epoch 5+:    unfreeze all              (走 discriminative LR)
```

实现：每个 epoch 检查当前 phase，对相应参数 `requires_grad = True/False`。

**v1 简化**：不做 gradual unfreezing，直接用 discriminative LR + 全模型一起训。
**v2 强化**：加 gradual unfreezing。

### 8.3 与 Self-Training 的协同
- LM 预训练 ckpt 只在 init phase 加载一次
- self-training rounds 之间**不**重新加载 LM（沿用 init 训练后的权重）
- discriminative LR 在 fine-tune rounds 也保留

---

## 9. 配置变更

### 9.1 `config.yaml` 新增段
```yaml
lm:
  enable: true                # 整个 LM 流程开关
  ckpt_path: null             # 若指定则跳过预训练直接加载（缓存复用）
  bptt_len: 128
  batch_size: 64
  epochs: 8
  lr: 1.0e-3
  weight_decay: 1.0e-6
  grad_clip: 0.5
  warmup_ratio: 0.05
  embed_dropout: 0.3
  dropout: 0.4
  init_from_w2v: true         # 用现有 w2v 初始化 LM 的 embedding[:V_cls]; EOS 行随机初始化
  early_stop_patience: 2
  val_ratio: 0.05             # 从 unlabeled 50k 中按文档抽 5% 当 LM val
  val_source: unlabeled_only  # 严禁从 labeled 抽 LM val（避免与分类器 val 重叠）

train:
  ...
  use_discriminative_lr: true # 新增
  lr_embedding: 1.0e-5        # 新增
  lr_lstm: 1.0e-4             # 新增
  lr_head: 5.0e-4             # 新增
  # 旧的 lr 仅在 use_discriminative_lr=false 时使用
```

### 9.2 `utils/config.py` 新增
- `LMConfig` dataclass
- `TrainConfig` 加 `use_discriminative_lr / lr_embedding / lr_lstm / lr_head`

---

## 10. CLI 与入口

### 10.1 `pretrain_lm.py`（独立入口）
```bash
python pretrain_lm.py --config config.yaml
# 输出: results_lm/<timestamp>/lm_ckpt.pt
```
- 加载语料、训练 LM、保存 ckpt
- 不依赖 train.py，可独立调试

### 10.2 `train.py` 改动
1. 在加载 w2v 后、构建 classifier 后，**如果 `cfg.lm.ckpt_path` 存在**：调用 `transfer_lm_to_classifier(...)`
2. 构建 optimizer 时，**如果 `cfg.train.use_discriminative_lr`**：用三档 param_groups
3. 在 logger 里打印「转移了 X 个参数 / 跳过 Y 个 / val acc 加载后立即评估」

### 10.3 `config.yaml` 工作流
```yaml
lm:
  enable: true
  ckpt_path: ./results_lm/20260511_HHMMSS/lm_ckpt.pt   # 训完后填入
```

完整跑通流程：
```bash
# 1. 一次性预训练
python pretrain_lm.py --config config.yaml

# 2. 把生成的 lm_ckpt 路径填回 config.yaml 的 lm.ckpt_path

# 3. 跑分类训练（自动加载 LM + 走 discriminative LR）
python train.py --config config.yaml
```

---

## 10.5 Vocab 一致性强制（防 silent failure）

**潜在灾难**：gensim 多线程 + 随机性 → 两次训练 w2v即使语料一致，可能产生不同的 vocab 顺序。LM 权重转移到分类器后会出现「形状对了、语义错位」：用「表现正常但 acc 莫名其妙差」的方式 silent fail。

**强制工作流**：
```
1. pretrain_lm.py 训完后同时保存：
   - results_lm/<ts>/w2v.model       （gensim Word2Vec 模型）
   - results_lm/<ts>/idx2word.json   （顺序完全一致的 vocab 列表）
   - results_lm/<ts>/vocab_hash.txt  （md5(idx2word) 的哈希值）
   - results_lm/<ts>/lm_ckpt.pt      （包含内嵌的 vocab_hash 与 V_cls）

2. train.py 启动时必须：
   - 如果 cfg.lm.ckpt_path 不为空:
     a. 强制 cfg.preprocess.w2v_cache_path 指向同一个 results_lm/<ts>/w2v.model
     b. 如果不一致 → 报错退出，绝不允许重新训练 w2v
     c. 加载 w2v 后重新生成 idx2word，与保存的 idx2word.json 逐项对比
     d. 计算 vocab_hash，与 lm_ckpt 内的 hash 比对
     e. 任何一项不匹配 → raise + 退出

3. 双重保险：hash + 全量 idx2word 对比（不只是前几项）
```

**为什么不足够只存 hash**：
- hash 冲突概率极低但不为 0
- 全量对比 cost 极低（vocab 7.4 万 × 平均 字符串 7 字节 ≈ 0.5MB， JSON 加载 < 100ms）
- 安全边际收益远大于成本

**实现要点**：`utils/vocab_io.py` 提供 `dump_vocab(idx2word, path)` / `load_vocab(path)` / `vocab_hash(idx2word)` 三个函数，pretrain_lm.py 和 train.py 都调用同一份实现。

---

## 11. 验证策略

### 11.1 LM 阶段验证
| 指标 | 阈值 | 含义 |
|---|---|---|
| Train PPL @ epoch 1 | < 200 | 收敛中 |
| Val PPL @ best | < 80 | LM 学到了语言结构 |
| Val PPL > 150 | 红线 | LM 没学好，转移收益会很小，需 debug |

### 11.2 转移后第 1 个 epoch 验证
| 信号 | 含义 |
|---|---|
| Val acc 第 1 epoch ≥ 80% | 转移成功，权重有效 |
| Val acc 第 1 epoch < 70% | 转移可能有 bug（参数顺序、shape 错） |
| Val acc 第 1 epoch ≈ baseline 第 1 epoch | 转移基本无效，检查 LM 质量 |

### 11.3 整体收益验证
- 与 run 20260510_154607（baseline 92.35）对比
- 期望：init RAW-best ≥ **93.0%**，FINAL ≥ **93.5%**
- 若 init RAW < 92.5%：LM 收益不及预期，需要调研

---

## 12. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| LM 训练 loss 爆炸 | 中 | 高 | grad_clip=0.5，lr 从 1e-3 试探，必要时降到 5e-4 |
| Vocab 不一致（LM 与分类器） | 中 | **致命** | ckpt 内存 vocab_hash + V_cls，加载时校验 |
| **EOS 当 PAD 用** | ~~中~~已规避 | **致命** | §4.2：EOS 单独占 `V_cls` ID；ignore_index 只 ignore PAD |
| **LM train/val 泄漏** | ~~中~~已规避 | 中 | §4.3：先按文档 split 再 flatten；LM val 仅从 unlabeled 抽 |
| 权重转移 shape 错配 | 中 | 高 | 单元测试 `transfer_lm_to_classifier`：随机生成 LM state，转移后 forward 不报错；embedding 用 `[:V_cls]` 切片 |
| LM 过拟合（小语料） | 低 | 中 | 100k 文本足够；embed_dropout=0.3 + tied weights 已经是强正则 |
| Discriminative LR 调参敏感 | 中 | 中 | 先按 [1e-5, 1e-4, 5e-4] 跑；若 val acc 低于预期，扫 [3e-5, 3e-4, 1e-3] |
| 训练时间过长 | 低 | 低 | LM 8 epoch ≈ 1 小时；分类 fine-tune 与现 35 分钟相当 |
| 与 self-training 冲突 | 低 | 中 | 严格分离：LM 只在 init 加载，self-training rounds 不重置 |

---

## 13. 时间表

| 阶段 | 内容 | 估时 |
|---|---|---|
| **D1** | 编写 `lstm_lm.py`, `lm_dataset.py`, `lm_trainer.py`, `pretrain_lm.py` | 3-4 h |
| **D2** | 编写 `weight_transfer.py` + 单元测试 | 1-2 h |
| **D3** | 改 `config.py / config.yaml / train.py`，集成 LM 加载 + discriminative LR | 1-2 h |
| **D4** | LM 预训练实验（8 epoch ≈ 1 h），监控 PPL | 1.5 h |
| **D5** | 分类 fine-tune 实验，对比 baseline | 1 h |
| **D6**（如需） | 调 LR / unfreezing / 后向 LM | 2-4 h |
| **总计** | v1 跑通到验证 | **~10 h，2 个工作日** |

---

## 14. 后续扩展（v2+）

按性价比排序：

1. **后向 LM**（+0.2~0.5%）：训一个 reversed-input LM，转移到 BiLSTM 后向方向
2. **Gradual unfreezing**（+0.1~0.3%）：分阶段解冻，避免预训练知识被早期梯度冲坏
3. **AWD-LSTM 正则**（+0.2~0.4%）：weight drop, embedding drop, activation regularization
4. **更长 BPTT**（+0.05~0.15%）：bptt_len 128 → 256
5. **跨 batch hidden state 传递**（+0.05~0.1%）：truncated BPTT，提升 LM PPL

---

## 15. 完成判据

v1 视为成功，需同时满足：
- [ ] LM 训练完成，val PPL < 80
- [ ] 权重转移单元测试通过
- [ ] 加载 LM 后 val acc 第 1 epoch ≥ 80%
- [ ] 完整 pipeline 跑通（pretrain → load → fine-tune → self-training → predict）
- [ ] **FINAL val acc ≥ 93.5%**（保守）或 **≥ 94.0%**（达成主目标）

---

## 16. 决策点速览

如果你现在 commit 这个计划，**第一步该做什么**：

1. **先确认我对你 vocab/embed 配置的理解正确**：
   - vocab_size = 74058（含 PAD/UNK）
   - embed_dim = 300
   - 分类器 hidden_dim = 192, num_layers = 2

2. **决定要不要保留 w2v 作为 LM 的 embedding 初值**（推荐：是。可省 1-2 epoch 收敛时间）

3. **决定 LM ckpt 的复用策略**：
   - 单次预训练，多次 fine-tune（推荐）
   - 或每次 fine-tune 重新 pretrain（不推荐，浪费时间）

4. **下一步动作**：
   - 我可以**直接开始实现**（按 §13 时间表）
   - 或者你**先看完文档提问 / 修改决策**，我再实现

---

**文档 owner**：Cascade
**最后更新**：2026-05-10
**关联 run**：baseline = 20260510_154607（FINAL 92.35）；目标 = 下一个 run ≥ 94.0
