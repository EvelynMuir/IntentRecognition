# Record 0306: Structure- and Uncertainty-aware Intent Learning

## 1. 目标

根据 [docs/design/design_0306_StructureUncertaintyAware.md](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/docs/design/design_0306_StructureUncertaintyAware.md) 落地第一版 **SUIL**，并在代码层面完成最小可运行实现。

本轮实现目标不是一次性做完整论文版，而是先做一个和强 baseline 兼容、风险最低、便于后续 ablation 的版本。

---

## 2. 基线落点确认

先确认了当前最适合作为 SUIL 载体的 baseline：

- 模型主干是 `frozen CLIP ViT + CLS + patch mean + 2-layer MLP`
- 对应实现路径是：
  - [src/models/components/clip_vit_layer_cls_patch_mean.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/src/models/components/clip_vit_layer_cls_patch_mean.py)
  - [src/models/intentonomy_clip_vit_layer_cls_patch_mean_module.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/src/models/intentonomy_clip_vit_layer_cls_patch_mean_module.py)
- loss 是 `AsymmetricLossOptimized`
- 验证/测试阶段已有一套 `macro F1` 驱动的评估逻辑

这条路径适合接 SUIL，因为：

- backbone/head 已经稳定；
- loss 是 multi-label ASL，方便做 confidence-aware weighting；
- 评估逻辑里已经支持 `class_thresholds`，方便接 class-wise calibration。

---

## 3. 迭代过程

### 3.1 迭代一：先打通 uncertainty 分支

最先解决的是 soft label 的保留问题。

原始 dataloader 在 `binarize_softprob=True` 时，会直接把 `category_ids_softprob` 覆盖成二值标签。这样虽然适合 baseline，但会导致 SUIL 看不到原始的标注一致性分数。

因此做了一个向后兼容的改动：

- `labels` 继续保持原有行为
- 新增 `soft_labels`
  - 对训练集，保留原始 `category_ids_softprob`
  - 对验证/测试集，没有 soft 标注时，直接复制 `labels`

对应改动文件：

- [src/data/intentonomy_datamodule.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/src/data/intentonomy_datamodule.py)

这个改动的好处是：

- 不破坏已有实验
- SUIL 可以直接消费 soft score
- 后续如果要做别的 uncertainty-aware loss，也不用再动数据管线

### 3.2 迭代二：hierarchy 选“动态构造”而不是直接读外部 tensor

一开始考虑过两条 hierarchy 路径：

1. 直接读取 `hierarchy_labels_*` 或 `adj_hierarchy.pkl`
2. 从 `label_tree.ipynb` 里还原出固定的 `28 -> 18 -> 15 -> 9` 层级映射，然后在代码里动态构造 coarse targets / parent-child relations

最后选了第二条，原因是：

- 更稳定，不依赖外部序列化格式
- 不要求样本顺序和外部保存文件强绑定
- 训练、验证、测试都可以统一从 fine labels 推出 coarse labels

我从 `label_tree.ipynb` 中提取了三层 hierarchy 分组定义，并实现了：

- `noisy-or / max` 聚合
- `coarse target` 动态构造
- 基于聚合后父节点概率的 coarse-level BCE
- `parent-child consistency loss` 工具函数（保留作分析/备选，不作为当前默认训练损失）

对应新增文件：

- [src/models/components/intentonomy_hierarchy.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/src/models/components/intentonomy_hierarchy.py)

### 3.3 迭代三：calibration 选 bias-only 为默认，但保留 affine 扩展口

设计文档里 calibration 有两种形式：

- `bias-only`
- `scale + bias`

本轮实现里两种形式都留了接口，但默认使用 `bias-only`，原因是：

- 更稳
- 参数更少
- 更接近“learnable class-specific threshold”的解释

实现上最终采用了下面这个约定：

- 训练时 loss 作用在 **calibrated logits / calibrated probabilities**
- 验证/测试时，评估同样直接使用 **calibrated probabilities**
- 当 `use_learned_thresholds_for_eval=true` 时，评估阈值保持为统一的 `0.5`，因为 calibration 已经直接写入 score space

这样做的好处是：

- 训练和评估处在同一个决策空间里，不会出现 train/test score semantics 不一致
- 不需要把 bias 额外翻译回一套新的 per-class threshold
- `bias-only` 和 `affine` 都可以沿用同一套评估接口

### 3.4 迭代四：hierarchy loss 最终落成 coarse supervision，而不是 parent-child hinge

一开始我有两个备选：

- 对父子边直接做 `parent-child consistency loss`
- 对由 fine prediction 聚合出来的 coarse probability 直接做 `BCE`

最后默认采用了第二条，即 **coarse supervision on calibrated probabilities**。原因是：

- 当前 parent probability 是由 child probability 通过 `noisy-or / max` 确定性聚合得到的
- 在这种实现下，`child > parent` 的情况天然很少出现
- 因此 parent-child hinge 在训练中容易退化成接近 `0` 的弱约束

最终实际训练的 hierarchy 项是：

- 先从 calibrated fine probabilities 构造 `28 -> 18 -> 15 -> 9`
- 再对各层 coarse probabilities 与动态构造的 coarse targets 做 `BCE`
- 可选地只对更高层 coarse levels 追加 auxiliary BCE

这样 structure 和 calibration 仍然是在同一个决策空间里协同优化的，只是层级监督信号从“边约束”调整成了“聚合节点监督”。

---

## 4. 当前落地的 SUIL

### 4.1 新增模块

新增了一个新的 LightningModule：

- [src/models/intentonomy_clip_vit_suil_module.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/src/models/intentonomy_clip_vit_suil_module.py)

这个模块复用了原始 `layer_cls_patch_mean` 的 backbone/head，只在 module 层增加了三件事：

1. `confidence-aware ASL`
2. `hierarchy regularization`（当前默认形态是 induced coarse supervision）
3. `class-wise calibration`

### 4.2 当前默认配置

第一版默认采用“最小稳定版”：

- `use_confidence_aware_supervision: true`
- `confidence_mapping: discrete`
- `confidence_weight_low/mid/high = 1.0 / 1.15 / 1.3`
- `use_hierarchy_regularization: true`
- `hierarchy_aggregation: noisy_or`
- `hierarchy_loss_weight: 0.1`
- `use_coarse_auxiliary_loss: false`
- `use_classwise_calibration: true`
- `calibration_mode: bias`
- `use_learned_thresholds_for_eval: true`

### 4.3 新增 Hydra 配置

新增配置文件：

- [configs/model/intentonomy_clip_vit_suil.yaml](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/configs/model/intentonomy_clip_vit_suil.yaml)
- [configs/experiment/intentonomy_clip_vit_suil.yaml](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/configs/experiment/intentonomy_clip_vit_suil.yaml)

默认实验仍然保持：

- `data.binarize_softprob: true`
- `ViT-L/14`
- `layer_idx: 24`

也就是说，这一版 SUIL 是直接叠加在当前强 baseline 之上的，而不是重新起一个复杂骨干。

---

## 5. 验证记录

### 5.1 环境

本轮验证使用用户指定环境：

```bash
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda
```

确认导入版本：

- `torch 2.9.1+cu128`
- `lightning 2.6.0`

### 5.2 已完成验证

完成了以下几项轻量验证：

1. `py_compile` 通过

验证文件：

- `src/data/intentonomy_datamodule.py`
- `src/models/components/intentonomy_hierarchy.py`
- `src/models/intentonomy_clip_vit_suil_module.py`

2. 数据集 sample 返回值验证通过

实际检查结果：

- `IntentonomyDataset(..., binarize_softprob=True)` 会返回：
  - `image`
  - `image_id`
  - `labels`
  - `soft_labels`
- 对训练样本，`labels` 为二值化结果，`soft_labels` 保留原始非零值，例如 `0.3333333`

3. 单元测试通过

测试文件：

- [tests/test_suil.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/tests/test_suil.py)

当前覆盖了两个关键行为：

- 当 coarse parent 被高置信误报、targets 全为 0 时，hierarchy loss 会产生明显惩罚
- `model_step` 返回的是 calibrated probabilities；当 calibration 打开时，评估阈值仍保持统一的 `0.5`

### 5.3 CLI 级验证中的发现

尝试用 `src/train.py` 做最小运行验证时，先遇到一个 Hydra 细节问题：

- `trainer.limit_train_batches`
- `trainer.limit_val_batches`

这两个字段不在默认 `trainer` config 的结构体里，因此必须写成：

```bash
+trainer.limit_train_batches=1
+trainer.limit_val_batches=1
```

而不能直接写成：

```bash
trainer.limit_train_batches=1
trainer.limit_val_batches=1
```

这个问题是命令行配置覆盖规则导致的，不是 SUIL 代码本身的错误。

### 5.4 短实验中发现并修复的真实 bug

第一次真实短实验没有通过，报错如下：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x1536 and 1024x512)
```

定位后发现，这不是 SUIL loss 的问题，而是底层组件
[src/models/components/clip_vit_layer_cls_patch_mean.py](/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/src/models/components/clip_vit_layer_cls_patch_mean.py)
对 `ViT-B/32` 的 `hidden_dim` 做了错误的模型名硬编码。

问题本质是：

- `ViT-B/32` 的 transformer hidden width 实际是 `768`
- 但原实现把它写成了 `512`
- 因此 `CLS + patch mean` 拼接后实际是 `1536`
- 线性层却按 `1024` 输入维度初始化，最终在第一层 `Linear` 处报错

修复方式是：

- 不再依赖模型名硬编码
- 优先从实际加载出来的 `CLIP backbone.width` / `transformer.width` 推断 hidden width

这次修复也顺手提升了 baseline 组件本身对不同 CLIP backbone 的兼容性。

### 5.5 已成功跑通的短实验

修复后，使用以下命令成功跑通了一轮真实短实验：

```bash
python src/train.py \
  experiment=intentonomy_clip_vit_suil \
  trainer=cpu \
  +trainer.fast_dev_run=1 \
  test=false \
  callbacks=none \
  logger=csv \
  data.batch_size=2 \
  data.num_workers=0 \
  model.net.clip_model_name="ViT-B/32" \
  model.layer_idx=12 \
  model.net.layer_idx=12 \
  model.compile=false
```

运行环境：

- `conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda`

实际结果显示：

- 模型成功实例化
- datamodule 成功加载
- optimizer / scheduler 成功创建
- 训练成功跑过 1 个 train batch
- 验证成功跑过 1 个 val batch
- Lightning 正常结束于：

```text
Trainer.fit stopped: `max_steps=1` reached.
```

本次短实验记录到的关键指标为：

- `train/loss: 0.030`
- `val/loss: 0.227`
- `val/f1_micro: 0.000`
- `val/f1_macro: 0.000`
- `val/mAP: 12.500`

这些数值本身没有研究意义，因为：

- 只跑了 `fast_dev_run=1`
- 只使用 1 个 train batch 和 1 个 val batch
- 目的是确认训练链路和评估链路是否可执行

但它足以说明：

- **SUIL 的训练主路径已经可以真实运行**
- **不是只停留在静态检查或 dummy test**

当前状态更新为：

- 代码路径已打通
- 轻量 smoke test 已通过
- 真正的短训练 / fast-dev-run 已确认训练主循环正常
- 仍然需要正式短实验来判断 loss 曲线和指标是否合理

---

## 6. 当前方法结论

截至当前这版代码，SUIL 的实际落地形态已经比较清晰：

- backbone 仍然完全沿用强 baseline：`frozen CLIP ViT + CLS + patch mean + 2-layer MLP`
- uncertainty 分支不是做 soft-target regression，而是保留 binary target，只把 `soft_labels` 用作正类监督强度
- hierarchy 分支当前默认不是 graph reasoning，也不是 parent-child hinge，而是从 fine prediction 诱导 coarse probability 后做多层 BCE
- calibration 分支直接作用在 train/eval 共用的 score space 上，因此评估阶段继续使用统一 `0.5` 阈值即可

这意味着当前方法的核心结论不是“已经证明涨点”，而是：

- **方法形态已经收敛到一个可运行、可解释、便于 ablation 的最小稳定版**
- **这版方法主要解决 supervision uncertainty、hierarchical label structure 和 class-wise decision bias**
- **它没有再次引入已经被前序实验证明容易伤害 baseline 的复杂视觉结构建模**

除此之外，本轮实现还顺手暴露并修复了 baseline 组件里的一个真实兼容性问题：

- `clip_vit_layer_cls_patch_mean` 对 `ViT-B/32` 的 hidden width 硬编码错误
- 修复后，SUIL 的短实验能在 `ViT-B/32` 上顺利跑通
- 这个修复本身也提升了 baseline 组件对不同 CLIP backbone 的稳健性

因此，当前最稳的研究判断是：

- 先把 SUIL 当成“强 baseline 上的误差建模层”，而不是新的视觉主干
- 先验证它是否能稳定提升 `macro F1`
- 如果能涨，再讨论是否扩到更复杂的 coarse auxiliary、affine calibration 或后续结构分支

---

## 7. 下一步建议

接下来最值得做的事情是：

1. 先跑一个真正的短实验

建议优先验证：

- `ViT-B/32 + layer=12` 的轻量版本
- `batch_size` 小一点
- 先看 loss 是否稳定、`val/f1_macro` 是否正常记录

2. 做最小 ablation

建议至少拆成四组：

- baseline
- baseline + confidence
- baseline + confidence + hierarchy
- baseline + confidence + hierarchy + calibration

3. 如果第一版能稳住，再开启两个扩展项

- `use_coarse_auxiliary_loss: true`
- `calibration_mode: affine`

我不建议在这之前就把 multi-layer fusion 一起塞进来，否则很难判断涨点来源。
