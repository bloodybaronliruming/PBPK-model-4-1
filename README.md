# 数据审计、底座与微观端点训练：运行说明

所有入口放在 `scripts`。这批程序实现：来源审计 → 新数据视图 → 固定骨架折 → 单任务与受保护级联 → 共享 MLP / MMoE 多任务训练。融合/PBPK 程序属于后续批次，本次没有以空壳文件代替它们。

程序供人工按顺序运行，不会自动启动下一阶段。旧 `data/processed` 和原始 SQLite 只读；新版本默认写入 `data/processed_v2`。输出目录已经存在时会立即报错，不覆盖模型或数据。若需重跑，请用新的 `--output` 路径，并在后续步骤用 `--audit-dir`、`--datasets-dir`、`--splits-dir` 指向对应版本。训练重跑使用新的 `--run-name`。

## 1. 文件与职责

| 程序 | 用途 | 默认输出 |
|---|---|---|
| `audit_endpoint_sources.py` | 匹配原始 ChEMBL，审计物种/端点/单位/删失及旧标签契约 | `data/processed_v2/audit` |
| `papp_provenance_overrides.py` | 读取原文 DOI 单位裁定，并在审计时选择性修正 Papp 的单位解释 | 由 `audit_endpoint_sources.py` 写入审计谱系 |
| `rebuild_endpoint_datasets.py` | 重建来源长表、精确测定任务视图及变换标签 | `data/processed_v2/datasets` |
| `build_split_manifest.py` | 固定 train/val/test 成员、结构排除清单、共享训练折 | `data/processed_v2/splits` |
| `dmpk_toolkit.py` | 公共训练模块；另提供 `check`、`predict`、`evaluate-test` 命令 | 由调用入口决定 |
| `train_fu.py` | 默认人血浆 fu | `models/stl/fu__human__plasma/<run-name>` |
| `train_CLint.py` | 默认人肝微粒体 apparent CLint | `models/stl/CLint__human__microsome/<run-name>` |
| `train_Papp.py` | 默认人源 Caco-2 A→B Papp | `models/stl/Papp__human__caco2_ab/<run-name>` |
| `train_F.py` | 默认人绝对口服生物利用度 F | `models/stl/F__human__absolute_oral/<run-name>` |
| `train_CL.py` | 默认人全身静脉 CL | `models/stl/CL__human__systemic_iv/<run-name>` |
| `train_VDss.py` | 默认人稳态静脉 VDss | `models/stl/VDss__human__steady_state_iv/<run-name>` |
| `train_Thalf.py` | 默认人终末静脉 Thalf 直接诊断基线 | `models/stl/Thalf__human__terminal_iv/<run-name>` |
| `train_else.py` | 显式选择物种、端点和体系的动物先验训练 | `models/stl/<task_id>/<run-name>` |
| `results_analysis.py` | 在 test 冻结时生成英文出版图表 | `results/analysis/<name>` |
| `cross_species_analysis.py` | 对照直接与跨物种模型的英文图表 | `results/analysis/cross_species_human_v1` |
| `train_cross_species.py` | 受保护大鼠→人软先验级联公共程序 | `models/cascade/cross_species/<task>/<run-name>` |
| `train_cross_species_{fu,CL,VDss,F,Thalf}.py` | 各端点跨物种级联入口 | 同上 |
| `train_multitask.py` | 共享 MLP / dense MMoE 掩码多任务训练 | `models/multitask/<run-name>` |
| `multitask_three_seed_analysis.py` | 三种子 MTL/STL 汇总、配对 OOF bootstrap、英文出版图表和模型选择报告 | `results/analysis/multitask_pappfixed_v3_three_seed_v1` |

配套模块：`pipeline_common.py`（所有入口共享自检和产物协议）、`endpoint_rules.py`（版本化分类/换算规则）、`neural_models.py`（Res-MLP/D-MPNN）。`tests/test_pipeline.py` 是合成夹具功能测试，独立于正式训练数据。

## 2. 执行前准备

```bash
cd /home/shahab/MTL-model-4-1
conda activate oneadmet
```

使用项目已有环境。核心依赖为 RDKit、NumPy、pandas、SciPy、scikit-learn、joblib、tqdm、threadpoolctl；选用 XGBoost/LightGBM/CatBoost/PyTorch 时才检查对应依赖。默认基线为常数模型、Ridge 和 RF，默认 CPU、4 线程。

每个入口调用同一 `startup_self_check`，以显式异常检查缺失文件、已有输出、契约/版本和前序完成标记；日志保存在 `results/pipeline_logs`。长操作显示进度条。阶段先写临时目录，全部成功后发布 `complete.json`，下游会校验产物 SHA-256。

输入必须已存在：

```text
data/processed/chembl_train.csv
data/processed/chembl_val.csv
data/processed/chembl_test.csv
data/raw/chembl_37/chembl_37_sqlite/chembl_37.db
```

旧人源子表可用于一致性诊断，但不是新标签真相。原始数据库必须含标准 ChEMBL 的 activities、assays、compound_structures、molecule_dictionary、docs、assay_parameters 表及所用字段。

## 3. 批次 A：按顺序执行数据程序

### A1. 来源审计

先检查路径和数据库结构，再正式执行：

```bash
python scripts/audit_endpoint_sources.py --check-only
python scripts/audit_endpoint_sources.py
```

输出重点：

- `molecule_mapping.csv`：按完整 standard InChIKey 的映射；未匹配/多归属记录明确标记。
- `source_records.csv`：activity/assay/doc、原物种、单位、关系符、assay 原文和参数，以及分类/换算结果。
- `task_audit_counts.csv`：accepted、censored、review 的数量与原因。
- `legacy_contract_issues.csv`：旧 raw/target/mask 不一致计数。

审计不会把所有带 CL 字样的测定都当成同一端点。`mL/min/kg` 不换成微粒体 `µL/min/mg`；`mL/min/g` 的 g 未说明蛋白材料时留在 review。PPB 到 fu 的补数转换同步反转不等式方向。未知物种、未知关系符、fu=0/1、来源质量标记和不明确的实验体系不进入默认精确回归。

若只想试跑：

```bash
python scripts/audit_endpoint_sources.py --limit 100 --output /tmp/dmpk_audit_trial
```

`--limit` 产物明确标记 partial，数据重建程序会拒绝消费，不能把试跑当完整数据。映射使用保守的完整 InChIKey，未做模糊结构匹配；未匹配记录不会自动强行关联。当前规则不从原文猜测体重、生理缩放、酸碱常数或检测限。

### A2. 重建任务数据

```bash
python scripts/rebuild_endpoint_datasets.py --check-only
python scripts/rebuild_endpoint_datasets.py
```

输出重点：

- `source_long.csv` 保留全部审计候选来源；`review_records.csv` 与 `censored_records.csv` 分开保存。
- `task_records.csv` 与 `tasks/<task_id>.csv` 只包含 accepted 精确测定。
- `task_registry.json` 固定任务物种、体系、单位和数学变换。
- `task_counts.csv` 报告每任务/每集合的记录数和独立分子数。

重复测量只在相同“分子 × 任务 × assay × doc”内用中位数聚合，保留活动 ID、源数量、最小/最大值。不同研究/体系保留独立记录，不将旧全局聚合值伪装成人体测量。训练时同一分子的多条记录合计权重相等。

raw 值是规范单位下的观测，`target_value` 是未标准化的变换值，`mask=1` 表示已确认精确标签。fu 内部值使用真实 logit，保留小于 0.01 的差异；CLint/Papp 使用 log10；F（审计支持，训练入口后置）使用原空间。z-score 均值/方差不在此阶段全局拟合，而在每个训练范围内单独学习。

review/censored 表不是被删除的数据。当前训练实现是精确标签基线，没有实现删失似然或把 `<LLOQ` 当成精确零。若任务容量不足，应根据来源文献改进明确的规则并重跑新版本，不能直接把 review 全部改为 accepted。

### A3. 固定骨架与交叉验证折

```bash
python scripts/build_split_manifest.py --check-only
python scripts/build_split_manifest.py --folds 5 --seed 2026
```

原 train/val/test 成员不移动。结构分组采用 RDKit FragmentParent、去电荷、规范互变异构体、去立体信息后的 Murcko 骨架；无环分子按规范母体分组。若相同母体/骨架跨集合，相关成员全部在新视图中排除，理由写入 `excluded_structures.csv`，不根据标签选择保留哪一侧。

RDKit 对个别结构抛出 `RuntimeError`/`ValueError` 时，记录完整 SMILES 与错误到 `structure_failures.csv`，并设置 `eligible=False`、`fold_id=-1`。不退回未经规范化的结构参与分组，因此这些记录不会进入训练。

结构计算每 25 条保存一次检查点，默认位于 `data/processed_v2/_cache/structure_groups.sqlite`，与最终 `splits` 目录分离。中断后直接重跑同一命令，会复用已保存的成功/失败结果；缓存按 RDKit 版本及分组函数代码隔离。`--retry-failed` 可重试失败记录，`--cache-path` 可指定另一缓存。旧版在内存中计算、未保存的进度无法追溯恢复。只有已发布的最终输出目录才会阻止同路径重跑。

训练折统一生成一次并存入 `split_manifest.csv`。所有端点与物种按同一分子/骨架读取 fold；端点脚本不会调用随机 `train_test_split`。查看 `task_fold_counts.csv`，确认每个任务有足够有效训练骨架和验证/测试覆盖；环骨架分组不保证隔离所有更宽泛的类似物系列。

### A4. 版本化证据增量核验

在新 `processed_vN` 的三个阶段都完成后，使用 `verify_dataset_version.py` 对已发布基准和候选版本进行只读核验。该命令逐一验证 audit/datasets/splits 的完成标记与文件哈希，要求两个 `split_manifest.csv` 的 SHA-256 一致，并按任务与固定集合报告记录/分子增量。`--expect-delta` 可将预期增量变成显式断言；任何不一致都会以非零退出，不应继续训练。

```bash
python scripts/verify_dataset_version.py \
  --baseline-dir data/processed_v7 --candidate-dir data/processed_v8 \
  --expect-delta Thalf__human__terminal_iv:train:1:1
```

## 4. 批次 B：底座自检与第一个 fu 基线

```bash
python scripts/dmpk_toolkit.py check --task fu__human__plasma
python scripts/train_fu.py --check-only
python scripts/train_CLint.py --check-only
python scripts/train_Papp.py --check-only
python scripts/train_CL.py --check-only
python scripts/train_VDss.py --check-only
python scripts/train_Thalf.py --check-only
```

如果某任务不存在或覆盖的训练折不足，自检会给出错误与可用任务，不会启动长时间训练。`dmpk_toolkit.py` 是公共模块，**不需要作为独立训练任务先运行**；这里的 `check` 用于验证接口。

首先运行 fu，检查真实数据规模、耗时和产物：

```bash
python scripts/train_fu.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
```

默认额外运行常数基线。`--trials 3` 表示每算法最多 3 套候选参数；在外层固定折内用其余固定折进行内层调参。每个外层验证分子的标签不参与其模型的参数选择、特征插补/标准化或目标标准化。最后在全部合格 train 中重新调参/拟合最终模型。

使用的结构特征为 2,048 位 ECFP4 + 当前 RDKit 有名称清单的 2D 描述符；确定性结构计算可覆盖各集合，所有需要拟合的操作只能看训练范围。未使用旧来源不明的 joblib/NPZ 特征缓存。

Ridge 固定使用 LSQR 求解器，适配 ECFP 与描述符组成的高维共线矩阵。分数端点在反 logit 前提升为 float64，并裁剪到浮点可表示的开区间 `(0,1)`，避免 float32 饱和为精确 0/1 后导致预测文件转换失败。

正值端点的反 log10 变换也带有数据派生的数值保护：预测超出训练变换范围 8 个训练标准差时裁剪，并限制在 `[-300, 300]` 的可计算区间；模型对象记录 `inverse_clip_count`。这只防止 `10**z` 溢出，不代表该模型在极端化学空间中具有可靠外推能力，验证报告应检查该计数。

## 5. 批次 C：CLint 与 Papp，可受控并行

fu 的运行流程与产物验证后，执行：

```bash
python scripts/train_CLint.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
python scripts/train_Papp.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
```

这些微观单任务之间没有模型前置依赖，可以在两个终端同时执行；以上连续写法则是顺序运行。建议首次使用顺序运行，确认峰值内存后再开两个 CPU 作业，每作业 4 线程。不要同时让多个 GPU 作业争用 12 GB 显存。

其他物种或 fu 基质需显式选择，例如：

```bash
python scripts/train_fu.py --species rat --system plasma --check-only
python scripts/train_CLint.py --species human --system microsome_unbound --check-only
```

只有审计产生了相应任务，才能训练；不自动混合动物和人、血浆和血清、apparent 与 unbound CLint。

## 6. 批次 D：人体宏观 CL 与 VDss 的直接基线

当前严格来源视图中，CLint 只有 20 个训练分子，因此它的 `diagnostic_v1` 只能用于检查流程，不能作为 CL 的级联先验。先训练不依赖 CLint 的直接 CL 与 VDss 基线：

```bash
python scripts/train_CL.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
python scripts/train_VDss.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
```

保持 test 集冻结。完成后检查 `metrics.json` 和 `selection.json`；只有扩充并重新审计 CLint、F 与半衰期数据后，才开始 CLint→CL、CL/VDss→F 或 PBPK 级联。

## 7. 批次 E：Thalf 直接诊断基线

人体 Thalf 只有 43 个有效训练分子。先运行直接基线以确认标签、固定折和可达到的误差尺度；它不是可报告的最终模型，也不启用 test 评估：

```bash
python scripts/train_Thalf.py --algorithms ridge,rf --trials 3 --threads 4 --run-name diagnostic_v1
```

随后运行 `train_Thalf_cascade.py`。它为每个 Thalf 外层折重新拟合 CL、VDss 上游模型，排除该外层折；用于外层训练行的先验还排除该行自身折。不得把 CL/VDss 全训练模型对 Thalf 训练分子的预测直接作为特征。

```bash
python scripts/train_Thalf_cascade.py --check-only
python scripts/train_Thalf_cascade.py --threads 4 --run-name cascade_diagnostic_v1
```

## 8. 批次 F：大鼠跨物种先验

人体 F 只有 15 个训练分子，人体 Thalf 也只有 43 个；在它们上面继续调参不会产生稳健结论。先在样本量充足、体系明确的大鼠任务上训练单任务先验。每条命令读取同一固定骨架划分，任务之间可并行，但首次建议先跑 CL 与 VDss：

```bash
python scripts/train_else.py --endpoint CL --species rat --system systemic_iv --check-only
python scripts/train_else.py --endpoint VDss --species rat --system steady_state_iv --check-only

python scripts/train_else.py --endpoint CL --species rat --system systemic_iv --algorithms ridge,rf --trials 3 --threads 4 --run-name prior_v1
python scripts/train_else.py --endpoint VDss --species rat --system steady_state_iv --algorithms ridge,rf --trials 3 --threads 4 --run-name prior_v1
```

两项成功后，再依次建立 `fu__rat__plasma`、`F__rat__absolute_oral` 和 `Thalf__rat__terminal_iv` 的先验。人源模型不会和动物标签混合训练；后续跨物种程序只消费动物模型的受保护预测。

大鼠 F 与 Thalf 的专用入口如下：

```bash
python scripts/train_F.py --species rat --system absolute_oral --check-only
python scripts/train_Thalf.py --species rat --system terminal_iv --check-only
```

## 9. 批次 G：受保护的大鼠→人跨物种级联

每个人源外层验证折都会重新拟合大鼠模型，并排除该折；供人源外层训练行使用的大鼠预测还排除该行自身折。因此大鼠模型不可能见过目标行所在的统一结构折。大鼠预测作为单独的软特征拼接，物种标签和物种测定值不在同一监督目标中混合。

支持的同体系任务为 `fu/plasma`、`CL/systemic_iv`、`VDss/steady_state_iv`、`F/absolute_oral`、`Thalf/terminal_iv`。Papp 缺少匹配大鼠 Caco-2 任务；CLint 的人/大鼠严格数据均不足，两个端点不会被自动迁移。

先执行容量充分的 fu、CL、VDss；三项可在独立终端并行，每项最多 4 CPU 线程：

```bash
python scripts/train_cross_species_fu.py --check-only
python scripts/train_cross_species_CL.py --check-only
python scripts/train_cross_species_VDss.py --check-only

python scripts/train_cross_species_fu.py --threads 4 --run-name cross_species_diagnostic_v1
python scripts/train_cross_species_CL.py --threads 4 --run-name cross_species_diagnostic_v1
python scripts/train_cross_species_VDss.py --threads 4 --run-name cross_species_diagnostic_v1
```

再顺序执行 Thalf，并最后运行 F。人体 Thalf（43 个训练分子）与 F（13 个）只生成诊断结果，不解冻 test；严格划分后 F 没有合格 val 行，因此仅报告受保护 OOF，不伪造验证指标：

```bash
python scripts/train_cross_species_Thalf.py --check-only
python scripts/train_cross_species_Thalf.py --threads 4 --run-name cross_species_diagnostic_v1

python scripts/train_cross_species_F.py --check-only
python scripts/train_cross_species_F.py --threads 4 --run-name cross_species_diagnostic_v1
```

## 10. 批次 H：多任务共享表征与 MMoE

`train_multitask.py` 从测定级长表建立 molecule × task 标签矩阵。任务是 `endpoint × species × system`，不使用旧来源不明宽表。重复测定先按分子/任务取中位数；缺失标签只作为张量占位，mask=0，不参与监督损失、目标标准化或指标。

默认选择有效训练分子不少于 40、覆盖至少 3 个统一 fold 的任务；当前自检选择 29 个任务、22,306 个分子和 17,903 个训练分子。每个外层折中，特征插补/缩放及每个任务的目标变换统计量均只由其余训练折拟合。任务损失在 batch 内按“有标签任务”平均，避免大任务吞没小任务。

先运行共享 MLP 基线，再独立运行 dense MMoE；二者均只报告逐任务 OOF 和冻结验证指标，不能以不同量纲任务的全局平均选择赢家：

```bash
python scripts/train_multitask.py --check-only

python scripts/train_multitask.py --architectures shared --width 192 --epochs 80 --batch-size 64 --threads 4 --run-name shared_diagnostic_v1
python scripts/train_multitask.py --architectures mmoe --width 192 --experts 4 --epochs 80 --batch-size 64 --threads 4 --run-name mmoe_diagnostic_v1
```

如果 CUDA 自检通过，可将最后两条命令中的 `--device cuda` 加入；同一 GPU 上只能运行一个多任务作业。对每个人源任务，只有当 MTL 在 OOF 和冻结验证均不劣于对应 STL 时，才进入最终候选；否则保留 STL。当前首版不实现 Top-2 稀疏门控、Kendall 权重或在线物理闭环，避免在未验证共享收益前增加不可分辨的复杂度。

## 11. 扩展候选算法

已实现的候选为：`dummy,ridge,rf,extratrees,xgboost,lightgbm,catboost,resmlp,dmpnn`。

树模型扩展可单独运行，避免第一轮直接开启大预算：

```bash
python scripts/train_fu.py --algorithms extratrees,xgboost,lightgbm,catboost --trials 10 --threads 4 --run-name trees_v1
```

神经候选示例（确认 CUDA 可用后）：

```bash
python scripts/train_fu.py --algorithms resmlp,dmpnn --trials 3 --epochs 80 --batch-size 32 --threads 4 --device cuda --run-name neural_v1
```

Res-MLP 与 D-MPNN 为此底座的小型 PyTorch 实现。D-MPNN 使用有向键消息传递、反向边排除、分子聚合及数值特征支路；不是对 Chemprop 训练结果的复现。模型保存 CPU state_dict，可在无 GPU 的机器推理。

当前神经训练使用指定的固定 epoch 和 Huber 损失，不用外层验证/测试做 early stopping。`epochs` 属于需预先选择/内层验证的训练配置。当前没有预训练 SMILES/BRICS、多任务、机理损失、smearing 或融合模块。嵌套调参会显著放大运行量，先基线后扩大预算。

## 12. 输出检查与冻结测试

每个端点运行目录包含：

```text
complete.json                 # 产物 hash、代码 hash、数据版本、算法与 seed
feature_schema.json           # 指纹参数、描述符名字/顺序、RDKit 版本
tuning_history.json           # 外层范围内及最终训练范围的候选得分
metrics.json                  # 各算法 OOF 和 val；默认无 test 指标
selection.json                # 依据训练嵌套 OOF 选择的算法及注意事项
environment.json              # 实际包版本
<algorithm>/
    model.joblib              # 最终 train 模型、变换器和前处理
    lineage.json              # 训练分子/骨架、超参、任务定义
    oof_predictions.csv       # 带 row_id、fold、物理/变换空间的外层预测
    val_predictions.csv       # 有合格 val 标签时生成
    fold_<k>/model.joblib
    fold_<k>/lineage.json
```

程序会重载每个最终模型并核对预测。指标同时报告原空间和变换空间误差；fu 优化分子等权原空间 MAE，CLint/Papp 优化分子等权 log10 RMSE；另报低 fu 误差、倍数误差和相应计数。没有样本的 val 不伪造结果，常数标签下 R² 记为 null。

算法选择基于 train OOF，所选算法的同一 OOF 得分存在选择偏差；val/test 用于评估泛化。默认不计算测试性能。方案冻结后可直接评估保存模型，无需重训：

```bash
python scripts/dmpk_toolkit.py evaluate-test --run-dir models/stl/fu__human__plasma/baseline_v1 --output results/test_evaluation/fu_baseline_v1
python scripts/dmpk_toolkit.py evaluate-test --run-dir models/stl/CLint__human__microsome/baseline_v1 --output results/test_evaluation/clint_baseline_v1
python scripts/dmpk_toolkit.py evaluate-test --run-dir models/stl/Papp__human__caco2_ab/baseline_v1 --output results/test_evaluation/papp_baseline_v1
```

该命令使用运行目录中按训练 OOF 选定的模型，验证数据版本后输出独立测试报告，不更新权重和选择结果。端点训练入口另有 `--evaluate-test`，仅适用于方案已冻结时的新运行，不建议探索阶段开启。

任意新分子推理（输入需要 `smiles` 或 `Canonical_SMILES` 列）：

```bash
python scripts/dmpk_toolkit.py predict --model models/stl/fu__human__plasma/baseline_v1/rf/model.joblib --input new_molecules.csv --output results/new_fu_predictions.csv
```

普通推理结果标为 `inference_not_oof`，不能当作训练级联先验。输出含规范单位及物理/变换空间预测。

## 8. 为下一阶段级联预留的保护范围

例如，人体宏观模型的外层保护折是 0，微观模型可执行：

```bash
python scripts/train_fu.py --algorithms rf --trials 3 --exclude-fold 0 --run-name cascade_outer0
```

该运行所有拟合均排除统一 fold=0，内部 OOF 又在其余折中交叉拟合，保存 `excluded_fold_predictions.csv` 及祖先训练组。这是正确嵌套级联所需的子范围。实际宏观脚本仍需对所有端点/物种一致使用该范围，并为缺失微观标签的分子生成对应范围的预测；本次尚未自动构建完整宏观先验矩阵。

不能把 `baseline_v1` 的全 train 模型回代 train 当成先验；也不能将按全局 OOF 选出的算法误称为对每个外层折都独立选择。后续级联需要在保护范围内选择算法/超参或预先固定算法，再审计完整祖先链。

## 9. 本轮验证

2026-09-09：13 项功能测试全部通过，7 个入口的 `--help` 均通过，真实数据库启动结构自检通过。原 `data/processed` 中 16 份被登记 CSV 的 SHA-256 与第一阶段核查一致。详见 [验证记录](/home/shahab/MTL-model-4-1/scripts/VALIDATION.md)。

```bash
python -m unittest discover -s scripts/tests -v
```

测试夹具在独立临时目录中创建合成 SMILES/SQLite，验证：单位量纲、PPB 删失方向、低 fu 保真、结构分组、旧/新产物协议、固定折嵌套训练、所有候选算法的拟合与重载、三个训练入口、保护外层折、普通推理和冻结测试评估。

本轮另用真实数据库前 24 个分子进行了只读审计试跑，读取 52 条候选测定；不把这些非随机抽样的通过率当作全库质量比例。未执行真实数据全量重建、全量骨架重算或正式模型训练，测试指标不代表科研性能。

**实际执行顺序：A1 审计 → A2 重建 → A3 清单 → B 自检与 fu 基线 → C CLint/Papp → D CL/VDss 直接基线 → E Thalf 直接诊断 → 受保护级联 → 候选扩展 → 方案冻结后的测试。**

## 10. Papp 来源审计与 MTL 任务均衡采样

Papp 审计只读 `datasets` 中的任务记录及其 `source_long.csv` 谱系。默认审计 train/val，避免在模型选择前读取测试标签；输出每条高值记录的原始单位、规范单位、换算规则、测定、文献和方向证据：

```bash
python scripts/audit_papp_provenance.py --check-only
python scripts/audit_papp_provenance.py --threshold 10000 --output results/analysis/papp_provenance_audit_v2
```

审计不自动删除或修正记录。若核查确认单位、方向或来源错误，必须建立新的 audit/datasets/splits 版本，再重跑受影响的 Papp 模型；若未确认错误，保留当前版本及审计报告。

`train_multitask.py` 的默认 `molecule_uniform` 是已完成 MTL 的可复现基线。`task_balanced` 在每个 batch 为各任务抽取有标签锚点并以均匀分子补足，缓解稀疏任务在 molecule-uniform batch 中被低频看见的问题；它不会更改 split、标签或测试集。

```bash
python scripts/train_multitask.py --check-only --batch-sampler task_balanced
python scripts/train_multitask.py --architectures shared,mmoe --batch-sampler task_balanced \
  --seed 2026 --width 192 --experts 4 --epochs 80 --batch-size 64 --threads 4 \
  --run-name balanced_seed2026_v1
```

随后使用 `--seed 2027` 和 `--seed 2028`，每次使用不同 `--run-name`。最终比较须按相同架构、相同 seed 汇总 protected OOF 的均值和标准差；测试集在模型锁定前保持冻结。

## 11. Papp 原文单位校正：新版本审计、数据集与划分

原文核查确认 Papp 的单位错误只发生在部分 DOI，不能全局修改 `cm/s` 转换。`audit_endpoint_sources.py` 可读取裁定表：确认 `10^-6 cm/s` 的 DOI 会选择性保留原数值；也识别 ChEMBL 的 `10'-6 cm/s` 拼写变体。`unresolved` 默认进入 review，因而不会训练。旧的 `processed_v2` 和已完成模型不可修改。

```bash
python scripts/audit_endpoint_sources.py --check-only \
  --papp-decision-registry results/analysis/papp_provenance_decisions_v2.csv

python scripts/audit_endpoint_sources.py \
  --papp-decision-registry results/analysis/papp_provenance_decisions_v2.csv \
  --papp-unresolved-policy exclude \
  --output data/processed_v3/audit

python scripts/rebuild_endpoint_datasets.py \
  --audit-dir data/processed_v3/audit \
  --output data/processed_v3/datasets

python scripts/build_split_manifest.py \
  --datasets-dir data/processed_v3/datasets \
  --cache-path data/processed_v3/_cache/structure_groups.sqlite \
  --output data/processed_v3/splits
```

每个命令都必须成功发布新的 `complete.json` 才能进行下一个。之后重跑 Papp STL 和所有 MTL 种子，并显式传入 `--datasets-dir data/processed_v3/datasets --splits-dir data/processed_v3/splits`；测试集继续冻结。

## 12. 三种子 MTL 正式分析

三个 v3 任务均衡种子完成后运行：

```bash
python scripts/multitask_three_seed_analysis.py --check-only
python scripts/multitask_three_seed_analysis.py \
  --output results/analysis/multitask_pappfixed_v3_three_seed_v1
```

程序验证三个 MTL 运行的数据、划分、代码、任务与采样协议一致，并确认所有输入均未评估 test。输出包括种子级和汇总 CSV、逐分子配对 OOF bootstrap、600 dpi 英文 PNG、模型选择表及同目录英文分析报告。

## 13. 冻结候选与一次性测试评估

先发布候选登记。该程序只读取 OOF/validation 结果，并逐个重载三个 Shared MLP 检查预测一致性；不读取测试标签：

```bash
python scripts/freeze_final_candidates.py --check-only
python scripts/freeze_final_candidates.py
```

当前登记将人血浆 fu 固定为 seeds 2026/2027/2028 的 Shared MLP 物理尺度均值集成；Papp、CL、VDss 和 Thalf 固定为预先选定的 STL RF。登记发布后运行一次统一测试：

```bash
python scripts/evaluate_frozen_candidates.py --check-only
MPLCONFIGDIR=/tmp/mtl_matplotlib python scripts/evaluate_frozen_candidates.py \
  --confirm-frozen-test
```

第二条命令不重新拟合，不允许更新选择，输出逐记录预测、完整指标 JSON/CSV、英文 600 dpi observed-vs-predicted 图和指标表。Thalf 仅有 2 个测试分子，程序将其明确标为探索性结果。

## 14. 冻结测试诊断与 Papp 单位敏感性

测试候选发布后，使用同一份不可修改的预测执行分子级 bootstrap、残差校准、ECFP4 最近训练分子适用域和 Papp 三情景单位敏感性分析：

```bash
MPLCONFIGDIR=/tmp/mtl_matplotlib python scripts/analyze_frozen_test.py --check-only
MPLCONFIGDIR=/tmp/mtl_matplotlib python scripts/analyze_frozen_test.py
```

Papp 三种情景为：保留冻结测试原值、将 DOI `10.1016/j.bmcl.2020.127669` 解释为隐含 `10^-6 cm/s`、排除该未解决来源。后两者是测试后敏感性分析，不能用于改变模型选择。输出目录为 `results/final/frozen_test_diagnostics_v1`，包括英文 600 dpi PNG、分子级明细、CSV 和分析报告。

## 15. 冻结候选推理与全局解释

面向新结构的推理必须使用冻结候选登记，而非任意单模型。输入 CSV 需要 `smiles` 或 `Canonical_SMILES` 列；输出为长表，每个输入结构对应每个登记端点一行：

```bash
python scripts/infer_frozen_candidates.py \
  --input new_molecules.csv \
  --output results/new_frozen_predictions.csv
```

结果包含物理/变换空间预测、规范单位、证据状态、最近训练分子的 ECFP4 Tanimoto 和适用域标志。`outside` 表示相似度低于预先固定的 0.30 阈值，预测仍会返回，但不应脱离该警告解读。该入口只加载候选登记的模型；fu 自动使用三个 Shared MLP 的物理尺度平均。

冻结 STL RF 的全局分裂重要性可重新生成，且不读取测试标签：

```bash
python scripts/report_frozen_interpretability.py
```

输出 `results/final/frozen_model_interpretability_v1`。重要性是描述性而非因果归因；相关描述符和 ECFP 位会相互分摊重要性。fu 的冻结模型是 Shared MLP 集成，故不会被误报为 RF 特征重要性。

## 16. 人源 F、CLint 与 Thalf 的证据扩充队列

下列程序将审计中保留的 review 记录组织为按 DOI、文献和 assay 聚合的人工核查队列。它不会更改数据、划分或模型：

```bash
python scripts/audit_human_evidence_expansion.py
```

输出 `results/analysis/human_evidence_expansion_v1`。只有在原始来源明确证明绝对口服 F 的 IV 参照、终末 IV 半衰期，或以 mg 蛋白为归一化基准的微粒体 CLint 后，才可通过版本化规则将记录写入新的数据集。

## 17. Thalf 已接受来源专项审计

人工 `rejected` 裁定会把精确匹配的 activity 标记为 `excluded`，清空规范标签，并保留原始值和裁定来源供追溯。`accepted` 既可提升保守 review，也可把已有 automatic accepted 转为原文确认。接受或拒绝都必须填写原文页码/表格和证据说明。

以下程序只审计 train/validation 中已接受的人源 terminal-IV Thalf 来源，不报告测试记录或测试指标：

```bash
python scripts/audit_thalf_accepted_sources.py --check-only \
  --audit-dir data/processed_v11/audit \
  --output results/analysis/thalf_accepted_sources_v11_r2

python scripts/audit_thalf_accepted_sources.py \
  --audit-dir data/processed_v11/audit \
  --output results/analysis/thalf_accepted_sources_v11_r2
```

输出记录级 `automatic_accepted_records.csv`、文献级 `automatic_accepted_documents.csv` 和人工/自动来源汇总 `summary.csv`。风险标志仅用于安排原文审核顺序，不能自动生成拒绝裁定。
