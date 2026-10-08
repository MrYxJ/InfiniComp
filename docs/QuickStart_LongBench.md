# InfiniComp × LongBench 评测运行说明

> 适用代码：`src/`（InfiniComp 主代码）  
> 评测脚本：`src/metric/metric_LongBench.py`  
> 数据集：`metrics/LongBench/`

---

## 一、目录结构要求（重要）

`metric_LongBench.py` 里的路径是**按固定相对层级推算的**，不能随意挪动。收到代码后请保持如下结构：

```
InfiniComp/                 ← 项目根目录（脚本靠它定位一切）
├── src/                         ← 我给你的代码
│   ├── pipline.py
│   ├── retrieval.py
│   ├── chunks.py
│   ├── thoughts.py
│   ├── text_split.py
│   ├── requirements.txt         ← 依赖清单
│   ├── run_longbench.sh         ← 一键评测脚本
│   ├── LongBench评测运行说明.md  ← 本文档
│   └── metric/
│       ├── metric_LongBench.py  ← 评测入口
│       └── define_benchmark_prompt.py
├── metrics/
│   └── LongBench/               ← 数据集必须在这里
│       ├── data/                ← 16 个 *.jsonl 任务文件
│       │   ├── narrativeqa.jsonl
│       │   ├── hotpotqa.jsonl
│       │   └── ...（共 16 个）
│       ├── config/
│       │   ├── dataset2prompt.json
│       │   └── dataset2maxlen.json
│       ├── pred/                ← 预测输出（自动创建）
│       └── eval.py              ← 官方打分脚本（自动调用）
├── experiences/                 ← 过程记录（自动创建）
└── logs/                        ← 日志（自动创建）
```

脚本顶部的路径推导逻辑：

```python
DirPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# → 从 .../InfiniComp/src/metric/metric_LongBench.py 向上三级 = .../InfiniComp
LongBenchDirPath = os.path.join(DirPath, "metrics", "LongBench")
```

**注意：`metrics` 位于 `src` 的上一级（`..\metrics\LongBench`），不是 `src` 内部。** 如果对方把 `metrics/LongBench` 放错位置，脚本会在 `load_dataset` 时报文件找不到。

数据集检查清单（`metrics/LongBench/data/` 下应有这些文件）：

```
narrativeqa  qasper  multifieldqa_en  hotpotqa  2wikimqa  musique
gov_report   qmsum   multi_news       trec       triviaqa  samsum
passage_count  passage_retrieval_en  lcc  repobench-p
```

缺哪个任务，就别用 `--task_name all`，改成单独指定已有任务。

---

## 二、环境安装

### 1. Python 版本

建议 **Python 3.10 – 3.12**（代码在 3.10+ 上验证过）。

### 2. 安装依赖

在项目根目录 `InfiniComp/` 下执行（依赖清单已随 `src/` 一起提供）：

```bash
pip install -r src/requirements.txt
```

如果根目录也有 `requirements.txt`，两者任一即可（内容等价）。

各包用途：

| 包                                   | 用途                                             |
| ----------------------------------- | ---------------------------------------------- |
| `torch` `transformers` `accelerate` | 模型加载与推理                                        |
| `datasets`                          | 读取 LongBench 的 jsonl 数据                        |
| `loguru`                            | 日志（**必需**，缺了直接报 ImportError）                   |
| `matplotlib` `seaborn` `scipy`      | `thoughts.py` 的注意力热图与 1D 卷积打分                  |
| `numpy`                             | 注意力矩阵数值计算                                      |
| `tiktoken`                          | `chunks.py` 的 token 计数分支                       |
| `rouge` `jieba` `fuzzywuzzy`        | `metrics/LongBench/eval.py` 打分（中文 ROUGE、代码相似度） |

### 3. GPU 要求

- 推理使用 `device_map="auto"`，**单卡即可**（0.5B/1.5B 模型 4GB 显存足够）
- 模型加载仍用 `attn_implementation="eager"`（脚本已写死），**不能用 FlashAttention**——它不返回注意力权重，而 InfiniComp 的检索完全依赖注意力矩阵。
- 默认 `exact_eager=true`（精确路径）：全部层 eager 前向，hook 在目标层**内部立即做 head 求和**（与打分端完全相同的 torch op），**与优化前原始实现逐位一致**（已在 LongBench 16 任务×10 样本上 160/160 验证），且驻留从 `heads·N²` 降到 `N²`。**论文复现/出正式数据用它。**
- `exact_eager=false`（快速路径）：非检索层用 SDPA（完全不物化 N² 矩阵），仅检索目标层按 eager 公式分组算概率。省掉 eager 每层的 N² 瞬时显存——**7B 模型或超长序列（>8K）显存受限时必须用它**。代价：SDPA 与 eager 内核的固有数值漂移会改变逐样本检索结果（实测 160 样本检索一致率 ~17%，但任务均分无系统性偏移），已实测 fast 路径内部异常时会自动回退精确路径并打印 warning。

---

## 三、准备模型

脚本默认模型路径是作者本机路径，**对方必须改成自己的路径**。两种方式：

**方式 A：命令行指定（推荐）**

```bash
python src/metric/metric_LongBench.py \
    --model_path /your/path/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B
```

**方式 B：改脚本默认值**

编辑 `src/metric/metric_LongBench.py` 第 22–23 行：

```python
parser.add_argument("--model_name", type=str, default="Qwen2.5-0.5B")
parser.add_argument("--model_path", type=str, default="D:\Code\models\Qwen2.5-1.5B-Instruct")
```

> `--model_path` 是模型权重的实际磁盘路径（HuggingFace 格式目录或模型 ID）；  
> `--model_name` 只是**输出目录和日志的标签名**，不参与加载，但会写进预测目录名，建议和模型保持一致。

---

## 四、运行评测

### 4.0 一键脚本（推荐）

`src/run_longbench.sh` 把「依赖安装 → 路径校验 → 数据检查 → 运行评测 → 结果汇总 → 打印分数」串成了一条命令。

**第一步：改配置。** 打开 `src/run_longbench.sh`，只改开头【配置区】的这几项：

```bash
MODEL_PATH="D:/Code/models/Qwen2.5-0.5B-Instruct"   # 【必改】改成你的模型路径
MODEL_NAME="Qwen2.5-0.5B"                            # 输出目录/日志标签
TASK_NAME="all"                                      # all 或单个任务名
CHUNK_SIZE=512
COMPRESS_BUDGET=300
EXACT_EAGER=true             # 注意力路径开关，见下方参数说明
```

**第二步：运行。**

```bash
cd InfiniComp
bash src/run_longbench.sh
```

**常用选项：**

| 命令                                          | 作用                               |
| ------------------------------------------- | -------------------------------- |
| `bash src/run_longbench.sh`                 | 完整流程（装依赖 + 全量评测 + 打分）            |
| `bash src/run_longbench.sh --skip-install`  | 跳过依赖安装，直接评测                      |
| `bash src/run_longbench.sh --task hotpotqa` | 只跑单个任务（覆盖配置区的 `TASK_NAME`）       |
| `bash src/run_longbench.sh --quick`         | **冒烟测试**：单任务 + 只跑前 3 条样本，几分钟验证流程 |
| `bash src/run_longbench.sh --help`          | 查看帮助                             |

> **推荐流程**：先 `--quick` 跑通（确认模型路径、显存、依赖都没问题），再直接  
> `bash src/run_longbench.sh` 出全量结果。
>
> `--quick` 会临时给脚本打上 3 条样本的断点，**运行结束后自动还原**，不会污染代码。

脚本自带的前置校验（任一不通过会立即报错并给出修复提示）：

1. `src/metric/metric_LongBench.py` 是否存在
2. `metrics/LongBench/data/` 是否存在、各任务 `*.jsonl` 是否齐全（缺的任务自动跳过并告警）
3. `config/dataset2prompt.json`、`config/dataset2maxlen.json` 是否存在
4. 模型路径是否存在（若不像本地路径则按 HF 模型 ID 处理）
5. 关键依赖能否正常 import（一次性列出所有缺失包）

### 4.1 手动运行（不用脚本时）

**单任务验证：**

```bash
cd InfiniComp
python src/metric/metric_LongBench.py \
    --model_path /your/path/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B \
    --task_name narrativeqa \
    --chunk_size 512 \
    --compress_budget 300
```

**全量评测（16 个任务）：**

```bash
cd InfiniComp
python src/metric/metric_LongBench.py \
    --model_path /your/path/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B \
    --task_name all \
    --chunk_size 512 \
    --compress_budget 300 \
    --phrase_token_num 15 \
    --iter_number 1 \
    --max_new_tokens 128
```

想快速验证流程不跑全量时，可临时打开脚本第 115 行的调试断点：

```python
# if index >= 3: break    ← 取消注释后每个任务只跑前 3 条样本
```

> 该断点**默认已注释，即默认跑全量样本**。若你为了调试重新打开，正式出结果前  
> 务必再注释回去，否则指标不可用。

`--task_name all` 会依次跑这 16 个任务：

```
narrativeqa  qasper  multifieldqa_en  hotpotqa  2wikimqa  musique
gov_report   qmsum   multi_news       trec       triviaqa  samsum
passage_count  passage_retrieval_en  lcc  repobench-p
```

### 4.3 单任务评测

把 `--task_name` 换成任意一个任务名即可，例如：

```bash
python src/metric/metric_LongBench.py \
    --model_path /your/path/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B \
    --task_name hotpotqa \
    --chunk_size 512 --compress_budget 300
```

---

## 五、参数说明

| 参数                    | 默认值                                | 含义                                                                           |
| --------------------- | ---------------------------------- | ---------------------------------------------------------------------------- |
| `--model_path`        | 作者本机路径                             | **必改**：模型权重路径或 HF 模型 ID                                                      |
| `--model_name`        | `Qwen2.5-0.5B`                     | 输出目录/日志标签，不参与加载                                                              |
| `--task_name`         | `all`                              | 任务名，`all` 表示全部 16 个                                                          |
| `--cache_granularity` | `sentence-level`                   | 检索粒度：`sentence-level`（句级，推荐）或 `token-level`                                  |
| `--chunk_size`        | `128`                              | 每个 chunk 的 token 数；**太小会拖慢速度**，常用 256/512                                    |
| `--topk`              | `None`（回退为 `--compress_budget` 的值） | 每轮选取的 top attention score index 数；粒度由 `--phrase_token_num` 决定，常用 100/300/500 |
| `--compress_budget`   | `512`                              | 压缩后 context 的 **token 数上限**（最终长度为动态、不超过该上限），常用 300/500                       |
| `--compress_ratio`    | `None`                             | 若指定（如 `0.05`），按原长比例计算上限，**优先于 `compress_budget`**                            |
| `--phrase_token_num`  | `15`                               | attention score 的粒度：`1` = 单 token；`>1` = phrase（连续 N 个 token 分数之和）           |
| `--iter_number`       | `1`                                | 检索迭代轮数；增大可提升多跳任务效果，但更慢                                                       |
| `--max_new_tokens`    | `128`                              | 生成答案的最大 token 数（脚本里实际会被 LongBench 的 `dataset2maxlen` 覆盖）                     |
| `--need_generate`     | `True`                             | 是否生成答案；`False` 时只做检索、不算分                                                     |
| `--exact_eager`       | `True`                             | 注意力计算路径开关，详见下方说明                                                             |
| `--disable_thinking`  | `True`                             | 思考模型输出开关，详见下方说明                                                              |

**`--disable_thinking` 思考模型开关（Qwen3 等 hybrid-thinking 模型）：**

| 取值         | 行为                                                                                                                                                                          | 适用场景                                                                                                                                                                               |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `True`（默认） | 关思考：chat 模板传 `enable_thinking=False`（Qwen3 官方模板自动插入空 think 块，即 NoThinking 模式），并在解码端屏蔽 think 标记 token（`<think>`/`</think>`，Qwen3 为 151667/151668）双保险；生成阶段对残留 think 块再做一次剥离兜底 | 评测 LongBenchV1 等**固定小输出预算**基准（dataset2maxlen 多为 32-128 token）：思考内容会耗尽预算导致答案未产出即被截断。做法与 CompKV（arXiv 2609.26300）一致："direct-answer setting for comparing under fixed output budgets" |
| `False`    | 保持模型默认思考模式                                                                                                                                                                  | 需要展示 thinking 能力的另设实验（须自行加大 max_new_tokens 并评后剥离 think，勿与 V1 主表混用）                                                                                                                 |

对 Qwen2.5 等无思考能力的模型该开关无任何影响（模板忽略 `enable_thinking` 变量、屏蔽列表自检为空、剥离正则无匹配）。

**`--exact_eager` 注意力路径开关（逐位复现 vs 显存/速度）：**

| 取值         | 路径                    | 行为                                                                                                        | 适用场景                 |
| ---------- | --------------------- | --------------------------------------------------------------------------------------------------------- | -------------------- |
| `True`（默认） | exact（eager+hook 预求和） | 全部层 eager 前向，hook 捕获目标层后**立即**做 head 求和（与打分端同一 torch op），**与原实现逐位一致**（160/160 样本验证），驻留 `N²` 而非 `heads·N²` | 论文复现、出正式数据           |
| `False`    | fast（SDPA）            | 非检索层用 SDPA 前向（不物化 N² 矩阵，省掉 eager 每层的 N² 瞬时显存），仅检索目标层按 eager 公式分组算概率并 head 求和                              | 7B 模型、超长序列（>8K）显存受限时 |

两路径的逐样本差异来源是 SDPA 与 eager 内核的固有数值漂移（实测：概率重算公式在相同 eager 状态下误差为 **0（逐位一致）**；漂移全部来自前 27 层隐状态，逐层累积 absmax 3→87、rel-L2 ~2%→11%）。漂移使逐样本检索一致率降至 ~17%、答案一致率 28-45%，但任务级均分无系统性偏移（16 任务均分差 <±1.3）；fast 路径内部异常时自动回退 exact 路径并打印 warning。

**参数联动提醒**：

- 压缩逻辑：按 attention score 选取 **topk 个 index**（`phrase_token_num=1` 时为单 token 粒度，`>1` 时为 phrase 粒度），再取**包含这些 index 的完整句子**组成压缩文本，因此最终长度是**动态的**；`compress_budget` 只作为 token 数上限截断（按分数顺序累加句子、达到上限即停），不设则为纯 topk 动态长度。
- `compress_budget` 和 `compress_ratio` 二者互斥，指定了 `compress_ratio` 就按比例算上限，`compress_budget` 失效。
- 未显式设置 `--topk` 时自动回退用 `compress_budget` 的值（与旧版"compress_budget 兼任 top-k"的行为尺度一致），因此旧命令无需修改即可运行。

---

## 六、输出与结果查看

### 6.1 输出位置

**预测结果**：

```
metrics/LongBench/pred/{model_name}_{cache_granularity}_{phrase_token_num}_topk{topk}_{预算}_{chunk_size}/{task_name}.jsonl
```

举例，`--model_name Qwen2.5-0.5B --cache_granularity sentence-level --phrase_token_num 15 --compress_budget 300 --chunk_size 512`（topk 未设、回退为 300）会生成：

```
metrics/LongBench/pred/Qwen2.5-0.5B_sentence-level_15_topk300_300_512/narrativeqa.jsonl
metrics/LongBench/pred/Qwen2.5-0.5B_sentence-level_15_topk300_300_512/hotpotqa.jsonl
...
```

> 每行是一个样本的 JSON，包含 `pred`（模型答案）、`answers`（标准答案）、  
> `origin_length`（原始 token 数）、`input_length`（压缩后 token 数）、  
> `answer_in_context` / `answer_in_retrieval`（答案是否在原文/检索结果中）等字段。

**过程日志**：

```
logs/metric_LongBench_{model_name}.log       # 主日志（含 tqdm 进度）
experiences/process/{model_name}/            # 每个样本的检索过程记录
excute_logs/                                 # 时间戳命名的执行日志
```

### 6.2 自动打分

脚本末尾会自动调用 LongBench 官方打分：

```python
os.system("python %s --model %s" % (python_file, out_dir))
# 实际执行：python metrics/LongBench/eval.py --model <上面那个目录名>
```

分数会打印在终端，同时写入 `logs/metric_LongBench_{model_name}.log`。

### 6.3 手动重新打分

只想重跑打分（不重新推理）时：

```bash
cd metrics/LongBench
python eval.py --model Qwen2.5-0.5B_sentence-level_15_300_512
```

`--model` 的值就是 `pred/` 下的目录名。

### 6.4 指标含义

| 任务类型                                                                       | 指标 |
| -------------------------------------------------------------------------- | -- |
| 问答类（narrativeqa/qasper/hotpotqa/2wikimqa/musique/multifieldqa_en/triviaqa） | F1 |


| 摘要类（gov_report/qmsum/multi_news/samsum） | ROUGE-L |  
| 分类类（trec） | Accuracy |  
| 检索类（passage_retrieval_en/passage_count） | 精确匹配 / 计数准确率 |  
| 代码类（lcc/repobench-p） | 编辑相似度 |

---

## 七、常见问题

**Q1：`FileNotFoundError: .../metrics/LongBench/data/xxx.jsonl`**

目录层级不对。确认 `metrics/` 与 `src/` **同级**，且都在项目根目录下。  
`metric_LongBench.py` 靠 `src/metric/` 向上三级定位根目录，不要移动脚本位置。

**Q2：`ImportError: No module named 'loguru'` / `matplotlib` / `seaborn`**

依赖没装全。`thoughts.py` 在模块顶层就 import 了 matplotlib / seaborn / scipy，  
缺任何一个都会导致 import 失败。

**Q3：`Index verify failed` / `Searching Context Index ... Error!`**

prompt 模板与数据字段不匹配。脚本通过 `define_benchmark_prompt.define_longbench`  
包装 `build_prompt`，要求模板用 `{context}` 占位、数据里有 `input` 字段。  
检查是否用错了 `define_benchmark_prompt` 的版本（LongBench 用 `define_longbench`，  
不是 `define_infinitebench`）。

**Q4：跑得特别慢**

- `--chunk_size` 太小（如 128）会让 chunk 数量暴涨，每轮都要跑一次前向。建议 512。
- 16 个任务串行跑，全程可能数小时，属正常。先用单任务 + 去掉 3 条断点验证。
- 关掉热图记录（默认已关：`record_attention_path=None`），开热图会显著变慢。

**Q5：指标明显偏低**

确认脚本第 115 行那个 `if index >= 3: break` 处于**注释状态**——这是最容易踩的坑，  
只跑 3 条样本算出来的分数没有意义。（默认已注释；用 `run_longbench.sh --quick` 时  
脚本会自动打补丁并在结束时还原。）

**Q6：显存不足**

换更小模型，或调小 `--chunk_size`（单次前向的序列更短）。

---

## 八、快速上手（三分钟版）

```bash
# 1. 进入项目根目录
cd InfiniComp

# 2. 确认数据集在 metrics/LongBench/data/ 下

# 3. 改 src/run_longbench.sh 开头的 MODEL_PATH 为你自己的模型路径

# 4. 冒烟测试（单任务 + 3 条样本，几分钟出结果）
bash src/run_longbench.sh --quick

# 5. 跑通后出全量结果（16 个任务）
bash src/run_longbench.sh

# 6. 看分数（脚本结尾也会自动打印日志末尾）
cat logs/metric_LongBench_Qwen2.5-0.5B.log | tail -30
```

不想用脚本时，手动三行：

```bash
cd InfiniComp
pip install -r src/requirements.txt
python src/metric/metric_LongBench.py \
    --model_path /your/path/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B \
    --task_name all --chunk_size 512 --compress_budget 300
```
