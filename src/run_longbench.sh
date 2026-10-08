#!/usr/bin/env bash
# =============================================================================
# InfiniComp x LongBench 一键评测脚本
#
# 用法：
#   1. 修改下面【配置区】的 MODEL_PATH（必改）和 MODEL_NAME
#   2. bash src/run_longbench.sh              # 完整流程
#      bash src/run_longbench.sh --skip-install   # 跳过依赖安装
#      bash src/run_longbench.sh --task hotpotqa  # 只跑单个任务
#      bash src/run_longbench.sh --quick          # 单任务 + 只跑3条样本（冒烟测试）
#
# 前置条件：
#   - metrics/LongBench/data/ 下有待评测的 *.jsonl 数据集
#   - metrics/LongBench/ 与 src/ 同级（脚本会自动校验）
# =============================================================================

set -euo pipefail

# ============================ 配置区（按需修改） ============================

# 【必改】模型权重路径（HuggingFace 格式目录，或 HF 模型 ID）
MODEL_PATH="D:/Code/models/Qwen2.5-0.5B-Instruct"

# 【建议】输出目录/日志标签，建议与模型同名，不要含空格
MODEL_NAME="Qwen2.5-0.5B"

# 评测任务：all = 全部 16 个任务；也可写单个任务名（如 hotpotqa）
TASK_NAME="all"

# 检索参数
CHUNK_SIZE=512              # 每个 chunk 的 token 数（太小会显著变慢）
TOPK=100                     # 每轮选取的 top attention score index 数（粒度由 PHRASE_TOKEN_NUM 决定：1=token，>1=phrase）；留空则回退为 COMPRESS_BUDGET 的值
COMPRESS_BUDGET=300         # 压缩后 context 的 token 数上限（最终长度为动态、不超过该上限）
PHRASE_TOKEN_NUM=25         # attention score 的粒度：1 = 单 token，>1 = phrase（连续 token 之和）
ITER_NUMBER=1               # 检索迭代轮数（增大可提升多跳任务效果）
CACHE_GRANULARITY="sentence-level"   # sentence-level 或 token-level
EXACT_EAGER=true            # true（默认）=逐层 eager 精确复现（bitwise 一致，论文复现用）；false=SDPA 快速路径（有内核级漂移，7B/超长序列显存受限时用）
DISABLE_THINKING=true       # true（默认）=关闭 Qwen3 等思考模型的 <think> 输出（enable_thinking=False + 解码端屏蔽 think token），使直接作答适配 LongBench 小输出预算；对 Qwen2.5 等非思考模型无影响

# 是否安装依赖（首次运行建议 true；已装好可命令行加 --skip-install）
INSTALL_DEPS=true

# ===========================================================================

# --------------------------------- 解析参数 --------------------------------
SKIP_INSTALL=false
QUICK_MODE=false
CLI_TASK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install)
            SKIP_INSTALL=true; shift ;;
        --quick)
            QUICK_MODE=true; shift ;;
        --task)
            CLI_TASK="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "[错误] 未知参数：$1（用 --help 查看用法）" >&2
            exit 1 ;;
    esac
done

[[ -n "$CLI_TASK" ]] && TASK_NAME="$CLI_TASK"

# --------------------------------- 终端着色 --------------------------------
if [[ -t 1 ]]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; NC=''
fi

info()  { echo "${BLUE}[信息]${NC} $*"; }
ok()    { echo "${GREEN}[完成]${NC} $*"; }
warn()  { echo "${YELLOW}[警告]${NC} $*"; }
fail()  { echo "${RED}[错误]${NC} $*" >&2; exit 1; }
title() { echo; echo "${BOLD}==== $* ====${NC}"; }

# --------------------------------- 路径定位 --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../src
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                   # .../InfiniComp
ENTRY="$SCRIPT_DIR/metric/metric_LongBench.py"
LONGBENCH_DIR="$PROJECT_ROOT/metrics/LongBench"
DATA_DIR="$LONGBENCH_DIR/data"

echo "${BOLD}InfiniComp x LongBench 评测${NC}"
echo "  项目根目录 : $PROJECT_ROOT"
echo "  评测入口   : $ENTRY"
echo "  数据集     : $DATA_DIR"
echo "  模型路径   : $MODEL_PATH"
echo "  模型标签   : $MODEL_NAME"
echo "  任务       : $TASK_NAME"

# ------------------------------ 1. 前置检查 --------------------------------
title "1/5 前置检查"

[[ -f "$ENTRY" ]] || fail "找不到评测入口：$ENTRY"
[[ -d "$DATA_DIR" ]] || fail "找不到数据集目录：$DATA_DIR
       请确认 metrics/LongBench/ 与 src/ 同级（即 ..\\metrics\\LongBench）。"

# 选定任务列表
if [[ "$TASK_NAME" == "all" ]]; then
    TASKS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique
           gov_report qmsum multi_news trec triviaqa samsum
           passage_count passage_retrieval_en lcc repobench-p)
else
    TASKS=("$TASK_NAME")
fi

# 逐个校验任务数据文件
MISSING=()
for t in "${TASKS[@]}"; do
    [[ -f "$DATA_DIR/$t.jsonl" ]] || MISSING+=("$t")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "以下任务缺少数据文件，将被跳过：${MISSING[*]}"
    EXISTING=()
    for t in "${TASKS[@]}"; do
        [[ -f "$DATA_DIR/$t.jsonl" ]] && EXISTING+=("$t")
    done
    TASKS=("${EXISTING[@]}")
    [[ ${#TASKS[@]} -gt 0 ]] || fail "没有任何可用的任务数据，请检查 $DATA_DIR"
fi
info "待评测任务（${#TASKS[@]} 个）：${TASKS[*]}"

# 校验配置文件
for cfg in dataset2prompt.json dataset2maxlen.json; do
    [[ -f "$LONGBENCH_DIR/config/$cfg" ]] || fail "缺少配置文件：$LONGBENCH_DIR/config/$cfg"
done

# 选择 Python 解释器
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    fail "未找到 python / python3，请先安装 Python 3.10+ 或设置 PYTHON 环境变量"
fi
info "Python 解释器：$PY（$($PY --version 2>&1)）"

# 校验模型路径（本地目录或 HF 模型 ID 均可）
if [[ -d "$MODEL_PATH" ]]; then
    info "模型目录校验通过：$MODEL_PATH"
elif [[ "$MODEL_PATH" == */* && ! "$MODEL_PATH" =~ ^[A-Za-z]: && ! "$MODEL_PATH" =~ ^/ ]]; then
    warn "MODEL_PATH 不是本地目录，将作为 HuggingFace 模型 ID 尝试在线拉取：$MODEL_PATH"
else
    fail "模型路径不存在：$MODEL_PATH
       请修改脚本顶部【配置区】的 MODEL_PATH 为你自己的路径。"
fi
ok "前置检查全部通过"

# ------------------------------ 2. 安装依赖 --------------------------------
title "2/5 依赖安装"
if [[ "$SKIP_INSTALL" == true || "$INSTALL_DEPS" != true ]]; then
    info "已跳过依赖安装"
else
    REQ="$SCRIPT_DIR/requirements.txt"
    if [[ -f "$REQ" ]]; then
        info "安装依赖：$REQ"
        "$PY" -m pip install -r "$REQ" || warn "pip 安装返回非零，请检查上面的报错信息"
    else
        warn "未找到 src/requirements.txt，改用项目根目录的 requirements.txt"
        "$PY" -m pip install -r "$PROJECT_ROOT/requirements.txt" \
            || warn "pip 安装返回非零，请检查上面的报错信息"
    fi
    ok "依赖处理完毕"
fi

# 快速验证关键依赖能否 import
info "校验关键依赖..."
"$PY" - <<'PYEOF' || fail "关键依赖缺失，请重新执行 pip install -r src/requirements.txt"
import importlib.util
import sys
required = ["torch", "transformers", "datasets", "loguru",
            "numpy", "scipy", "seaborn", "matplotlib",
            "tiktoken", "tqdm", "jieba", "rouge"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("  缺失：", ", ".join(missing))
    sys.exit(1)
print("  关键依赖齐全：", ", ".join(required))
PYEOF
ok "依赖校验通过"

# ------------------------------ 3. 运行评测 --------------------------------
title "3/5 开始评测"
cd "$PROJECT_ROOT"

# --quick 模式：只跑单任务的前 3 条样本，用于快速验证流程
if [[ "$QUICK_MODE" == true ]]; then
    if [[ "$TASK_NAME" == "all" ]]; then
        warn "--quick 模式需要指定单个任务，已自动切换为 narrativeqa"
        TASK_NAME="narrativeqa"
    fi
    warn "quick 模式：将临时启用「只跑前 3 条样本」的调试断点"
    PATCHED_ENTRY="$ENTRY"
    cp "$ENTRY" "$ENTRY.runbak"
    "$PY" - "$ENTRY" <<'PYEOF'
import re, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
src = src.replace("                # if index >= 3: break",
                  "                if index >= 3: break")
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("  已启用 3 条样本断点")
PYEOF
    trap '[[ -f "$ENTRY.runbak" ]] && mv -f "$ENTRY.runbak" "$ENTRY"' EXIT
fi

CMD=("$PY" "$ENTRY"
     --model_path "$MODEL_PATH"
     --model_name "$MODEL_NAME"
     --task_name "$TASK_NAME"
     --chunk_size "$CHUNK_SIZE"
     --compress_budget "$COMPRESS_BUDGET"
     --phrase_token_num "$PHRASE_TOKEN_NUM"
     --iter_number "$ITER_NUMBER"
     --cache_granularity "$CACHE_GRANULARITY"
     --exact_eager "$EXACT_EAGER"
     --disable_thinking "$DISABLE_THINKING")

# TOPK 留空时回退为 COMPRESS_BUDGET（legacy 行为）
TOPK_RESOLVED="$COMPRESS_BUDGET"
if [[ -n "$TOPK" ]]; then
    CMD+=(--topk "$TOPK")
    TOPK_RESOLVED="$TOPK"
fi

info "执行命令："
echo "  ${CMD[*]}"
echo

START_TS=$(date +%s)
if "${CMD[@]}"; then
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))
    ok "评测执行完毕，耗时 $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒"
    RUN_FAILED=false
else
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))
    warn "评测进程返回非零退出码（耗时 $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒）"
    RUN_FAILED=true
fi

# ------------------------------ 4. 结果汇总 --------------------------------
title "4/5 结果位置"

OUT_DIR_NAME="${MODEL_NAME}_${CACHE_GRANULARITY}_${PHRASE_TOKEN_NUM}_topk${TOPK_RESOLVED}_${COMPRESS_BUDGET}_${CHUNK_SIZE}"
PRED_DIR="$LONGBENCH_DIR/pred/$OUT_DIR_NAME"
LOG_FILE="$PROJECT_ROOT/logs/metric_LongBench_${MODEL_NAME}.log"

if [[ -d "$PRED_DIR" ]]; then
    ok "预测结果：$PRED_DIR"
    ls -1 "$PRED_DIR" | sed 's/^/    /'
else
    warn "未找到预测目录：$PRED_DIR"
fi

if [[ -f "$LOG_FILE" ]]; then
    ok "运行日志：$LOG_FILE"
else
    warn "未找到日志文件：$LOG_FILE"
fi

# ------------------------------ 5. 打分结果 --------------------------------
title "5/5 打分结果"
if [[ -f "$LOG_FILE" ]]; then
    echo "${BOLD}--- 日志末尾（含各任务分数） ---${NC}"
    tail -n 40 "$LOG_FILE"
    echo
    info "提示：每个任务跑完会自动调用 metrics/LongBench/eval.py 打分，"
    info "      分数也会同步打印在上面这段日志里。"
else
    warn "日志文件不存在，无法展示分数"
fi

echo
if [[ "$RUN_FAILED" == true ]]; then
    warn "评测未正常结束，请检查上方输出定位问题"
    exit 1
fi
ok "全部流程结束"
