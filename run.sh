#!/bin/bash

VENV_PATH="venv310"

# 默认参数
JSON_FILE="LargeModel/data/extend_1039.json"
JSON_DIR="LargeModel/data/extend_1039"

# 定义颜色（ANSI 转义序列）
GREEN='\033[0;32m'  # 绿色
YELLOW='\033[0;33m' # 黄色
RESET='\033[0m'     # 重置颜色

# 用户输入参数
RUN_ROUNG=$1
MODEL=$2
MODE=$3

PYTHON_ARG="--$MODE"

# 检查参数是否完整
if [ -z "$RUN_ROUNG" ] || [ -z "$MODEL" ] || [ -z "$MODE" ]; then
    echo "Usage: sh run.sh RUN_RANK MODEL MODE"
    echo "MODE: Cni, Ano, Cno, Nfc, Full"
    exit 1
fi

# 初始化计数器
count_seq=0
bar_length=100       # 进度条长度（字符数）

projects=$(jq -r 'keys[]' "$JSON_FILE")
# 提取所有项目名称，并逐行处理
#jq -r 'keys[]' "$JSON_FILE" | while read -r project; do
for project in $projects; do
    # 去除多余空格和换行
    project=$(echo "$project" | xargs)
    commit_seq=0
    # 调试：打印当前项目名
    echo "[INFO]: Processing project '$project'"

    # 提取提交列表
    commits=$(jq -r --arg project "$project" '
      .[$project] | if type == "object" then keys[] else empty end
    ' "$JSON_FILE")
    total_commit=$(echo "$commits" | wc -l)

    # 打印提交列表
    if [ -z "$commits" ]; then
        echo "No commits found for project: $project"
    else
        for commit in $commits; do
          percent=$((commit_seq * 100 / total_commit))
          # 动态生成进度条
          completed=$((commit_seq * bar_length / total_commit))  # 已完成部分的字符数
          remaining=$((bar_length - completed))  # 剩余部分的字符数
          # 构建进度条
          bar=$(printf "%${completed}s" "" | tr ' ' '#')   # 已完成部分用 '#'
          bar+=$(printf "%${remaining}s" "" | tr ' ' '-')  # 剩余部分用 '-'
          # 打印进度条和百分比
          printf "\r${GREEN}Progress: [${YELLOW}%s${GREEN}] ${RESET}%d%% \r" "$bar" "$percent"
          ((commit_seq++))
          ((count_seq++))
          commit=$(echo "$commit" | xargs)
          echo ""
          echo "[INFO]: Processing the $count_seq commit"
          echo "[INFO]: Processing commit '$commit'"
          echo "[INFO]: It is the $commit_seq commit of $project"
          $VENV_PATH/Scripts/python JIT_LSM.py\
              $PYTHON_ARG\
              --test_data_file $JSON_DIR\
              --round $RUN_ROUNG \
              --model_name $MODEL \
              --project $project \
              --commit $commit \
              --seq $commit_seq \
              --output_file "results/large/RQ3/$MODE"
        done
    fi
done