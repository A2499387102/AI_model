#!/bin/bash
# ml_tool 环境创建脚本
# 用法：bash setup_env.sh [环境名称]
# 默认环境名：ml_tool_env

ENV_NAME=${1:-ml_tool_env}
CONDA_EXE=${CONDA_EXE:-conda}

echo "======================================"
echo " ml_tool 环境创建脚本"
echo " 环境名称: $ENV_NAME"
echo "======================================"

# 检查 conda 是否可用
if ! command -v "$CONDA_EXE" &> /dev/null; then
    # Windows 常见路径兜底
    for candidate in \
        "/c/miniconda/Scripts/conda.exe" \
        "/c/ProgramData/miniconda3/Scripts/conda.exe" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda"; do
        if [ -f "$candidate" ]; then
            CONDA_EXE="$candidate"
            break
        fi
    done
fi

if [ ! -f "$CONDA_EXE" ] && ! command -v conda &> /dev/null; then
    echo "[ERROR] 未找到 conda，请确认 Miniconda/Anaconda 已安装并配置到 PATH"
    exit 1
fi

echo "[1/4] 创建 Python 3.9 虚拟环境..."
"$CONDA_EXE" create -n "$ENV_NAME" python=3.9 -y
if [ $? -ne 0 ]; then
    echo "[ERROR] 环境创建失败"
    exit 1
fi

# 获取该环境的 Python 路径
PYTHON_BIN=$("$CONDA_EXE" run -n "$ENV_NAME" python -c "import sys; print(sys.executable)" 2>/dev/null)
echo "    Python 路径: $PYTHON_BIN"

echo "[2/4] 安装依赖包..."
"$CONDA_EXE" run -n "$ENV_NAME" pip install \
    pandas==2.0.3 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    lightgbm==4.4.0 \
    xgboost==2.1.0 \
    hyperopt==0.2.7 \
    scipy==1.13.1 \
    openpyxl==3.1.2 \
    jupyterlab==4.2.5 \
    notebook==7.2.2 \
    ipykernel==6.29.5 \
    --no-cache-dir

if [ $? -ne 0 ]; then
    echo "[ERROR] 依赖安装失败"
    exit 1
fi

echo "[3/4] 验证安装..."
"$CONDA_EXE" run -n "$ENV_NAME" python -c "
import pandas, numpy, sklearn, lightgbm, xgboost, hyperopt, scipy, openpyxl
import jupyter_core, notebook, jupyterlab
print('  pandas      :', pandas.__version__)
print('  numpy       :', numpy.__version__)
print('  scikit-learn:', sklearn.__version__)
print('  lightgbm    :', lightgbm.__version__)
print('  xgboost     :', xgboost.__version__)
print('  hyperopt    :', hyperopt.__version__)
print('  scipy       :', scipy.__version__)
print('  openpyxl    :', openpyxl.__version__)
print('  jupyter     :', jupyter_core.__version__)
print('  notebook    :', notebook.__version__)
print('  jupyterlab  :', jupyterlab.__version__)
print('  所有依赖验证通过')
"

if [ $? -ne 0 ]; then
    echo "[ERROR] 依赖验证失败"
    exit 1
fi

echo "[4/5] 注册 Jupyter kernel..."
"$CONDA_EXE" run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
if [ $? -ne 0 ]; then
    echo "[ERROR] Jupyter kernel 注册失败"
    exit 1
fi

echo "[5/5] 测试 ml_tool 包导入..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$CONDA_EXE" run -n "$ENV_NAME" python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from ml_tool import FeatureAnalyzer, Binning, FeatureSelector, ModelTrainer, ReportGenerator
print('  ml_tool 所有模块导入成功')
"

if [ $? -ne 0 ]; then
    echo "[ERROR] ml_tool 包导入失败，请检查代码路径"
    exit 1
fi

echo ""
echo "======================================"
echo " 环境创建完成！"
echo ""
echo " 激活环境："
echo "   conda activate $ENV_NAME"
echo ""
echo " 启动 Jupyter："
echo "   conda run -n $ENV_NAME jupyter notebook"
echo "   或激活环境后：jupyter notebook"
echo "======================================"
