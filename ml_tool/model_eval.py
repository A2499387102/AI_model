import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score, roc_curve


def _ks_auc(y_true, y_score, weights=None):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=weights)
        auc = roc_auc_score(y_true, y_score, sample_weight=weights)
        if auc < 0.5:
            auc = 1 - auc
        ks = round(float(np.abs(tpr - fpr).max()), 4)
        return ks, round(float(auc), 4)
    except Exception:
        return None, None


def evaluate_by_group(
    df: pd.DataFrame,
    group_col: str,
    y_col: str,
    score_col: str,
    weight_col: Optional[str] = None,
    model_type: str = "clf",
) -> pd.DataFrame:
    """
    按 group_col 分组计算 KS / AUC（分类）或 RMSE / MAE（回归）。

    参数
    ----
    df         : 包含标签、预测分数、分组列的 DataFrame
    group_col  : 分组列名，如 'dataset'、'month'、'ym'
    y_col      : 真实标签列名
    score_col  : 预测分数/预测值列名
    weight_col : 样本权重列名，None 表示等权
    model_type : 'clf'（分类，输出 KS/AUC）或 'reg'（回归，输出 RMSE/MAE）

    返回
    ----
    DataFrame，每行对应一个分组的评估指标
    """
    rows = []
    for grp_val, sub in df.groupby(group_col, sort=True):
        y = sub[y_col].values
        s = sub[score_col].values
        w = sub[weight_col].values if weight_col else None
        n = len(sub)

        if model_type == "clf":
            n_pos = int(y.sum())
            n_neg = n - n_pos
            if n_pos == 0 or n_neg == 0:
                ks, auc = None, None
            else:
                ks, auc = _ks_auc(y, s, w)
            rows.append({
                group_col: grp_val,
                "样本数": n,
                "正样本数": n_pos,
                "正样本率": round(n_pos / n, 4),
                "KS": ks,
                "AUC": auc,
            })
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            rmse = round(float(np.sqrt(mean_squared_error(y, s))), 4)
            mae  = round(float(mean_absolute_error(y, s)), 4)
            rows.append({
                group_col: grp_val,
                "样本数": n,
                "RMSE": rmse,
                "MAE": mae,
            })

    return pd.DataFrame(rows)


def evaluate_clf_by_group(
    df: pd.DataFrame,
    group_col: str,
    y_col: str,
    score_col: str,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """按分组输出分类指标（KS/AUC）的快捷入口"""
    return evaluate_by_group(df, group_col, y_col, score_col, weight_col, model_type="clf")


def evaluate_reg_by_group(
    df: pd.DataFrame,
    group_col: str,
    y_col: str,
    score_col: str,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """按分组输出回归指标（RMSE/MAE）的快捷入口"""
    return evaluate_by_group(df, group_col, y_col, score_col, weight_col, model_type="reg")
