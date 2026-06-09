import pandas as pd
import numpy as np
from typing import Union, Optional


def split_dataset(
    df: pd.DataFrame,
    date_col: str,
    oot_date: Union[str, pd.Timestamp],
    train_ratio: float = 0.8,
    dataset_col: str = "dataset",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    按日期列划分 train / test / oot，结果写入 dataset_col 列。

    划分逻辑
    --------
    1. date_col < oot_date  → train_pool（历史数据）
       date_col >= oot_date → oot（样本外验证集，保持时间顺序不做随机切分）
    2. train_pool 按 train_ratio : (1-train_ratio) 随机划分为 train / test
       默认 8:2

    参数
    ----
    df           : 原始 DataFrame
    date_col     : 日期列名，支持字符串 / datetime 类型；若为字符串会自动转换
    oot_date     : OOT 切点，等于或晚于此日期的样本进入 OOT
                   可传字符串（如 '2024-03-01'）或 pd.Timestamp
    train_ratio  : train 占 train_pool 的比例，默认 0.8
    dataset_col  : 写入结果的列名，默认 'dataset'
    random_state : 随机种子，保证可复现

    返回
    ----
    带 dataset_col 列的 DataFrame（原始行顺序不变，index 不重置）

    示例
    ----
    df = split_dataset(df, date_col='report_date', oot_date='2024-03-01')
    print(df['dataset'].value_counts())
    """
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' 不在 DataFrame 中")
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio 须在 (0, 1) 之间，当前值: {train_ratio}")

    df = df.copy()

    # 日期列统一转为 datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    oot_date = pd.Timestamp(oot_date)

    # 按切点分组
    mask_oot = df[date_col] >= oot_date
    idx_oot        = df.index[mask_oot]
    idx_train_pool = df.index[~mask_oot]

    if len(idx_train_pool) == 0:
        raise ValueError(
            f"oot_date '{oot_date}' 早于或等于所有样本日期，train_pool 为空，"
            "请检查 oot_date 是否设置正确。"
        )
    if len(idx_oot) == 0:
        raise ValueError(
            f"oot_date '{oot_date}' 晚于所有样本日期，OOT 为空，"
            "请检查 oot_date 是否设置正确。"
        )

    # train_pool 随机切分 train / test
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(idx_train_pool)
    n_train = int(len(shuffled) * train_ratio)
    idx_train = shuffled[:n_train]
    idx_test  = shuffled[n_train:]

    # 写入 dataset 列
    df[dataset_col] = None
    df.loc[idx_train, dataset_col] = "train"
    df.loc[idx_test,  dataset_col] = "test"
    df.loc[idx_oot,   dataset_col] = "oot"

    # 打印分布摘要
    counts = df[dataset_col].value_counts().reindex(["train", "test", "oot"])
    total  = len(df)
    print(f"数据集划分完成（切点: {oot_date.date()}, train_ratio={train_ratio}）")
    for ds, cnt in counts.items():
        print(f"  {ds:5s}: {cnt:6d}  ({cnt/total:.1%})")

    return df
