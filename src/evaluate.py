import time
import numpy as np
from typing import Any

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """精度を計算する関数

    Parameters
    ----------
    y_true : np.ndarray
        正解ラベル
    y_pred : np.ndarray
        予測ラベル

    Returns
    -------
    float
        精度
    """
    return (y_true == y_pred).mean()

def measure_runtime(func: callable, **func_kwargs: dict) -> Any:
    """与えられた関数の実行時間を計測する関数

    Parameters
    ----------
    func : callable
        計測対象の関数
    **func_kwargs : dict
        関数に渡すキーワード引数
    
    Returns
    -------
    Any
        関数の戻り値
    """
    start_time = time.perf_counter()
    ret = func(**func_kwargs)
    end_time = time.perf_counter()
    print(f"Runtime: {end_time - start_time:.6f} seconds")
    return ret