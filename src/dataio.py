import os
import tqdm
import pickle
import pandas as pd
import numpy as np
import cv2

def read_data(dirpath: str) -> tuple[pd.DataFrame, np.ndarray]:
    """データを読み込む関数

    Parameters
    ----------
    dirpath : str
        データが格納されているディレクトリのパス

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        データフレームとラベルの配列
    """
    
    df = pd.read_csv(os.path.join(dirpath, "labels.csv"))
    images = []
    for filename in tqdm.tqdm(df["filename"]):
        filepath = os.path.join(dirpath, filename)
        if os.path.exists(filepath):
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            images.append(image)
        else:
            print(f"Warning: {filepath} does not exist.")
    return df, np.array(images)

def read_logistic_weights(filepath: str) -> dict[str, np.ndarray]:
    """ロジスティック回帰の重みを読み込む関数

    Parameters
    ----------
    filepath : str
        重みファイルのパス

    Returns
    -------
    dict[str, np.ndarray]
        各クラスのロジスティック回帰の重みの配列
        'weights' : np.ndarray
        'intercept' : np.ndarray
        'mean' : np.ndarray
        'std' : np.ndarray
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weight file {filepath} does not exist.")

    with open(filepath, "rb") as f:
        weights = pickle.load(f)
    return weights