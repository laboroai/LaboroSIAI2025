import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_images(images: np.ndarray, df_labels: pd.DataFrame, ncols: int = 5, figsize: tuple[int, int] = (15, 15)) -> None:
    """画像をプロットする関数

    Parameters
    ----------
    images : np.ndarray
        プロットする画像のリスト
    ncols : int, optional
        列数, by default 5
    figsize : tuple[int, int], optional
        図のサイズ, by default (15, 15)
    """
    
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
            # ラベルを表示
            label = df_labels.iloc[i]["label"]
            ax.set_title(label)
        else:
            ax.remove()
    plt.tight_layout()
    plt.show()