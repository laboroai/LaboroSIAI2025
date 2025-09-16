# SIAI2025・2hours企業体験コード

2025/09/12-13に実施した，SIAI #7　産学クロススクエア　「ミライをつくるAI人材」の2hours企業体験用のコードです．

## フォルダ構成

```txt
.
├── dataset                 :データセットディレクトリ
│   ├── test                      :テスト用画像ディレクトリ
│   ├── train                     :学習用画像ディレクトリ
│   ├── test-images-idx3-ubyte      :テスト用画像ファイル
│   ├── test-labels-idx1-ubyte      :テスト用ラベルファイル
│   ├── train-images-idx3-ubyte     :学習用画像ファイル
│   └── train-labels-idx1-ubyte     :学習用ラベルファイル
├── notebooks               :ノートブックディレクトリ
├── src                     :共通関数ディレクトリ
├── weights                 :重みファイルディレクトリ
│   └── logistic_dict.pkl     :Ph1のロジスティック回帰学習済みモデルファイル
├── README.md       :本ファイル
├── byte2png.py     :MNISTの画像化スクリプト
└── pyproject.toml  :モジュール関連
```

## インストール手順

### `uv`のインストール

以下のコマンドで`uv`をインストールしてください．

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 仮想環境構築及びモジュールのインストール

以下のコマンドでご自身の環境に合わせてモジュールをインストールしてください．

```bash
uv pip install ".[gpu]"   # Linux環境＋GPUの場合
uv pip install ".[metal]" # MacでM1以降の場合
uv pip install ".[cpu]"   # CPU環境の場合（※）
```

※CPU環境の場合，`2*_answer_*.ipynb`は実行できない（適宜`gpu`→`cpu`処理を行えば実行可能）点にご注意ください．

## 実行

### MNISTデータの展開

初回限定で以下のコマンドでMNISTデータを`dataset`フォルダ以下の`dataset/train`,`dataset/test`フォルダに画像とラベル情報を展開します．

```bash
uv run byte2png.py
```

### コードの実行

`notebooks`フォルダの各`*.ipynb`を実行してください．詳細は[notebooks/README.md](./notebooks/README.md)を参照してください．
