inari
======
Library for GNN Model using torch-geometric

inariとは
==========

GNNのモデルを比較するためのライブラリ

Getting Started
===============

uvをインストールします。
[uv](https://docs.astral.sh/uv/getting-started/installation/)のページを参考に、自分の環境に合った方法でインストールしてください。

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

inariを使えるようにするため、installします。

```
uv sync
```

次にデータセットの準備をします

```
uv run python preprocess/create_dataset_categorical.py
```

あるモデルでCOX2を学習してみます。

```
uv run python scripts/train_cox2.py
```
