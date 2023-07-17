inari
======
Library for GNN Model using torch-geometric

inariとは
==========

GNNのモデルを比較するためのライブラリ

Getting Started
===============

poetryをインストールします。
[poetry](https://python-poetry.org/docs/)のページを参考に、自分の環境に合った方法でインストールしてください。

```
curl -sSL https://install.python-poetry.org | python -
```

inariを使えるようにするため、installします。

```
poetry install
```

次にデータセットの準備をします

```
poetry run python preprocess/create_dataset_categorical.py
```

あるモデルでCOX2を学習してみます。

```
poetry run python src/train_cox2.py
```
