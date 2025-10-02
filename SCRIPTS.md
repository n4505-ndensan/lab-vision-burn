# Lab Vision Burn - スクリプトガイド

このプロジェクトでは、MNISTとCIFAR-10の両方でコンピュータビジョンモデルの学習が可能です。

## 🚀 クイックスタート

```bash
# プロジェクト全体のセットアップ（初回のみ）
pnpm setup

# MNISTモデルの完全パイプライン
pnpm mnist:all

# CIFAR-10モデルの完全パイプライン  
pnpm cifar10:all

# ウェブアプリケーション起動
pnpm web
```

## 📋 利用可能なコマンド

### MNIST関連
- `pnpm mnist:train` - MNIST学習（10エポック）
- `pnpm mnist:eval` - MNISTモデル評価
- `pnpm mnist:wasm` - MNIST WebAssembly生成
- `pnpm mnist:all` - 上記3つを一括実行

### CIFAR-10関連
- `pnpm cifar10:setup` - CIFAR-10データセットダウンロード
- `pnpm cifar10:train` - CIFAR-10学習（10エポック）
- `pnpm cifar10:eval` - CIFAR-10モデル評価
- `pnpm cifar10:wasm` - CIFAR-10 WebAssembly生成
- `pnpm cifar10:all` - 学習→評価→WASM生成を一括実行

### 開発・ビルド関連
- `pnpm model:clean` - Rustビルドキャッシュクリア
- `pnpm model:check` - Rustコードチェック
- `pnpm web` - 開発サーバー起動
- `pnpm web:build` - ウェブアプリケーションビルド
- `pnpm build:all` - 全モデル学習+ウェブビルド

## 🎯 推奨ワークフロー

### 初回セットアップ
```bash
# 1. 依存関係とCIFAR-10データの準備
pnpm setup

# 2. MNISTモデルをテスト（高速）
pnpm mnist:all

# 3. CIFAR-10モデルを学習（時間がかかる）
pnpm cifar10:all

# 4. ウェブアプリで確認
pnpm web
```

### 開発時
```bash
# コード変更後の確認
pnpm model:check

# 特定のデータセットのみ再学習
pnpm mnist:train
# または
pnpm cifar10:train

# WebAssembly再生成
pnpm mnist:wasm
# または  
pnpm cifar10:wasm
```

## 📊 期待される性能

- **MNIST**: ~98%精度（手書き数字認識）
- **CIFAR-10**: ~67%精度（カラー画像分類）

## 📁 生成されるファイル

```
model/
├── artifacts/
│   ├── mnist/
│   │   ├── model.burn    # Burnネイティブ形式
│   │   └── model.bin     # バイナリ形式
│   └── cifar10/
│       ├── model.burn
│       └── model.bin
└── pkg/                  # WebAssembly出力
    ├── *.wasm
    └── *.js
```

## ⚡ カスタムパラメータ

個別にパラメータを調整したい場合：

```bash
# エポック数やバッチサイズのカスタマイズ
cd model
cargo run --release -- train --dataset cifar10 --epochs 20 --batch-size 64

# 推論テスト
cargo run --release -- infer --dataset mnist --path my_digits/
```