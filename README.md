# Simulation code using MPM

個人制作の物理シミュレーションエンジンです。

## 特徴

- CUDA実装によるGPU加速
- Material Point Method (MPM) による超弾性体シミュレーション
- 完全オリジナル実装（継続開発中）
- `experimental` フォルダには、MPMによる非圧縮性流体シミュレーション関数を含みます

## 技術スタック

- C++ / CUDA
- cuBLAS
- thrust

## デモGIF

- ![cool_particles](https://github.com/user-attachments/assets/1967f114-ec7c-4aca-ab43-30287ff17d5c)
- ![mochi](https://github.com/user-attachments/assets/68712171-d710-4601-938e-78206ce9beb3)

## 使い方

1. リポジトリをクローンしてビルド：

   ```bash
   git clone https://github.com/OwlPurr/mpm.git
   cd mpm
   mkdir build
   cd build
   cmake ..
   make
   ```

2. 実行前に `output` フォルダを作成（出力先用）：

   ```bash
   mkdir ../output
   ```

3. 実行ファイルを実行：

   ```bash
   ./mpm_cuda_executable
   ```
   流体の時は
   ```bash
   ./mpm_cuda_executable FLUID
   ```

5. 可視化が必要な場合は `build` ディレクトリ内で：

   ```bash
   python3 ../3dvisualize.py
   ```

   実行後、`mpm/output/mochi.gif` が生成されます。いろいろmain.cu内のパラメータなどを変えてみてください。

## 追記（2025/06/12）

- `experimental` フォルダ内の流体用関数を整備しました！
- 本実装の離散化部分は、以下のサイトを参考にしています。とても感謝しています：

  > https://alishelton.github.io/apic-writeup/

- 流体のデモGIF：
  
  ![mochi](https://github.com/user-attachments/assets/0c1067fa-6c7e-4e37-81ea-605a0e91c3c3)

