# MPM Simulation Engine

個人制作の物理シミュレーションエンジン

## 特徴
- CUDA実装によるGPU加速
- Material Point Method (MPM) による超弾性体シミュレーション
- オリジナル実装（継続開発中）
- experimentalフォルダには開発中のMPMによる非圧縮性流体シミュレーション関数を含む

## 技術スタック
- C++/CUDA
- cuBLAS
- thrust

## デモGIF
- ![cool_particles](https://github.com/user-attachments/assets/1967f114-ec7c-4aca-ab43-30287ff17d5c)

## 使い方
- ディレクトリに入りmkdir build; cd build; cmake ..; make
- 実行ファイルができたらbuildディレクトリ内で./mpm_cuda_executable
- 可視化が必要でしたらbuildディレクトリ内でpython3 ../3dvisualize.pyで../output/mochi.gifが出力されます
- 初期速度などを変えて実行してみると面白いと思います
