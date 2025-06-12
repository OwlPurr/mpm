# Simulation code using MPM

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
- ![mochi](https://github.com/user-attachments/assets/68712171-d710-4601-938e-78206ce9beb3)


## 使い方
- ディレクトリに入りmkdir build; cd build; cmake ..; make
- 実行ファイルができたらbuildディレクトリ内で./mpm_cuda_executable
- 可視化が必要でしたらbuildディレクトリ内でpython3 ../3dvisualize.pyで../output/mochi.gifが出力されます
- 初期速度などを変えて実行してみると面白いと思います

## 追記
- experimentalフォルダ内の流体用関数を整備しましたので近日挙げます
- 本実装の離散化部分は、以下のサイトを参考にさせていただきました。I am incredibly grateful for your site!
- URL:https://alishelton.github.io/apic-writeup/
- 現在のデモGIF(調整中)
- ![mochi](https://github.com/user-attachments/assets/0c1067fa-6c7e-4e37-81ea-605a0e91c3c3)
