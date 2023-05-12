# Nanodet (Quant and Deploy based on Openvino)

官方代码仓库：

https://github.com/RangiLyu/nanodet



## 介绍

本仓库基于 NNCF 工具将 nanodet-plus-m_320 模型量化 (PTQ) 至 int8 精度，推理速度更快！！！

在 OpenVINO 推理框架下部署 Nanodet 检测算法，并重写预处理 Warp Affine 和 后处理 NMS 部分，具有超高性能！！！

让你在 Intel CPU 平台上的检测速度起飞！！！



**优势**：方便部署，高性能。



## 推理速度

|          Model          | (fp32) infer avg latency | (int8) infer avg latency |
| :---------------------: | :----------------------: | :----------------------: |
|   nanodet-plus-m_320    |         3.12 ms          |         2.41 ms          |
|   nanodet-plus-m_416    |         4.80 ms          |         3.58 ms          |
| nanodet-plus-m-1.5x_320 |         4.70 ms          |         3.41 ms          |
| nanodet-plus-m-1.5x_416 |         7.59 ms          |         5.18 ms          |

**注**：实际程序运行速度与图像中目标数量有关，目标越多，后处理解码和NMS耗时则越多。



## 安装 OpenVINO Toolkit

参考官网安装教程 [Get Started Guides](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_apt.html)

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino/2022 focal main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list
sudo apt update
apt-cache search openvino
sudo apt install openvino
apt list --installed | grep openvino
```

python 安装

```bash
pip install openvino
```



## 模型导出与修改

1. Export ONNX model

   ```bash
   cd nanodet
   python tools/export_onnx.py --cfg_path config/nanodet-plus-m_320.yml --model_path weights/nanodet-plus-m_320.pth
   ```

2. Convert to OpenVINO

   ```bash
   mo --framework onnx --input_model nanodet.onnx
   ```

3. Add Preprocess

   ```bash
   cd tools
   python add_preprocess
   # 注意修改其中的模型路径
   ```



## 模型量化

参考官网量化教程 [Post-training Quantization with NNCF](https://docs.openvino.ai/latest/nncf_ptq_introduction.html)

1. 准备300张标定数据集，放入 `tools/imgs` 路径下

2. 安装NNCF

   ```bash
   pip install nncf
   ```

3. 执行量化

   ```bash
   cd tools
   python quant.py
   # 注意修改其中的模型路径
   ```



## C++ demo Build and Run

#### build

```bash
cd /your_path/Nanodet_openvino_quant_deploy
mkdir build && cd build
cmake .. && make -j
```

#### run

```bash
cd workspace
```

**图片输入:**

```bash
./pro 0 "imgs/car.jpg"
```

**摄像头输入:**

```bash
./pro 1 0
```

**视频文件输入:** 

```bash
./pro 2 "videos/palace.mp4"
```

**benchmark** 

```bash
./pro 3 0
```



