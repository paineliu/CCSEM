ubuntu 22.04 环境搭建

1、创建conda环境

conda create -n ccse python=3.9


2、安装cuda 11.1
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run

3、安装 pytorch 1.10.1
https://pytorch.org/get-started/previous-versions/
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

https://hub.docker.com/r/pytorch/pytorch/tags

4、安装detectron2
https://detectron2.readthedocs.io/en/latest/tutorials/install.html
git clone https://github.com/facebookresearch/detectron2.git
如果git无法下载，下载后解压，不要进入解压目录下，执行：
python -m pip install -e detectron2

5、不支持g++ 11 因此 安装 g++ 10 
https://blog.csdn.net/qq_39779233/article/details/105124478

sudo apt-get install -y gcc-10 g++-10

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 20 --slave /usr/bin/g++ g++ /usr/bin/g++-10

6、安装依赖库
pip install opencv-python
pip install scipy


apt install libgl1-mesa-glx
apt install libglib2.0-0


7、报错解决方法：

错误1：
AttributeError: module ‘distutils‘ has no attribute ‘version‘ 解决方案
https://zhuanlan.zhihu.com/p/556704117

pip install setuptools==58.0.4

错误2：

AttributeError: module ‘PIL.Image‘ has no attribute ‘ANTIALIAS‘
原来是在pillow的10.0.0版本中，ANTIALIAS方法被删除了，使用新的方法即可：
原文链接：https://blog.csdn.net/light2081/article/details/131517132

pip uninstall -y Pillow
pip install Pillow==9.5.0

二、修改配置文件

config\instance_segmentation\mask_rcnn_R_50_FPN_3x_kaiti.yaml

  WEIGHTS: "./output/mask_rcnn_R_50_FPN_3x_kaiti/model_final.pth"
  OUTPUT_DIR: ./output
  DATA_ROOT: ./dataset/kaiti_chinese_stroke_2021
  VIS_DATASET_RESULT: true 
  
三、拷贝数据文件到dataset目录

在dataset目录下执行
unzip -d kaiti_chinese_stroke_2021 kaiti_chinese_stroke_2021.zip

四、训练以及推理

训练：train_instance.py
推理：inference_instance.py


附：CCSE源码修改内容

1、移动脚本文件

script/train_instance.py
script/inference_instance.py
到上一层目录

2、添加默认参数

/mnt/data1/CCSE/common/cmd_parser.py

+    # parser.add_argument('--config', type=str, default='./config/instance_segmentation/mask_rcnn_R_50_FPN_3x_handwritten.yaml', help='path to config file')
+    parser.add_argument('--config', type=str, default='./config/instance_segmentation/mask_rcnn_R_50_FPN_3x_kaiti.yaml', help='path to config file')

3、 注释代码
module\sparse_rcnn\util\misc.py
module\reference_sparse_rcnn\util\misc.py

# if float(torchvision.__version__[:3]) < 0.7:
#     from torchvision.ops import _new_empty_tensor
#     from torchvision.ops.misc import _output_size

3、修改 module/instance/evaluator.py

-                    use_fast_impl=self._use_fast_impl,
+                    # use_fast_impl=self._use_fast_impl,
