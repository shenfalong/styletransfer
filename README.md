Neural Style Transfer
=====


This repository is for [Meta Networks for Neural Style Transfer](https://arxiv.org/abs/1709.04111). The meta network takes in the style image and generated an image transformation network for neural style transfer. The fast generated model is only `449KB`, which is able to real-time execute on a mobile device. For more details please refer and cite this paper

    @inproceedings{shen2017style,
      author = {Falong Shen, Shuicheng Yan and Gang Zeng},
      title = {Meta Networks for Neural Style Transfer},
      booktitle = {arXiv:1709.04111},
      year = {2017}
    }

Installation
----
This library is based on [Caffe](https://github.com/BVLC/caffe). [CuDNN 7](https://developer.nvidia.com/cudnn) and [NCCL 1](https://github.com/NVIDIA/nccl) are required. Please follow
the installation instruction of [Caffe](https://github.com/BVLC/caffe).

Meta Network Architecture
----
<div align=center>
<img src="python/network.png", width="500" height="300"/>
</div>

Examples
----
The size of image transformation network for the following images is `7MB`.

<div align=center>
<img src="python/1.png", width="400" height="500"/> <img src="python//2.png", width="400" height="500"/>
</div>


The size of image transformation network for the following images is `449KB`.
<div align=center>
<img src="python/4.png", width="400" height="300"/> 
</div>


Scripts
----
Python code. Please execute the scripts in Python folder. Meta model is very huge while the generated model is very small. 

* pretrained meta models</br>
    Meta model [train_8](http://pan.baidu.com/s/1mhGwQJA) (`130M`), generated model is `449KB`.</br>
    Meta model [train_32](http://pan.baidu.com/s/1eRQI01O) (`968M`), generated model is `7MB`.</br>
    
Put these models into <font color=red>python/model/<font> and modify the model name in <font color=red>demo.py<font>.