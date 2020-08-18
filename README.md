# Aligned Region-CNN

Created by Lu Zhang, Institute of Automation, Chinese Academy of Science.

## Introduction

We propose a novel detector, called AR-CNN, that tackles the practical position shift problem in multispectral pedestrian detection. For more details, please refer to our [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Weakly_Aligned_Cross-Modal_Learning_for_Multispectral_Pedestrian_Detection_ICCV_2019_paper.html). 

### KAIST-Paired Annotation

The KAIST-Paired annotation is available through [[Google Drive]](https://drive.google.com/open?id=1FLkoJQOGt4PqRtr0j6namAaehrdt_A45) and [[BaiduYun]](https://pan.baidu.com/s/1Dl9jxEH2r83CzMRwLkxZwg). If you have any problem, please feel free to contact me.

## Preparation


First of all, clone the code:
```
git clone https://github.com/luzhang16/AR-CNN.git
```

Then, create a folder:
```
cd $AR-CNN && mkdir data
```

### prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0
* CUDA 8.0 or higher

### Data Preparation

It is recommended to symlink the dataset root to `$AR-CNN/data`. 

```
AR-CNN
├── cfgs
├── lib
├── data
│   ├── kaist-paired
│   │   ├── annotations
│   │   ├── images
│   │   ├── splits
```

**KAIST dataset**: Please follow the instructions in [rgbt-ped-detection](https://github.com/SoonminHwang/rgbt-ped-detection) to prepare KAIST dataset. 

**KAIST-Paired annotation**: [Google Drive](https://drive.google.com/open?id=1FLkoJQOGt4PqRtr0j6namAaehrdt_A45) or [BaiduYun](https://pan.baidu.com/s/1Dl9jxEH2r83CzMRwLkxZwg).

**Trainval & Test splits**: `mv $AR-CNN/splits/ $AR-CNN/data/kaist-paired/`

### Pretrained Model

We use VGG16 pretrained models in our experiments. You can download the model from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

Download them and put them into the data/pretrained_model/.

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the CUDA code:

| GPU model  | Architecture |
| ------------- | ------------- |
| TitanX (Maxwell/Pascal) | sm_52 |
| GTX 960M | sm_50 |
| GTX 1080 (Ti) | sm_61 |
| Grid K520 (AWS g2.2xlarge) | sm_30 |
| Tesla K80 (AWS p2.xlarge) | sm_37 |

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## Test

If you want to get the detection results on KAIST "reasonable" test set, simply run:
```
python test_net.py --dataset kaist --net vgg16 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --reasonable --cuda
```
Specify the specific model session, checkepoch and checkpoint, e.g., SESSION=1, EPOCH=3, CHECKPOINT=17783.

If you want to run with our pretrained model, download the pretrained model through [Google Drive](https://drive.google.com/file/d/1NIU3vKQDEi39hZt1ayhNDYdQA0UGw9uH/view?usp=sharing) or [BaiduYun (pwd: 3bxs)](https://pan.baidu.com/s/1xS6dtfxGYHr6Jr7fRYI0Qg). 

### Evaluate the output

You can use the evaluation script provided by the original KAIST dataset or this matlab evaluation tool: [Google Drive](https://drive.google.com/drive/folders/1ZAfQMsgu9BzMtxRMnTivwk-Se6ZpGuYa?usp=sharing) or [BaiduYun (pwd: 41qk)](https://pan.baidu.com/s/1FFmCUR1JMROk0d-tzjA00w)

### Robustness test with manual position shift

If you want to get the detection results under the metric `S`, simply run:

``` 
sh test_shift.sh
```

If you want to get the detection results under certain position shift, simply run:

```
python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --reasonable --cuda --sx 4 --sy -4
```

Note that `sx` and `sy` denote the pixels of position shift along x-axis and y-axis respectively.

## Acknowledgement

We appreciate much the code [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) developed by [Jianwei Yang](https://github.com/jwyang) and [Jiasen Lu](https://github.com/jiasenlu). This code is built mostly based on it. 
