# AutoSpeaker: Neural Architecture Search for Speaker Recognition

Code for this paper [AutoSpeaker: Neural Architecture Search for Speaker Recognition](https://arxiv.org/abs/2005.03215)

Shaojin Ding*, Tianlong Chen*, Xinyu Gong, Weiwei Zha, Zhangyang Wang

## Overview
Speaker  recognition  systems  based  on  Convolutional  Neural Networks (CNNs) are often built with off-the-shelf backbones such as VGG-Net or ResNet. However, these backbones were originally proposed for image classification, and therefore may not  be  naturally fit for  speaker  recognition.   Due  to  the  prohibitive complexity of manually exploring the design space, we propose the first neural architecture search approach approach for the speaker recognition tasks, named as AutoSpeech.  Our algorithm first identifies the optimal operation combination in a neural cell and then derives a CNN model by stacking the neural cell for multiple times. The final speaker recognition model can  be  obtained  by  training  the  derived  CNN  model  through the standard scheme. To evaluate the proposed approach,  we conduct experiments on both speaker identification and speaker verification tasks using the VoxCeleb1 dataset. Results demonstrate  that  the  derived  CNN  architectures  from  the  proposed approach significantly outperform current speaker recognition systems  based  on  VGG-M,  ResNet-18,  and  ResNet-34  back-bones, while enjoying lower model complexity.

## 

## Quick start
### Requirements
* Python 3.7

* Pytorch>=1.0: `pip install torch torchvision`

* Other dependencies: `pip install -r requirements`

### Dataset
[VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html): You will need `DevA-DevD` and `Test` parts. Additionally, you will need original files: `vox1_meta.csv`, `iden_split.txt`, and `veri_test.txt` from official website.

The data should be organized as:
* VoxCeleb1
   * wav
   * vox1_meta.csv
   * iden_split.txt
   * veri_test.txt
   
### Running the code
* data preprocess:

    `python data_preprocess.py /path/to/VoxCeleb1`

* Training and evaluating ResNet-18, ResNet-34 baselines:

    `python train_baseline.py --cfg exps/baseline/resnet18.yaml`
    
    `python train_baseline.py --cfg exps/baseline/resnet34.yaml`
    
    You need to modify the `DATA_DIR` field in `.yaml` file.

* Architecture search:

    `python search.py --cfg exps/search.yaml`
    
    You need to modify the `DATA_DIR` field in `.yaml` file.
    
* Training from scratch:
    
    `python train.py --cfg exps/scratch/scratch.yaml --text_arch GENOTYPE`
    
    You need to modify the `DATA_DIR` field in `.yaml` file.
    
    `GENOTYPE` is the search architecture object. For example, the `GENOTYPE` of the architecture report in the paper is:
    
    `"Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))"`
    
* Evaluation:

  * Identification

    `python evaluate_identification.py --cfg exps/scratch/scratch.yaml --load_path /path/to/the/trained/model`

  * Verification
  
    `python evaluate_verification.py --cfg exps/scratch/scratch.yaml --load_path /path/to/the/trained/model`


### Visualization

left: normal cell. right: reduction cell
<p align="center">
<img src="figures/searched_arch_normal.png" alt="progress_convolutional_normal" width="45%">
<img src="figures/searched_arch_reduce.png" alt="progress_convolutional_reduce" width="45%">
</p>

## Results

Our proposed approach outperforms speaker recognition systems based on VGG-M, ResNet-18, and ResNet-34 backbones. The detailed comparison can be found in our paper.

|    Method     | Top-1 |  EER  | Parameters |
| :------------: | :---: | :---: | :---: |
|  VGG-M    | 80.50 | 10.2 | 67M |
| ResNet-18 | 79.48 | 8.17 | 12M |
| ResNet-34 | 81.34 | 4.64 | 22M |
| Proposed  | **87.66** | **1.45** | **18M** |


## Citation

If you use this code for your research, please cite our paper.

```
@misc{ding2020autospeech,
    title={AutoSpeech: Neural Architecture Search for Speaker Recognition},
    author={Shaojin Ding and Tianlong Chen and Xinyu Gong and Weiwei Zha and Zhangyang Wang},
    year={2020},
    eprint={2005.03215},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
