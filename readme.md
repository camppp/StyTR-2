# Image Style Transfer with Transformers: [Project Report](https://github.com/camppp/StyTR-2/blob/main/neural_style_transfer_paper.pdf)

## Results presentation 
<p align="center">
<img src="https://github.com/diyiiyiii/StyTR-2/blob/main/Figure/Unbiased.png" width="90%" height="90%">
</p>
Compared with some state-of-the-art algorithms, our method has a strong ability to avoid content leakage and has better feature representation ability.  
We improve upon the original paper by introducing new mechanisms for positional encoding. We also greatly simplified the model structure and provided support for arbitrary input dimensions
In addition, we implemented a desktop application that enables applying this style transfer model to videos. <br>

## Framework
<p align="center">
<img src="https://github.com/diyiiyiii/StyTR-2/blob/main/Figure/network.png" width="100%" height="100%">
</p> 
The overall pipeline of our StyTr^2 framework. We split the content and style images into patches, and use a linear projection to obtain image sequences. Then the content sequences added with CAPE are fed into the content transformer encoder, while the style sequences are fed into the style transformer encoder. Following the two transformer encoders, a multi-layer transformer decoder is adopted to stylize the content sequences according to the style sequences. Finally, we use a progressive upsampling decoder to obtain the stylized images with high-resolution.




## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [vit_embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing), [decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing), [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing)   <br> 
Please download them and put them into the floder  ./experiments/  <br> 
```
python test.py  --content_dir input/content/ --style_dir input/style/ --output out
```
### Training  
Style dataset is WikiArt collected from [WIKIART](https://www.wikiart.org/)  <br>  
content dataset is COCO2014  <br>  
```
python train.py --style_dir ./datasets/Images/ --content_dir ./datasets/train2014 --save_dir models/ --batch_size 8
```
