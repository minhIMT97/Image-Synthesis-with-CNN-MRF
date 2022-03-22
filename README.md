# Image-Synthesis-with-CNN-MRF

#### Group 5-a: Binh Minh NGUYEN ([minhIMT97](https://github.com/minhIMT97)) - Minh Triet VO ([trietvo3105](https://github.com/trietvo3105))
This is the repository for the course Computational Imaging project with the subject: Combination of Convolutional Neural Network and Markov Random Field for image synthesis.

In this project, we replace the feature extractor VGG19 by other pretrained networks such as resnet34. The goal is to examize the influence of the feature extractor CNN on the result. 

The implementation is based on: https://github.com/jonzhaocn/cnnmrf-pytorch. We added our modification to adopt resnet34, compute the quantitative metrics, and plot the loss function. 

### Content and style images

![Content and style used](images/CNNMRF-C&S.png)

### Algorithm testing

To run the code, clone this repository and run the command below in the terminal:

'''
!python3 main.py --content_path data/content1.jpg --style_path data/style1.jpg --max_iter 60 --num_res 3 --content_weight 0.4 --style_weight 0.5 --tv_weight 0.001 --mrf_style_stride 1 --mrf_synthesis_stride 1 --model resnet
'''

### Defaut results on VGG19

![VGG result](images/CNNMRF-VGG19.png)

### Results on Resnet34

![Resnet34 result](images/CNNMRF-Resnet34.png)


