## Neural Style Transfer:
This code is an implementation of the style transfer algorithm from - *A Neural Algorithm of Artistic Style* (Gatys et al., 2016).  It is based on an implementation originally from Chip Huyen (chiphuyen@cs.stanford.edu).

In short, the algorithm works by setting the Gram matrices of the input and style images at specific layers in a Deep Neural Network to be the same.  The algorithm also aims to perserve the contents of the original image via an L1 loss also at specific layers in the Deep Neural Network.  The Deep Neural Network used here is a pre-trained VGG-Network [1].


## Instructions:
It's easy to use.  The code takes in an input image, the image that will be transformed, and a style image, the style of the image with which to perform the transformation.  
1. 


## References:
1. Simonyan, K. & Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image
Recognition. arXiv:1409.1556 [cs] (2014). URL http://arxiv.org/abs/1409.
1556 ArXiv: 1409.1556
