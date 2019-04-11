# ColorizerGAN
Colorization of greyscaled images with GAN

The purpose of this experimentation is to test some basic GANs architectures applied to the problem of greyscaled images colorization and merge different approaches in order to obtain the colorization on a large domain of different images. 
We tested different the different implementations on the place365 dataset wich offers a good start point in terms of variety.

The tested networks are:

- [pix2pix model](https://arxiv.org/abs/1611.07004) starting from [this nice implementation](https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py).
  This network is a conditional GAN that uses a U-NET generator and a classic convolutional discriminator. 
  
  
- [encoder-decoder model](https://github.com/baldassarreFe/deep-koalarization) with a fusion layer starting from [this Keras implementation](https://github.com/emilwallner/Coloring-greyscale-images/tree/master/Full-version). The fusion layer is built extracting high-level features from  [Inception-ResNet-v2](https://arxiv.org/abs/1602.07261) pre-trained on ImageNet dataset. Instead of using Inception-ResNET-v2 we tested the [Emil Wallner](https://github.com/emilwallner) implementation with [MobileNet](https://arxiv.org/abs/1704.04861) wich is lighter on Emil advice (thanks, Emil!). The strenght of this implementation is that we can do transfer learning with few samples, because the fusion layer is pre-trained

- Finally we implemented our own model wich combines the previous ones: we used the encoder-decoder model with the fusion layer as generator and the pix2pix discriminator. We wanted to use the MobileNet knowledge trained on ImageNet to colorize greyscaled photos and make them better throught the Pix2pix's LSGAN discriminator wich take in input the real and the fake photos, combines them and 
We conducted on this model different type of preliminary trainings.

We tested the trained models on 50 images taken from the place365 dataset not used during the traininig phase, evaluating the results with the simple turing test. 
The preliminary results shows that the models are able to recognize and to colorize skies and vegetation. With a longer training we should appreciate improvements.


### Some results with different training and model conditions
In the images below, we can see the different preliminary experiments on the architectures above with different training parameters:
starting from left : ...,...,...,....,...,....,Ground Thruth

![alt text](https://github.com/C1AFFA/ColorizerGAN/blob/master/RESULTS/TEST-0-7.jpg "Preliminary testing results 1")
![alt text](https://github.com/C1AFFA/ColorizerGAN/blob/master/RESULTS/TEST-42-49.jpg "Preliminary testing results 2")

### Implementation details:
- We used a 11GB GPU (GTX 1080TI); the training on 10 epochs on our models required 11 hours.


