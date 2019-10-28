Hello world

## Data:
We trained our Colorizer model with face picture from the open-source dataset [*Labeled Faces in the Wild*](http://vis-www.cs.umass.edu/lfw/) (LFW). 
It contains more than 13,000 images of faces collected from the web.

To download the dataset you can use the shell script from `colorful-world/data/download_data.sh` 

```
cd colorful_world/data
chmod +x download_data.sh #Rq: you might not need this
download_data.sh
```

We designed a `pytorch Dataset` to handle the generation of training samples. 
It takes a colored image and transforms it into a black & white image.

<img src = "/media/color2black&white.png" height="250">


## Model:
This repo is largely inspired by the paper [*Image-to-Image Translation with Conditional Adversarial Networks*](https://arxiv.org/pdf/1611.07004.pdf)
published in 2016 by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros.

We build a conditional Generative Adversarial Network (cGAN) made of:
- a generator `G` taking a black&white image as input and generating a colorized version of this image (conditioned on the image)
- a discriminator `D` taking a black&white image and a colorized image (either ground truth or generated by `G`). 
It predicts, **conditionally** to the B&W image if the colorized input is the ground truth or a generated example.

<img src = "/media/GAN.png" height="150">


The generator has a UNet architecture. This architecture is often used for image segmentation, and one could justify that this architecture
helps the generator avoid coloring beyond the edges of the B&W image. A bit like a child.

<img src = "/media/color_edge.jpg" height="150">

It is a variation of the classical autoencoder:
<img src = "/media/Unet.png" height="200">

The `pytorch` computational graph of the model is: 
<img src = "/media/generator.png" height="200">

The discriminator is a classical Convet classifier that takes both a B&W and a colored image as input:
<img src = "/media/discriminator.png" height="200">

## Training algorithm

As in the traditional GAN setting, the generator and the discriminator play a Min-Max game. Here our loss function is
<img src = "/media/loss_fn.png" height="75">

That being said, contrary to the pix2pix paper, we did not implement any source of randomness in the generation of the colorized images.

## Results

<img src = "/colorful_wolrd/results/loss_graph.png" height="200">

<img src = "/colorful_wolrd/results/color_evolution/colorization_training.gif" height="200">


# TODO:
    
- Work with collab
    - create a script to clone the project
    - create a script to launch the training

- Pay a AWS server to train / and predict



- Make a Flask API & deploy in serverless
