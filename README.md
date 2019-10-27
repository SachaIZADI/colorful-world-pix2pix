Hello world

## Data:
We trained our Colorizer model with face picture from the open-source dataset (*Labeled Faces in the Wild*)[http://vis-www.cs.umass.edu/lfw/] (LFW). 
It contains more than 13,000 images of faces collected from the web.

To download the dataset you can use the shell script from `colorful-world/data/download_data.sh` 

```
cd colorful-world/data
chmod +x download_data.sh #Rq: you might not need this
colorful-world/data/download_data.sh
```


# Model:
- https://arxiv.org/pdf/1611.07004.pdf
title={Image-to-Image Translation with Conditional Adversarial Networks},
    author={Phillip Isola and Jun-Yan Zhu and Tinghui Zhou and Alexei A. Efros},
    year={2016},


# TODO:

- Read/find pix2pix paper
- Compare with existing implementations
    - https://github.com/SachaIZADI/Colorful-World/blob/master/TensorFlow_implementation/model.py
    - https://github.com/znxlwm/pytorch-pix2pix/blob/master/network.py


- Work with collab
- Pay a AWS server to train / and predict



- Make a Flask API & deploy in serverless
