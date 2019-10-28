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



# Model:
- https://arxiv.org/pdf/1611.07004.pdf
title={Image-to-Image Translation with Conditional Adversarial Networks},
    author={Phillip Isola and Jun-Yan Zhu and Tinghui Zhou and Alexei A. Efros},
    year={2016},

# Objective function

# TODO:
    
- Work with collab
    - create a script to clone the project
    - create a script to launch the training

- Pay a AWS server to train / and predict



- Make a Flask API & deploy in serverless
