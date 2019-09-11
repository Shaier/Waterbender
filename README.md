# Waterbender
![Input_Demo](https://github.com/Shaier/Waterbender/blob/master/data/input/62_rain.jpg)
![Output_Demo](https://github.com/Shaier/Waterbender/blob/master/data/input/62_prediction.jpg)

## Installation
1. Clone this repo
2. Install dependencies with   
pip install -r requirements.txt
3. Place weights (see below) in 'inputs/models', or train the model yourself

## Testing
python predict.py --input_dir data/input --output_dir data/output

## Dataset
The whole dataset can be found here:  
https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K  

## Training
I thought that it will be quite intuitive, straightforward and educational to train using a jupyter notebook script. Hence, please see [Demo.ipynb](Demo.ipynb)    

##### Training Set:

861 image pairs for training.

##### Testing Set A:

For quantitative evaluation where the alignment of image pairs is good. A subset of testing set B.

##### Testing Set B:

239 image pairs for testing.
## Weights
If you choose to not train the model yourself you can use the pre-trained model I made.   
Weights can be downloaded from here:  
https://drive.google.com/open?id=1-CPFTmrehlQ2JMApI_-1eifuvIYCV6Ax

## Demo
The demo pictures are put under 'data/input' and 'data/output'. They are a sample of the ouputs of the model.

## About this project
Water has an amazing ability to adhere (stick) to itself and to other substances.  
I will save you the discussion on hydrogen bonds and some more chemical properties of water, though it is quite fascinating and you should [explore](https://manoa.hawaii.edu/exploringourfluidearth/chemical/properties-water/hydrogen-bonds-make-water-sticky).
  
Anyway, raindrops adhered to windows, glasses, camera lens, etc. can severely reduce the visibility of the background scene. In this project I addressed
this problem by developing a model that removes raindrops, and thus transforming a raindrop degraded image into a clean one.  
To resolve the problem, I knew that I needed to create a model which can take a bad input photo (image with noise, bad quality, etc.)  
and output something that looks much better. So clearly what I wanted to do is to use a U-Net, because we know that U-Nets can do 
exactly that kind of thing. For the U-Net I used the architecture of ResNet 34.  
I also wanted some loss function that does a better job than pixel mean squared error loss, since if you think about it, 
the mean squared error between the pixels of the bad image and the good image is actually very small most often.  
There's a fairly good way to solve that, and it's something called a Generative adversarial network (GAN). 
GAN tries to solve this problem by using a loss function which actually calls another model (they go back and forth but I won't go into that). 
Some of the issues with GANs though were that they take a long time to train and the outputs are still improvable.   
And so the next step was "can we get rid of GANs entirely?
Obviously! The thing we really want to do is come up with a better loss function. We want a loss function that does a good job 
of saying this is a high-quality image without having to go through all the GAN trouble.  
The real trick here comes back to this paper from a couple of years ago, [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155). 
The authors created this thing they call perceptual losses, though, in fast.ai you'll see this referred to as feature losses.
Anyway, so we have this generator and we want a loss function that says "is the image that is created similar to the image that we want?" 
And so the way they do that is by taking the prediction and putting it through a pre-trained ImageNet network (VGG).  
Normally, the output of that would tell you "hey, is this generated thing a pony, a spider, or an airplane?" 
But in the process of getting to that final classification, it goes through lots of different layers.  
What we could do is say "let's not take the final output of the VGG model on this generated image, but let's take something in the middle". 
Let's take the activations of some layer in the middle. Those activations might be a feature map of 256 channels by 28 by 28. 
So those kind of 28 by 28 grid cells might say  like "in this part of that 28 by 28 grid, 
is there something that looks kind of furry? Or is there something that looks kind of shiny? Or is there something that was kind of circular?  
So what we do is take the target (i.e. the actual y value) and we put it through the same pre-trained VGG network, 
and we pull out the activations of the same layer. Then we do a mean square error comparison. So it'll say "in the real image, grid cell 
(1, 1) of that 28 by 28 feature map is furry and blue and round shaped. And in the generated image, it's furry and blue and not round 
shape." So it's an okay match.    
