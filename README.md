# Colorization using Deep Learning Models

This script performs automatic colorization of black-and-white (grayscale) images using two pre-trained deep learning models: ECCV16 and SIGGRAPH17. These models are designed to predict the color (chrominance) information from the luminance (brightness) component of the image.

## Overview

Colorization is a process of adding color to monochrome images. This script utilizes deep learning models trained on large datasets to automatically 
colorize black and white images.

## Features

- Colorize black and white images using two pre-trained deep learning models: ECCV16 and SIGGRAPH17.
- Save the colorized images as JPG files.
- Display the original black and white image, input image, and colorized images side by side.

## Key steps:
- Load a grayscale image.
- Preprocess the image to prepare it for the models.
- Use the models to predict color information.
- Post-process and save the colorized images.
- Visualize the original grayscale, input, and colorized outputs.

## The BaseColor Class (base_color.py)
This class provides essential normalization and unnormalization functions for the luminance (L) and color (ab) components of the images:
  - Normalization: Converts raw pixel values into a scaled range suitable for neural networks.
  - Unnormalization: Converts the scaled outputs back to standard image pixel ranges.
### Details:
  - l_cent and l_norm are used to normalize the L (lightness) channel of the LAB color space.
  - ab_norm is for the a and b color channels.
### Functions:
  - normalize_l(in_l): Centers the L channel around zero and scales it.
  - unnormalize_l(in_l): Converts normalized L back to pixel range.
  - normalize_ab(in_ab): Scales the a and b channels.
  - unnormalize_ab(in_ab): Converts normalized a/b back to actual color values.
This normalization ensures the data fits within ranges the models were trained on, leading to better predictions.

## ECCV16 Model
### Purpose: 
The ECCVGenerator class is a convolutional neural network (CNN) inspired by the ECCV 2016 paper for image colorization.
### Architecture:
- Consists of multiple convolutional layers with ReLU activations.
- Uses downsampling (stride=2 convolution) to capture features at different scales.
- Ends with a regression layer that outputs the predicted a and b channels.
### Functionality:
- Takes in the luminance channel (L) of the grayscale image.
= Outputs the predicted color channels (a, b).
- Uses a softmax layer before the final regression to model the color distribution.
### Pre-trained Weights: 
Loads weights trained on a large dataset, enabling it to predict realistic colors.

## SIGGRAPH17 Model
### Purpose: 
The SIGGRAPHGenerator class is a CNN inspired by the SIGGRAPH 2017 paper, designed for more advanced or refined colorization.
### Architecture:
- More complex with an encoder-decoder structure.
- Uses skip connections (feature map shortcuts) for better detail preservation during upsampling.
- Employs transposed convolutions (upsampling layers) to reconstruct high-resolution color maps.
### Functionality:
- Accepts the luminance and optional extra inputs (like initial color hints).
- Produces the predicted a/b color channels.
- Uses a Tanh activation at the output to keep predictions within a specific range.

## How the Colorization Works
### Input: 
Grayscale image is converted to LAB color space, extracting the L (lightness) component.
### Preprocessing: 
The L component is normalized to fit the model's expected input range.
### Model Prediction: 
The pre-trained models (eccv16 and siggraph17) take the normalized L channel and output the predicted color channels.
### Postprocessing: 
The predicted a/b channels are unnormalized back to the LAB space, combined with the original L channel, and converted back to RGB for display or saving.



## Results

The script generates colorized versions of the input black and white image using the ECCV16 and SIGGRAPH17 models. 
It saves the colorized images as JPG files and displays them using matplotlib.


![adams2](https://github.com/Pavi-NP/colorization/assets/148129933/9cc9efed-45f0-4827-afeb-ab0b95db09d7)  </n>

![sea1_opt_jpg](https://github.com/Pavi-NP/colorization/assets/148129933/2e6e3fca-4239-498b-83fc-3176288d9db7)

## Summary:
- The BaseColor class handles normalization to prepare images for the models and unnormalization to convert model outputs back to natural image pixel ranges.
- The ECCV16 model provides a fast, efficient way to colorize images based on the ECCV 2016 architecture.
- The SIGGRAPH17 model offers a more detailed, potentially more accurate colorization based on the SIGGRAPH 2017 architecture.
- Together, these components enable automatic, high-quality colorization of black-and-white photos using deep learning. 
 
## Credits

This project is inspired by the work of Richard Zhang, Phillip Isola, and Alexei A. Efros. The ECCV16 model is based on the research paper 
"Colorful Image Colorization" by Zhang et al., and the SIGGRAPH17 model is based on the research paper "Real-Time User-Guided Image Colorization with 
Learned Deep Priors" by Zhang et al.

## Other Models and Approaches for Image Colorization
### 1. DeOldify
#### Description: 
one of the most popular open-source deep learning projects for colorizing black-and-white images and videos.
#### Architecture: 
Uses a GAN (Generative Adversarial Network) combined with a U-Net and other advanced techniques.
#### Advantages: 
Produces highly realistic and vibrant colorizations, with the ability to handle various styles and image types.
#### Repository: 
DeOldify on GitHub

### 2. ChromaGAN
#### Description: 
Uses a GAN framework to generate plausible colorizations with attention mechanisms.
#### Highlights: 
Focuses on producing diverse and contextually consistent colors.
#### Application: 
Suitable for artistic and realistic colorization.

### 3. Let there be Color! (2016)
#### Description: 
A pioneering deep learning model that uses a deep convolutional neural network trained on large datasets for automatic colorization.
#### Approach: 
Uses a classification-based approach over color bins, similar to ECCV16 but with some architectural differences.

### 4. Deep Colorization with Conditional GANs
#### Description: 
Conditional GANs trained to produce color images conditioned on grayscale inputs.
#### Example: 
The work by Zhang et al. (2016) that employs conditional GANs for more realistic and diverse colorization outputs.
### 5. DCC-Net (Deep Colorization with Color Constraints)
#### Description: 
Incorporates user inputs or color hints to guide the colorization process.
#### Use Case: 
Useful when partial color hints are available or specific colorization results are desired.
### 6. Colorful Image Colorization (2017)
#### Description: 
Uses deep residual networks and adversarial training for more vivid colors.
#### Advantage: 
Better handling of complex scenes and diverse color palettes.

## ------------------------------------------------------------------------
#### Note
- GAN-based models like DeOldify and ChromaGAN tend to produce more vibrant and realistic results.
- Conditional models allow user guidance for more control.
- Recent research continues to improve realism, diversity, and computational efficiency.

#### Choosing a Model
Choice depends on:
- Quality needed: For highly realistic, artistic results, GAN-based models like DeOldify are excellent.
