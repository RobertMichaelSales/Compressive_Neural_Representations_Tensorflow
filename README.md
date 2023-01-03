# Compressive Neural Representations (Tensorflow)

A re-implemented and extended version of the original code from "Compressive Neural Representations" repository ([click here](https://github.com/matthewberger/neurcomp)). This, the newer version, is written using the Tensorflow machine learning library. 

Original Authors: [Yuzhe Lu](), [Kairong Jiang](), [Joshua A. Levine](https://jalevine.bitbucket.io/), [Matthew Berger](https://matthewberger.github.io/). 

Link to original ArXiv preprint: [Compressive Neural Representations of Volumtric Scalar Fields](https://arxiv.org/pdf/2104.04523.pdf).

Author: [Robert Sales](https://github.com/RobertMichaelSales). 

## Tests:

In testing on simulated data, compression of scalar values is achieved up-to and beyond a factor of 1000 without severe degradation in visualisation quality. The figure below shows a series of 2D contour slices extracted from the middle of a cube (150x150x150) for increasing compression ratios. The relative error is also presented.

![](https://github.com/RobertMichaelSales/Compressive_Neural_Representations_Tensorflow/blob/encoding/contours.gif)

## To do:
- [x] Upload files
- [X] Develop code to recreate compression results
- [X] Test code on "test_vol.npy" and plot results
- [X] Develop code to save network architecture
- [X] Develop code to save network weights & biases
- [X] Develop code to load network architecture
- [X] Develop code to load network weights & biases
- [ ] Test code for loading/saving 
