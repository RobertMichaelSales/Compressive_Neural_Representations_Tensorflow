# Compressive Neural Representations (Tensorflow)

A re-implemented and extended version of the original code from "Compressive Neural Representations" repository ([click here](https://github.com/matthewberger/neurcomp)). This, the newer version, is written using the Tensorflow machine learning library. 

Original Authors: [Yuzhe Lu](), [Kairong Jiang](), [Joshua A. Levine](https://jalevine.bitbucket.io/), [Matthew Berger](https://matthewberger.github.io/). 

Link to original ArXiv preprint: [Compressive Neural Representations of Volumtric Scalar Fields](https://arxiv.org/pdf/2104.04523.pdf).

Author: [Robert Sales](https://github.com/RobertMichaelSales). 

## Tests:

In testing on simulated data, compression of scalar values is achieved up-to and beyond a factor of 1000 without severe degradation in visualisation quality. The figure below shows a series of 2D contour slices extracted from the middle of a cube (150x150x150) for increasing compression ratios. The relative error is also presented.

![](https://github.com/RobertMichaelSales/Compressive_Neural_Representations_Tensorflow/blob/main/contours.gif)

## To Do List:
- [X] Develop code to recreate compression results
- [X] Test code on "test_vol.npy" and plot results
- [X] Develop code to save network architecture
- [X] Develop code to save network parameters
- [X] Develop code to load network architecture
- [X] Develop code to load network parameters
- [X] Validate code for saving network architecture
- [X] Validate code for saving network parameters
- [X] Validate code for loading network architecture
- [X] Validate code for loading network parameters
- [ ] Extend code to work for 4D (3D+Time) data
- [ ] Test code extension for 4D (3D+Time) data
- [X] Develop DataClass to separate volume and values
- [ ] Develop code for plotting results
- [X] Develop auxiliary code to convert .vts files to .npy files
- [X] Test auxiliary code to convert .vts files
- [ ] Test compression on real 3D data fields
- [ ] Experiment with batch size and learning rate (as function of input size)
- [ ] Finish code for saving decompressed outputs
