msDyNODE - Neural Ordinary Differential Equation-based multi-scale Dynamical modeling
====================================================

This code implements the model from the paper "[Multiscale effective connectivity analysis of brain activity using neural ordinary differential equations](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0314268)". It is a deep learning-based
approach to characterize brain communications between regions and scales. By modeling the continuous dynamics of hidden states using the neural network-based ordinary differential equations, the requirement of downsampling the faster sampling signals is discarded, thus preventing from losing dynamics information. Another advantageous feature of the proposed method is flexibility. An adaptable framework to bridge the gap between scales is necessary. Depending on the neural recording modalities utilized in the experiment, any suitable pair of well-established models can be plugged into the proposed multi-scale modeling framework. Thus, this method can provide insight into the brain computations of multiscale brain activity.


Prerequisites
----------------------------------------------------

The code is written in Python 3.7.4. You will also need:
- **PyTorch**
- **torchdiffeq** for solving ordinary differential equations (ODEs) in PyTorch.

Getting started
----------------------------------------------------

Before starting, you need to build the dynamical systems based on the data types you are working with. Given different system dynamics equations you may need to modify the update function codes in the recurrent layers. The components, in other words, model sub-units, identified from the dynamical systems will be fed into the msDyNODE as inputs when creating the model. In this repository, the models utilized are the firing rate model (for firing rate signals) and the Jasen-Rit model (for local field potential signals). Please refer to the reference for more details. 

Training and Evaluating a msDyNODE model are all included in the msDyNODE.py file.
----------------------------------------------------

Contact
----------------------------------------------------

File any issues with the [issue tracker](https://github.com/DerekYJC/NBGNet/issues). For acy questions or problems, this code is maintained by [@DerekYJC](https://github.com/DerekYJC).

## Reference

- Y. J. Chang, Y. I. Chen, H. M. Stealey, Y. Zhao, H. Y. Lu, E. Contreras-Hernandez, M. N. Baker, E. Castillo, H. C. Yeh, & S. R. Santacruz (2024). [Multiscale effective connectivity analysis of brain activity using neural ordinary differential equations](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0314268), PLoS ONE 19(12): e0314268.
