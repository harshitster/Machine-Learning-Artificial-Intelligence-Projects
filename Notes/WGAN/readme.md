# WGAN 

What are Probability Distributions and Probability Denstity Functions? 
- A Probability Distribution is list of data samples and the associated probabilities to occur in the dataset. 
- Probability Denstity Functions represent the continuous the probability distributions in the mathematical form.

It can be inferred from the above statements that to learn a probability distribution, learning the density function might suffice.
But this will not learning of sub-distribution in the low dimensional manifold that are embedded in higher dimensions. 

Rather than estimating the density of the actual distribution, random variable 'Z' with fixed distributions p(z), typically Gaussian Distribution, 
is established and passed through the parametric function (neural network) that directly generates samples following a certain distribution.
First of all, unlike densities, this approach can represent distribu- tions confined to a low dimensional manifold. Second, the ability to easily generate samples is often more useful than knowing the numerical value of the density (for example in image superresolution or semantic segmentation when considering the conditional distribution of the output image given the input image).

Variational AutoEncoders and Generative Adversarial Networks make use of such approach to generate new data samples. 