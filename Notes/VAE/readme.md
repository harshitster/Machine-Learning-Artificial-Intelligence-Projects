# Variational Autoencoder

A Variational Autoencoder (VAE) is a type of generative model that learns to encode and decode data. It consists of an encoder network that maps input data into a latent space representation and a decoder network that reconstructs the input data from this latent space representation. Lets know more about VAEs. 

- q(z|x) is a variational approximation to the true posterior of the latent variables given the data. True Posteriors involve handling an intractable integral to arrive at the latent, hence a statistical inference problem is converted to a optimization problem where a simple distribution like a Gaussian Distribution is optimized to the true posterior distribution, p(z|x).
![](kl_q(z|x)_p(z|x).jpg)

\[
\begin{aligned}
K L\left(q(\mathbf{z} \mid \mathbf{x}^{(i)}) \| p(\mathbf{z} \mid \mathbf{x}^{(i)})\right) & =\int_{\mathbf{z}} q(\mathbf{z} \mid \mathbf{x}^{(i)}) \log \frac{q(\mathbf{z} \mid \mathbf{x}^{(i)})}{p(\mathbf{z} \mid \mathbf{x}^{(i)})} d \mathbf{z} \\
& =\mathbb{E}_{\| \mid}[\log q(\mathbf{z} \mid \mathbf{x}^{(i)})]-\mathbb{E}_{\| \mid}[\log p(\mathbf{z} \mid \mathbf{x}^{(i)})] \\
& =\mathbb{E}_{\| 1}[\log q(\mathbf{z} \mid \mathbf{x}^{(i)})]-\mathbb{E}_{\| \mid}[\log p(\mathbf{x}^{(i)}, \mathbf{z})]+E_q[\log p(\mathbf{x}^{(i)})] \\
& =\mathbb{E}_{\| \mid}[\log q(\mathbf{z} \mid \mathbf{x}^{(i)})]-\mathbb{E}_{\| \mid}[\log p(\mathbf{x}^{(i)}, \mathbf{z})]+\log p(\mathbf{x}^{(i)}) \\
& =-\operatorname{ELBO}+\log p(\mathbf{x}^{(i)})
\end{aligned}
\]

- The term log p(x) is constant and hence ELBO 
