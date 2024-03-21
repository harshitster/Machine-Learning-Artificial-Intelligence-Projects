# Variational Autoencoder

A Variational Autoencoder (VAE) is a type of generative model that learns to encode and decode data. It consists of an encoder network that maps input data into a latent space representation and a decoder network that reconstructs the input data from this latent space representation. Lets know more about VAEs. 

- The training of a VAE involves minimizing a lower bound on the marginal likelihood of the observed data. This lower bound is known as the Evidence
Lower Bound.
    - \text{ELBO}(q) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}[q(z|x) || p(z)]
        - where, 
            - q(z|x) is a variational approximation to the true posterior of the latent variables given the data. True Posteriors involve handling an intractable integral to arrive at the latent, hence a statistical inference problem is converted to a optimization problem where a simple distribution like a Gaussian Distribution is optimized to the true posterior distribution, p(z|x).
            ![](kl_q(z|x)_p(z|x).jpg)

            - The term log p(x) is constant and hence ELBO 

        \begin{aligned}
        K L\left(q\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right) \| p\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)\right) & =\int_{\mathbf{z}} q\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right) \log \frac{q\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)}{p\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)} d \mathbf{z} \\
        & =\mathbb{E}_{\| \mid}\left[\log q\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)\right]-\mathbb{E}_{\| \mid}\left[\log p\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)\right] \\
        & =\mathbb{E}_{\| 1}\left[\log q\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)\right]-\mathbb{E}_{\| \mid}\left[\log p\left(\mathbf{x}^{(i)}, \mathbf{z}\right)\right]+E_q\left[\log p\left(\mathbf{x}^{(i)}\right)\right] \\
        & =\mathbb{E}_{\| \mid}\left[\log q\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)\right]-\mathbb{E}_{\| \mid}\left[\log p\left(\mathbf{x}^{(i)}, \mathbf{z}\right)\right]+\log p\left(\mathbf{x}^{(i)}\right) \\
        & =-\operatorname{ELBO}+\log p\left(\mathbf{x}^{(i)}\right)
        \end{aligned}