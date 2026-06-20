# 10. ELBO Optimization and Expectation Maximization

> Core theme: Learn a latent variable model by maximizing the **Evidence Lower Bound (ELBO)**.

---

# 1. Where We Are in the Story

We start from a latent variable model:

$$
P_\theta(x)
=
\int P_\theta(x,z)\,dz
$$

where:

- $x$ is the observed variable
- $z$ is the hidden (latent) variable

The overall goal is still the same:

$$
P_\theta \approx P_X
$$

where:

- $P_X$ is the true data distribution
- $P_\theta$ is the model distribution

---

# 2. Intuition First

Imagine you are a detective.

You observe evidence $X$, but the true cause $Z$ remains hidden.

Examples:

| Observed Data $X$ | Hidden Variable $Z$ |
|------------------|-------------------|
| Face image | Pose, age, lighting |
| Customer purchases | Customer type |
| Speech signal | Phoneme sequence |
| Medical measurements | Disease category |

The challenge is to learn:

1. Model parameters $\theta$
2. Hidden causes $Z$

simultaneously.

---

# 3. From KL Divergence to Maximum Likelihood

Prathosh starts from the objective:

$$
\theta^*
=
\arg\min_\theta
D_{KL}\!\left(P_X \,\|\, P_\theta\right)
$$

Expanding KL divergence:

$$
D_{KL}\!\left(P_X \,\|\, P_\theta\right)
=
\int
P_X(x)
\log
\frac{P_X(x)}
     {P_\theta(x)}
\,dx
$$

Further expansion gives:

$$
D_{KL}\!\left(P_X \,\|\, P_\theta\right)
=
\int P_X(x)\log P_X(x)\,dx
-
\int P_X(x)\log P_\theta(x)\,dx
$$

The first term does not depend on $\theta$.

Therefore:

$$
\theta^*
=
\arg\max_\theta
\mathbb{E}_{P_X}
\big[
\log P_\theta(x)
\big]
$$

Thus minimizing KL divergence is equivalent to maximizing expected log-likelihood.

---

# 4. Key Insight

> Minimizing KL divergence is equivalent to maximizing log-likelihood.

This procedure is called:

## Maximum Likelihood Estimation (MLE)

---

# 5. The New Problem in Latent Variable Models

For a latent variable model:

$$
P_\theta(x)
=
\int P_\theta(x,z)\,dz
$$

the log-likelihood becomes:

$$
\log P_\theta(x)
=
\log
\left(
\int P_\theta(x,z)\,dz
\right)
$$

The logarithm sits outside the integral.

This is the core computational challenge.

---

# 6. Introducing the Variational Distribution

Introduce an auxiliary distribution:

$$
Q(z \mid x)
$$

Think of it as a guess about the hidden cause.

Multiply and divide by $Q(z|x)$:

$$
P_\theta(x,z)
=
P_\theta(x,z)
\frac{Q(z|x)}
     {Q(z|x)}
$$

Nothing changes mathematically, but it enables a useful reformulation.

---

# 7. Rewriting the Likelihood

Substituting:

$$
\log P_\theta(x)
=
\log
\left(
\int
Q(z|x)
\frac{P_\theta(x,z)}
     {Q(z|x)}
\,dz
\right)
$$

Recognize that:

$$
\int
Q(z|x)f(z)\,dz
=
\mathbb{E}_{Q(z|x)}
[f(z)]
$$

Therefore:

$$
\log P_\theta(x)
=
\log
\left(
\mathbb{E}_{Q(z|x)}
\left[
\frac{P_\theta(x,z)}
     {Q(z|x)}
\right]
\right)
$$

---

# 8. Jensen's Inequality

For any random variable $X$:

$$
\log \mathbb{E}[X]
\ge
\mathbb{E}[\log X]
$$

Applying Jensen's inequality:

$$
\log P_\theta(x)
\ge
\mathbb{E}_{Q(z|x)}
\left[
\log
\frac{P_\theta(x,z)}
     {Q(z|x)}
\right]
$$

---

# 9. Evidence Lower Bound (ELBO)

Define:

$$
J_\theta(Q)
=
\mathbb{E}_{Q(z|x)}
\left[
\log
\frac{P_\theta(x,z)}
     {Q(z|x)}
\right]
$$

Then:

$$
\log P_\theta(x)
\ge
J_\theta(Q)
$$

This quantity is called the **Evidence Lower Bound (ELBO)**.

---

# 10. Why Is It Called Evidence Lower Bound?

The likelihood:

$$
P_\theta(x)
$$

is often called the **evidence**, because it measures how well the model explains observed data.

Since ELBO is a lower bound on the log evidence:

$$
\text{ELBO}
=
\text{Evidence Lower Bound}
$$

---

# 11. The Fundamental Optimization Problem

Originally:

$$
\max_\theta
\log P_\theta(x)
$$

After introducing ELBO:

$$
(\theta^*,Q^*)
=
\arg\max_{\theta,Q}
J_\theta(Q)
$$

ELBO depends on:

1. Model parameters $\theta$
2. Variational distribution $Q(z|x)$

Both must be optimized.

---

# 12. Variational Latent Posterior

Prathosh refers to:

$$
Q(z|x)
$$

as the **variational latent posterior**.

Why?

- It is a probability distribution.
- It approximates the true posterior.
- It is optimized during learning.
- Optimization over functions/distributions belongs to variational calculus.

---

# 13. The Most Important Result

Every latent-variable generative model ultimately solves:

$$
(\theta^*,Q^*)
=
\arg\max_{\theta,Q}
J_\theta(Q)
$$

This is the foundational optimization problem behind:

- Gaussian Mixture Models (GMMs)
- Variational Autoencoders (VAEs)
- Diffusion models

---

# 14. Example: Gaussian Mixture Model (GMM)

Let:

$$
Z \in \{1,2,\ldots,M\}
$$

be a discrete latent variable.

The model becomes:

$$
P_\theta(x)
=
\sum_{j=1}^{M}
P_\theta(z=j)
P_\theta(x|z=j)
$$

For a GMM:

$$
P_\theta(z=j)
=
\alpha_j
$$

and

$$
P_\theta(x|z=j)
=
\mathcal{N}
\left(
x;\mu_j,\Sigma_j
\right)
$$

Therefore:

$$
P_\theta(x)
=
\sum_{j=1}^{M}
\alpha_j
\,
\mathcal{N}
\left(
x;\mu_j,\Sigma_j
\right)
$$

This is a weighted mixture of Gaussians.

---

# 15. Parameters of a GMM

The parameter set is:

$$
\theta
=
\{
\alpha_j,\mu_j,\Sigma_j
\}_{j=1}^{M}
$$

subject to:

$$
\alpha_j \ge 0
$$

and

$$
\sum_{j=1}^{M}
\alpha_j
=
1
$$

---

# 16. Expectation Maximization (EM)

EM alternates between optimizing:

- $Q$
- $\theta$

---

## E-Step

Fix $\theta$ and optimize:

$$
Q^{(t+1)}
=
\arg\max_Q
J_{\theta^{(t)}}(Q)
$$

---

## M-Step

Fix $Q$ and optimize:

$$
\theta^{(t+1)}
=
\arg\max_\theta
J_\theta
\Big(
Q^{(t+1)}
\Big)
$$

---

# 17. EM in Words

1. Start with an initial guess for $\theta$
2. Infer latent assignments (E-step)
3. Update model parameters (M-step)
4. Repeat until convergence

This alternating optimization is the EM algorithm.

---

# 18. Why EM Works

EM guarantees that likelihood never decreases:

$$
L^{(t+1)}
\ge
L^{(t)}
$$

Thus each iteration improves or preserves the objective.

EM does not guarantee a global optimum, but guarantees monotonic improvement.

---

# 19. Why EM Works for GMMs

The posterior:

$$
P_\theta(z|x)
$$

can be computed exactly.

Using Bayes' rule:

$$
P_\theta(z|x)
=
\frac{
P_\theta(x|z)
P_\theta(z)
}{
P_\theta(x)
}
$$

Since all terms are known in a GMM, the E-step is tractable.

---

# 20. Solving the M-Step

After computing the posterior, maximize ELBO with respect to:

$$
\theta
=
\{
\alpha_j,
\mu_j,
\Sigma_j
\}
$$

This is a standard optimization problem:

1. Differentiate
2. Set gradients to zero
3. Solve for optimal parameters

---

# 21. Why EM Eventually Fails

EM requires:

$$
P_\theta(z|x)
$$

to be computable.

This is true for simple models such as GMMs.

For deep neural latent-variable models, the posterior is usually intractable.

Then the E-step becomes impossible to compute exactly.

---

# 22. The Big Question

> How do we learn latent-variable models when
> $P_\theta(z|x)$ is intractable?

This question motivates Variational Autoencoders.

---

# 23. Bridge to Variational Autoencoders (VAEs)

VAEs approximate:

$$
P_\theta(z|x)
$$

using neural networks.

Instead of computing the exact posterior, they learn an approximation:

$$
Q_\phi(z|x)
\approx
P_\theta(z|x)
$$

This becomes the foundation of modern deep latent-variable models.

---

# Final Takeaway

A latent-variable model starts from:

$$
P_\theta(x)
=
\int P_\theta(x,z)\,dz
$$

and learns by maximizing the ELBO:

$$
(\theta^*,Q^*)
=
\arg\max_{\theta,Q}
J_\theta(Q)
$$

For simple models like GMMs, EM can optimize this objective because the posterior is tractable.

For deep models like VAEs and diffusion models, the posterior is generally intractable, requiring learned neural approximations.

This naturally leads to the next topic: **Variational Autoencoders (VAEs)**.
