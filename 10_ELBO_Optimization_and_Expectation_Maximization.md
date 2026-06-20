>
> Core theme: learn a latent variable model by maximizing the Evidence Lower Bound (ELBO).

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
- $z$ is the hidden or latent variable

The overall goal is still the same:

$$
P_\theta \approx P_X
$$

where $P_X$ is the true data distribution and $P_\theta$ is the model distribution.

---

# 2. Intuition First

Imagine you are a detective.

You see evidence $X$, but the real cause $Z$ is hidden.

Examples:

| Observed data $X$ | Hidden variable $Z$ |
|---|---|
| Face image | pose, age, lighting |
| Customer purchases | customer type |
| Speech signal | phoneme sequence |
| Medical measurements | disease category |

The challenge is to learn:

1. the model parameters $\theta$
2. the hidden causes $Z$

at the same time.

---

# 3. From KL Divergence to Maximum Likelihood

Prathosh starts from the divergence objective:

$$
\theta^*
=
\arg\min_\theta
D_{KL}(P_X \| P_\theta)
$$

Expanding the KL divergence:

$$
D_{KL}(P_X \| P_\theta)
=
\int
P_X(x)
\log
\frac{P_X(x)}{P_\theta(x)}
\,dx
$$

Expanding further:

$$
D_{KL}(P_X \| P_\theta)
=
\int P_X(x)\log P_X(x)\,dx
-
\int P_X(x)\log P_\theta(x)\,dx
$$

The first term does not depend on $\theta$. Therefore:

$$
\theta^*
=
\arg\max_\theta
\mathbb{E}_{P_X}\left[\log P_\theta(x)\right]
$$

So minimizing KL divergence is equivalent to maximizing the expected log likelihood.

---

# 4. Key Insight

Prathosh's main message here is:

> Minimizing KL divergence is the same as maximizing log likelihood.

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

so the log likelihood becomes:

$$
\log P_\theta(x)
=
\log\left(\int P_\theta(x,z)\,dz\right)
$$

The logarithm is outside the integral. That is the core difficulty.

We know how to estimate expectations from samples, but not a log of an expectation in this direct form.

---

# 6. Introducing the Variational Distribution

Prathosh introduces a distribution over the latent variable:

$$
Q(z|x)
$$

Think of it as a guess about the hidden cause.

We multiply and divide by $Q(z|x)$:

$$
P_\theta(x,z)
=
P_\theta(x,z)\frac{Q(z|x)}{Q(z|x)}
$$

This does not change the expression, but it helps us rewrite the likelihood.

---

# 7. Rewriting the Likelihood

We obtain:

$$
\log P_\theta(x)
=
\log\left(
\int
Q(z|x)
\frac{P_\theta(x,z)}{Q(z|x)}
\,dz
\right)
$$

Now recognize that:

$$
\int Q(z|x)f(z)\,dz
=
\mathbb{E}_{Q(z|x)}[f(z)]
$$

Therefore:

$$
\log P_\theta(x)
=
\log\left(
\mathbb{E}_{Q(z|x)}
\left[
\frac{P_\theta(x,z)}{Q(z|x)}
\right]
\right)
$$

---

# 8. Jensen's Inequality

Prathosh now uses Jensen's inequality.

For the logarithm:

$$
\log \mathbb{E}[X]
\ge
\mathbb{E}[\log X]
$$

Applying Jensen gives:

$$
\log P_\theta(x)
\ge
\mathbb{E}_{Q(z|x)}
\left[
\log\frac{P_\theta(x,z)}{Q(z|x)}
\right]
$$

---

# 9. Evidence Lower Bound (ELBO)

The right-hand side is called the Evidence Lower Bound, or ELBO.

Define:

$$
J_\theta(Q)
=
\mathbb{E}_{Q(z|x)}
\left[
\log\frac{P_\theta(x,z)}{Q(z|x)}
\right]
$$

Then:

$$
\log P_\theta(x)
\ge
J_\theta(Q)
$$

So ELBO is a lower bound on the log likelihood.

---

# 10. Why Is It Called Evidence Lower Bound?

The likelihood $P_\theta(x)$ is also called the evidence, because it tells us how well the model explains the observed data.

Since we derived a lower bound on that quantity, it is called the Evidence Lower Bound.

So:

$$
\text{ELBO}
=
\text{Evidence Lower Bound}
$$

---

# 11. The Fundamental Optimization Problem

Originally, the goal was:

$$
\max_\theta \log P_\theta(x)
$$

Now the problem becomes:

$$
(\theta^*,Q^*)
=
\arg\max_{\theta,Q}
J_\theta(Q)
$$

Notice that ELBO depends on two things:

1. the model parameters $\theta$
2. the variational distribution $Q(z|x)$

So both must be learned simultaneously.

---

# 12. Variational Latent Posterior

Prathosh calls $Q(z|x)$ the variational latent posterior.

Why?

Because:

- it is a distribution
- it is optimized
- optimization over functions or distributions belongs to variational calculus

So the variational distribution is our approximation to the true posterior over the latent variable.

---

# 13. The Most Important Result

Every latent variable generative model solves:

$$
(\theta^*,Q^*)
=
\arg\max_{\theta,Q}
J_\theta(Q)
$$

This is the foundational optimization problem behind latent variable models.

It applies to:

- Gaussian Mixture Models
- Variational Autoencoders
- Diffusion models

---

# 14. Example: Gaussian Mixture Model

Now Prathosh gives a classical machine learning example.

Let the latent variable $Z$ be discrete:

$$
Z \in \{1,2,\ldots,M\}
$$

Then the model distribution is:

$$
P_\theta(x)
=
\sum_{j=1}^{M}
P_\theta(z=j)P_\theta(x|z=j)
$$

For a Gaussian mixture model:

$$
P_\theta(z=j)=\alpha_j
$$

and

$$
P_\theta(x|z=j)=\mathcal{N}(x;\mu_j,\Sigma_j)
$$

Therefore:

$$
P_\theta(x)
=
\sum_{j=1}^{M}
\alpha_j\,\mathcal{N}(x;\mu_j,\Sigma_j)
$$

That is why it is called a Gaussian Mixture Model.

---

# 15. Parameters of a GMM

We need to estimate:

$$
\theta
=
\{\alpha_j,\mu_j,\Sigma_j\}_{j=1}^{M}
$$

subject to:

$$
\alpha_j \ge 0
$$

and

$$
\sum_{j=1}^{M}\alpha_j = 1
$$

---

# 16. Expectation Maximization (EM)

Prathosh then introduces the Expectation Maximization algorithm.

The idea is to alternate between optimizing $Q$ and optimizing $\theta$.

---

## E-Step

Fix $\theta$ and optimize the variational distribution:

$$
Q^{(t+1)}
=
\arg\max_Q
J_{\theta^{(t)}}(Q)
$$

---

## M-Step

Fix $Q$ and optimize the model parameters:

$$
\theta^{(t+1)}
=
\arg\max_\theta
J_\theta\bigl(Q^{(t+1)}\bigr)
$$

---

# 17. EM in Words

The algorithm works like this:

1. start with an initial guess for $\theta$ and $Q$
2. infer the latent posterior using the E-step
3. update the model parameters using the M-step
4. repeat

This alternating process is EM.

---

# 18. Why EM Works

Prathosh states that EM guarantees the likelihood will never decrease.

So if $L^{(t)}$ is the likelihood at iteration $t$, then:

$$
L^{(t+1)} \ge L^{(t)}
$$

This does not guarantee the global optimum, but it does guarantee monotonic improvement.

---

# 19. Why EM Works for GMM

EM works nicely for Gaussian mixture models because the posterior:

$$
P_\theta(z|x)
$$

can be computed analytically.

Using Bayes' rule:

$$
P_\theta(z|x)
=
\frac{P_\theta(x|z)P_\theta(z)}{P_\theta(x)}
$$

Since every term is known for a GMM, the posterior is tractable.

So the E-step is feasible.

---

# 20. Solving the M-Step

Once we have the optimal posterior, we maximize the ELBO with respect to the parameters:

$$
\theta = \{\alpha_j,\mu_j,\Sigma_j\}
$$

This is standard optimization: differentiate, set gradients to zero, and solve.

---

# 21. Why EM Eventually Fails

EM works only when the posterior $P_\theta(z|x)$ can be computed.

That is true for simple models like GMMs.

But for neural-network-based latent variable models, the posterior is usually intractable.

Then the E-step fails.

---

# 22. The Big Question

Prathosh ends with the important question:

> How do we learn latent variable models when $P_\theta(z|x)$ is unknown?

That is the motivation for Variational Autoencoders.

---

# 23. Bridge to VAEs

VAEs solve the problem by approximating the posterior $P_\theta(z|x)$ using neural networks.

Instead of computing the exact posterior, they learn an approximation.

This becomes the foundation of modern deep latent variable models.

---

# Final Takeaway

A latent variable model starts from:

$$
P_\theta(x)
=
\int P_\theta(x,z)\,dz
$$

and learns by maximizing ELBO:

$$
(\theta^*,Q^*)
=
\arg\max_{\theta,Q}
J_\theta(Q)
$$

For simple models like GMMs, EM can optimize this exactly enough because the posterior is tractable.

For deep models like VAEs and diffusion models, the posterior is not tractable, so we need neural approximations.

That is the transition to the next topic.
