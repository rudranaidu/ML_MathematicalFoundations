# 10. ELBO Optimization and Expectation Maximization

> **Core Theme:** Learn a latent variable model by maximizing the **Evidence Lower Bound (ELBO)**.

---

# 1. Where We Are in the Story

We start from a latent variable model:

```math
P_\theta(x) = \int P_\theta(x,z)\,dz
```

where:

* `x` is the observed variable
* `z` is the hidden (latent) variable

The overall goal is still the same:

```math
P_\theta \approx P_X
```

where:

* `P_X` is the true data distribution
* `P_\theta` is the model distribution

---

# 2. Intuition First

Imagine you are a detective.

You observe evidence `X`, but the true cause `Z` remains hidden.

Examples:

| Observed Data X      | Hidden Variable Z   |
| -------------------- | ------------------- |
| Face image           | Pose, age, lighting |
| Customer purchases   | Customer type       |
| Speech signal        | Phoneme sequence    |
| Medical measurements | Disease category    |

The challenge is to learn:

1. Model parameters `θ`
2. Hidden causes `Z`

simultaneously.

---

# 3. From KL Divergence to Maximum Likelihood

Prathosh starts from the divergence objective:

```math
\theta^* = \arg\min_\theta D_{KL}(P_X \| P_\theta)
```

Expanding KL divergence:

```math
D_{KL}(P_X \| P_\theta)
=
\int P_X(x)\log\frac{P_X(x)}{P_\theta(x)}\,dx
```

Expanding further:

```math
D_{KL}(P_X \| P_\theta)
=
\int P_X(x)\log P_X(x)\,dx
-
\int P_X(x)\log P_\theta(x)\,dx
```

The first term does not depend on `θ`.

Therefore:

```math
\theta^*
=
\arg\max_\theta
\mathbb{E}_{P_X}[\log P_\theta(x)]
```

Thus minimizing KL divergence is equivalent to maximizing expected log-likelihood.

---

# 4. Key Insight

Prathosh's main message is:

> Minimizing KL divergence is equivalent to maximizing log-likelihood.

This procedure is called:

## Maximum Likelihood Estimation (MLE)

---

# 5. The New Problem in Latent Variable Models

For a latent variable model:

```math
P_\theta(x) = \int P_\theta(x,z)\,dz
```

the log-likelihood becomes:

```math
\log P_\theta(x)
=
\log\left(
\int P_\theta(x,z)\,dz
\right)
```

The logarithm sits outside the integral.

This is the core computational challenge.

We know how to estimate expectations from samples, but a logarithm of an expectation is much harder to optimize directly.

---

# 6. Introducing the Variational Distribution

Prathosh introduces an auxiliary distribution:

```math
Q(z|x)
```

Think of it as a guess about the hidden cause.

Multiply and divide by `Q(z|x)`:

```math
P_\theta(x,z)
=
P_\theta(x,z)
\frac{Q(z|x)}{Q(z|x)}
```

Nothing changes mathematically.

However, this trick allows us to rewrite the likelihood into a more useful form.

---

# 7. Rewriting the Likelihood

Substituting the variational distribution:

```math
\log P_\theta(x)
=
\log\left(
\int
Q(z|x)
\frac{P_\theta(x,z)}
{Q(z|x)}
\,dz
\right)
```

Recognize the expectation identity:

```math
\int Q(z|x)f(z)\,dz
=
\mathbb{E}_{Q(z|x)}[f(z)]
```

Therefore:

```math
\log P_\theta(x)
=
\log\left(
\mathbb{E}_{Q(z|x)}
\left[
\frac{P_\theta(x,z)}
{Q(z|x)}
\right]
\right)
```

Now the expression is written as a logarithm of an expectation.

This is exactly where Jensen's inequality becomes useful.

---

# 8. Jensen's Inequality

For a concave function such as the logarithm:

```math
\log \mathbb{E}[X]
\ge
\mathbb{E}[\log X]
```

Applying Jensen's inequality:

```math
\log P_\theta(x)
\ge
\mathbb{E}_{Q(z|x)}
\left[
\log
\frac{P_\theta(x,z)}
{Q(z|x)}
\right]
```

This inequality is one of the most important steps in the derivation of ELBO.

It converts an intractable optimization problem into a tractable lower bound optimization problem.

---

# 10. Why Is It Called Evidence Lower Bound?

The likelihood

Pθ(x)

is often called the **evidence** because it measures how well the model explains the observed data.

Since we derived a lower bound on the log evidence, the quantity is called the **Evidence Lower Bound (ELBO)**.

ELBO = Evidence Lower Bound

Intuitively:

* Evidence = how well the model explains the observed data.
* Lower Bound = a quantity that is guaranteed to be less than or equal to the true log-likelihood.

Instead of maximizing the difficult quantity

log Pθ(x)

we maximize its lower bound.

---

# 11. The Fundamental Optimization Problem

Originally, our objective was:

maxθ log Pθ(x)

After introducing the ELBO, the optimization problem becomes:

(θ*, Q*) = argmax(θ,Q) Jθ(Q)

Notice that the ELBO depends on two objects:

1. Model parameters θ
2. Variational distribution Q(z|x)

Both must be optimized.

This is the central optimization problem behind latent variable generative models.

---

# 12. Variational Latent Posterior

Prathosh refers to

Q(z|x)

as the **variational latent posterior**.

Why is it called "variational"?

Because:

* It is a probability distribution.
* It approximates the true posterior.
* It is optimized during learning.
* Optimization over distributions belongs to variational calculus.

The goal is:

Q(z|x) ≈ Pθ(z|x)

The variational distribution acts as our approximation to the true posterior.

---

# 13. The Most Important Result

Every latent-variable generative model ultimately solves:

(θ*, Q*) = argmax(θ,Q) Jθ(Q)

This is the foundational optimization problem behind:

* Gaussian Mixture Models (GMMs)
* Variational Autoencoders (VAEs)

Many modern latent-variable models are built upon the same variational principle.

---

# 14. Example: Gaussian Mixture Model (GMM)

Let the latent variable be discrete:

Z ∈ {1, 2, ..., M}

Then the model becomes:

Pθ(x) = Σ[j=1→M] Pθ(z=j) Pθ(x|z=j)

For a Gaussian Mixture Model:

Pθ(z=j) = αj

and

Pθ(x|z=j) = N(x; μj, Σj)

Substituting into the model:

Pθ(x) = Σ[j=1→M] αj N(x; μj, Σj)

This is a weighted sum of Gaussian densities.

Hence the name:

**Gaussian Mixture Model.**

---

# 15. Parameters of a GMM

The parameter set is:

θ = {αj, μj, Σj} for j = 1,...,M

The mixing coefficients must satisfy:

αj ≥ 0

and

Σ[j=1→M] αj = 1

These constraints ensure that the mixing coefficients form a valid probability distribution.

Therefore, learning a GMM means estimating:

* Mixing weights αj
* Means μj
* Covariance matrices Σj

for every Gaussian component.

---

# 16. Expectation Maximization (EM)

Prathosh now introduces the **Expectation Maximization (EM)** algorithm.

The key idea is simple:

Instead of optimizing both Q and θ simultaneously, we alternate between them.

EM repeatedly performs:

1. Optimize Q while keeping θ fixed.
2. Optimize θ while keeping Q fixed.

This converts one difficult optimization problem into two simpler optimization problems.

## E-Step (Expectation Step)

Keep the parameters fixed:

θ(t)

and optimize the variational distribution:

Q(t+1) = argmaxQ Jθ(t)(Q)

Interpretation:

Given the current model, estimate the latent variables.

In a GMM, this means computing the probability that each datapoint belongs to each Gaussian component.

## M-Step (Maximization Step)

Keep the variational distribution fixed:

Q(t+1)

and optimize the parameters:

θ(t+1) = argmaxθ Jθ(Q(t+1))

Interpretation:

Given the inferred latent assignments, update the model parameters.

## Intuition Behind EM

Think of clustering customers.

### E-Step

Estimate:

"Which customer belongs to which cluster?"

### M-Step

Update:

"What should the cluster centers and cluster spreads be?"

Then repeat until convergence.

This alternating optimization is the EM algorithm.

