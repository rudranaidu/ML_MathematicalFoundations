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

# 9. Evidence Lower Bound (ELBO)

The right-hand side is given a special name.

Define:

```math
J_\theta(Q)
=
\mathbb{E}_{Q(z|x)}
\left[
\log
\frac{P_\theta(x,z)}
{Q(z|x)}
\right]
```

Then:

```math
\log P_\theta(x)
\ge
J_\theta(Q)
```

The quantity `Jθ(Q)` is called the **Evidence Lower Bound (ELBO)**.
