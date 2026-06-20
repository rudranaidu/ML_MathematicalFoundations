# 10_ELBO_Optimization_and_Expectation_Maximization

> Based on Prof. Prathosh A.P's lecture on Generative AI Principles and Applications.

## Core Theme

How do we learn a latent variable model?

The answer is:

**Maximize the Evidence Lower Bound (ELBO).**

---

# 1. Where We Are in the Story

In the previous note, we introduced latent variable models.

A latent variable model assumes:

$$
P_\theta(x)
=
\int P_\theta(x,z)\,dz
$$

where:

- \(x\) = observed variable
- \(z\) = hidden (latent) variable

The objective remains unchanged:

$$
P_\theta \approx P_X
$$

where:

- \(P_X\) = true data distribution
- \(P_\theta\) = model distribution

---

# 2. Intuition First

Imagine you are a detective.

You observe evidence \(X\), but the true cause \(Z\) is hidden.

Examples:

| Observed Data \(X\) | Hidden Variable \(Z\) |
|---------------------|-----------------------|
| Face Image | Pose, Age, Lighting |
| Customer Purchases | Customer Type |
| Speech Signal | Phoneme Sequence |
| Medical Measurements | Disease Category |

The challenge is:

1. Learn model parameters \(\theta\)
2. Learn hidden causes \(Z\)

simultaneously.

---

# 3. From KL Divergence to Maximum Likelihood

Prathosh starts from:

$$
\theta^*
=
\arg\min_\theta
D_{KL}(P_X \,\|\, P_\theta)
$$

Expanding the KL divergence:

$$
D_{KL}(P_X \,\|\, P_\theta)
=
\int
P_X(x)
\log
\frac{P_X(x)}
     {P_\theta(x)}
\,dx
$$

Expanding further:

$$
D_{KL}(P_X \,\|\, P_\theta)
=
\int P_X(x)\log P_X(x)\,dx
-
\int P_X(x)\log P_\theta(x)\,dx
$$

The first term does not depend on \(\theta\).

Therefore:

$$
\theta^*
=
\arg\max_\theta
\mathbb{E}_{P_X}
\left[
\log P_\theta(x)
\right]
$$

---

# 4. Key Insight

Prathosh emphasizes:

> Minimizing KL Divergence is equivalent to Maximizing Log Likelihood.

This procedure is called:

## Maximum Likelihood Estimation (MLE)

---

# 5. The New Problem

For a latent variable model:

$$
P_\theta(x)
=
\int P_\theta(x,z)\,dz
$$

Therefore:

$$
\log P_\theta(x)
=
\log
\left(
\int P_\theta(x,z)\,dz
\right)
$$

The logarithm sits outside the integral.

This is the core difficulty.
