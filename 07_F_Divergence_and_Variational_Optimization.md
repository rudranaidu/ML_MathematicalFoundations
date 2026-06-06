# Mathematical Foundations of Machine Learning
# 07 - F-Divergence and Variational Optimization

---

# 1. The Fundamental Problem

We want:

$$
P_\theta \approx P_X
$$

But how do we measure:

"how close"

two distributions are?

---

# Real World Analogy

Suppose:

- one map is Google Maps
- another map is Apple Maps

How similar are they?

Need a distance measure.

Same for probability distributions.

---

# 2. Divergence

A divergence measures difference between distributions.

Notation:

$$
D(P_X||P_\theta)
$$

Smaller value:

```text
more similar
```

Larger value:

```text
more different
```

---

# 3. F-Divergence

General family:

$$
D_f(P_X||P_\theta)
=
\int
P_\theta(x)
f\left(
\frac{P_X(x)}
{P_\theta(x)}
\right)
dx
$$

---

# Intuition

Compare densities point-by-point.

Compute ratio:

$$
\frac{P_X(x)}
{P_\theta(x)}
$$

If ratio close to 1:

good match.

If very different:

bad match.

---

# 4. Requirements on f

Function:

$$
f:\mathbb R^+\rightarrow\mathbb R
$$

must satisfy:

$$
f(1)=0
$$

and be convex.

---

# Convex Function Intuition

Think:

```text
bowl shape
```

Examples:

$$
x^2
$$

or

$$
e^x
$$

Convexity ensures:

optimization behaves nicely.

---

# 5. Why Is Divergence Always Positive?

Property:

$$
D_f(P_X||P_\theta)\ge0
$$

---

# Intuition

Distance cannot be negative.

Like:

- physical distance
- road distance

Always non-negative.

---

# 6. KL Divergence

Most famous divergence.

Obtained using:

$$
f(u)=u\log u
$$

Result:

$$
D_{KL}
(P_X||P_\theta)
=
\int
P_X(x)
\log
\frac
{P_X(x)}
{P_\theta(x)}
dx
$$

---

# Real World Intuition

KL divergence measures:

extra surprise

caused by using the wrong distribution.

---

# Example

Suppose:

Real coin:

```text
90% heads
```

Model:

```text
50% heads
```

Model will be frequently surprised.

KL divergence becomes large.

---

# 7. Density Functions

Density:

$$
p(x)
$$

is NOT probability.

---

# Common Mistake

Wrong:

```text
p(170cm)=0.3
means probability
```

Correct:

Density is concentration.

---

# Example

Human heights.

Probability:

$$
P(165<X<175)
$$

obtained by integrating density.

---

# 8. Expectation

Given:

$$
g(X)
$$

Expectation:

$$
E[g(X)]
=
\int
P_X(x)
g(x)
dx
$$

---

# Real World Analogy

Expectation means:

average outcome if experiment repeated infinitely many times.

---

# Example

Dice:

$$
E[X]=3.5
$$

Not an actual roll.

Long-term average.

---

# 9. Why Expectations Matter

We never know:

$$
P_X
$$

exactly.

We only have samples:

$$
x_1,\dots,x_n
$$

Therefore:

expectations become sample averages.

This enables practical training.

---

# 10. Variational Optimization

Directly computing divergence is difficult.

Instead:

transform problem into optimization over functions.

This is called:

Variational Optimization.

---

# Real World Analogy

Measuring mountain height directly is difficult.

Instead:

measure shadows.

Indirect measurement.

Variational methods do the same.

---

# 11. Conjugate Functions

Every convex function has:

a convex conjugate.

This allows:

rewriting difficult optimization problems into easier forms.

---

# Why Important?

GANs

VAEs

Diffusion Models

all rely on this idea.

---

# Final Takeaway

Generative AI training is:

1. Define divergence

$$
D(P_X||P_\theta)
$$

2. Rewrite using variational methods

3. Optimize

4. Learn:

$$
P_\theta \approx P_X
$$

This is the mathematical foundation of modern generative modeling.
