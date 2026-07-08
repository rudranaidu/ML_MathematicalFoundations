# Mathematical Foundations of Machine Learning
# 07 - F-Divergence and Variational Optimization

---

# 1. The Fundamental Problem of Generative Modeling

In generative modeling, our goal is to learn a model distribution

$$
P_\theta
$$

that is as close as possible to the true data distribution

$$
P_X.
$$

Mathematically, we want:

$$
P_\theta \approx P_X.
$$

The natural question is:

> How do we measure the difference between two probability distributions?

This leads us to the concept of **divergence**.

---

# 2. What is a Divergence?

A divergence is a measure of dissimilarity between two probability distributions.

Notation:

$$
D(P_X \,\|\, P_\theta)
$$

Interpretation:

- Small divergence → distributions are similar.
- Large divergence → distributions are very different.

---

## Real-World Analogy

Suppose two weather forecasting systems predict rainfall probabilities.

If both systems produce nearly identical forecasts, the divergence between them is small.

If one predicts sunshine and the other predicts heavy rain, the divergence becomes large.

Similarly, we want the model distribution to behave like the true data distribution.

---

# 3. F-Divergence

A very general family of divergences is called the **F-Divergence Family**.

It is defined as

### F-Divergence

The F-divergence between two distributions is

$$D_f(P_X \| P_\theta)=\int P_\theta(x)\,f\left(\frac{P_X(x)}{P_\theta(x)}\right)\,dx$$

where

$$f:\mathbb{R}^{+}\rightarrow\mathbb{R}$$

is a convex function satisfying

$$f(1)=0$$
---

# 4. Understanding the Formula

The most important quantity inside the divergence is

$$
\frac{P_X(x)}
     {P_\theta(x)}.
$$

This ratio compares:

- what the real world believes,
- what the model believes.

---

## Case 1: Perfect Match

If

$$
P_X(x)=P_\theta(x),
$$

then

$$
\frac{P_X(x)}
     {P_\theta(x)}
=1.
$$

Since

$$
f(1)=0,
$$

the divergence contribution becomes zero.

This is exactly what we want.

---

## Case 2: Model Underestimates

Suppose

$$
P_X(x)=0.8
$$

but

$$
P_\theta(x)=0.2.
$$

Then

$$
\frac{P_X(x)}
     {P_\theta(x)}
=4.
$$

The divergence becomes large.

This means:

> The model is assigning too little probability to something that occurs frequently in reality.

---

## Case 3: Model Overestimates

Suppose

$$
P_X(x)=0.1
$$

but

$$
P_\theta(x)=0.8.
$$

Then the ratio becomes very small.

Again, the divergence becomes large.

This means:

> The model believes in events that rarely occur in the real world.

---

# 5. Why Convex Functions?

The function

$$
f
$$

must be convex.

Convex functions have a bowl-like shape.

Examples:

$$
f(u)=u^2
$$

$$
f(u)=e^u
$$

Convexity ensures:

- non-negative divergences,
- stable optimization,
- useful mathematical guarantees.

---

# 6. Important Property of F-Divergence

For every valid F-divergence,

$$
D_f(P_X \,\|\, P_\theta)
\ge 0.
$$

The divergence can never be negative.

---

## Intuition

Distance between two cities cannot be negative.

Similarly, difference between two distributions cannot be negative.

---

# 7. KL Divergence

The most famous F-divergence is the **Kullback-Leibler (KL) Divergence**.

Choose

$$
f(u)=u\log u.
$$

Substituting into the F-divergence formula yields

$$D_{KL}(P_X || P_\theta)=\int P_X(x)\log\left(\frac{P_X(x)}{P_\theta(x)}\right)dx$$

$$ P_X || P_\theta) pronounced as P X relative to P theta

---

# 8. Intuition Behind KL Divergence

KL divergence measures:
The gap between what you believed and what actually happened
is what KL divergence measures.

## GPS Analogy

Suppose Google Maps says:

Road A is the fastest route.

But reality is:

Road A is blocked.

You keep taking the wrong road.

The cost of using the wrong map becomes large.
---

## Interpretation

Smaller KL divergence means:

- model beliefs closely match reality.

Larger KL divergence means:

- model beliefs differ significantly from reality.

---

# 9. Density Functions

Generative models usually work with probability densities.

A density function is written as

$$
p(x).
$$

---

## Important Warning

Density is NOT probability.

For continuous random variables,

$$
P(X=x)=0.
$$

always.

---

## Example: Human Heights

The value

$$
p(170)
$$

does NOT mean

> probability of a person being exactly 170 cm tall.

Instead, it measures how concentrated probability mass is around 170 cm.

Probability only exists over intervals:

$$
P(165 < X < 175).
$$

---

# 10. Expectation

Suppose

$$
g(X)
$$

is any function of a random variable.

Its expectation is

$$
\mathbb{E}[g(X)]
=
\int
P_X(x)\,
g(x)\,
dx.
$$

---

## Intuition

Expectation is the long-run average outcome of an experiment.

---

## Example: Dice

For a fair die,

$$
\mathbb{E}[X]
=
\frac{1+2+3+4+5+6}{6}
=
3.5.
$$

Notice:

- 3.5 never appears on a die,
- but it represents the long-term average.

---

# 11. Why Expectations Matter in ML

In practice:

We never know

$$
P_X
$$

exactly.

We only have samples:

$$
x_1,x_2,\dots,x_n.
$$

Therefore expectations are approximated using sample averages.

This is the foundation of:

- stochastic gradient descent,
- Monte Carlo estimation,
- variational inference,
- deep learning optimization.

---

# 12. Variational Optimization

Directly computing

$$
D(P_X \,\|\, P_\theta)
$$

is often impossible.

Instead, we rewrite the problem into a form that is easier to optimize.

This approach is called:

# Variational Optimization

---

## Real-World Analogy

Imagine trying to measure the height of a mountain.

Direct measurement is difficult.

Instead, you use:

- shadows,
- angles,
- indirect calculations.

Variational methods use a similar idea.

They transform a difficult problem into an easier one.

---

# 13. Convex Conjugates

A deep result from convex analysis states:

> Every convex function has a corresponding convex conjugate.

This allows difficult divergences to be rewritten into optimization objectives that can be solved using neural networks.

---

# Why This Matters

Many modern generative models rely on this idea:

- GANs
- VAEs
- Variational Inference
- Diffusion Models

---

# 14. The Big Picture

Generative modeling follows four steps:

### Step 1

Define a divergence:

$$
D(P_X \,\|\, P_\theta).
$$

### Step 2

Rewrite it using variational techniques.

### Step 3

Optimize the objective.

### Step 4

Obtain

$$
P_\theta \approx P_X.
$$

---

# Final Takeaway

The central problem of generative modeling is:

$$
P_\theta \approx P_X.
$$

F-divergences provide a mathematical framework for measuring the difference between distributions.

Different choices of divergence lead to different generative models:

| Divergence | Used In |
|------------|----------|
| KL Divergence | VAE |
| Reverse KL | Variational Inference |
| Jensen-Shannon | GAN |
| Wasserstein Distance | WGAN |
| Score Matching | Diffusion Models |

Modern generative AI can be viewed as different ways of solving the same problem:

> Learn a distribution that behaves like the real world.
