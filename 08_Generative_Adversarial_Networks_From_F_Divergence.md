# 1. Where We Are in the Story

Recall the generative modeling goal:

$$
D = \{x_1, x_2, \ldots, x_N\}
$$

where the dataset consists of samples drawn from an unknown true distribution

$$
P_X.
$$

We want to learn a model distribution

$$
P_\theta
$$

such that

$$
P_\theta \approx P_X,
$$

and then sample from the learned model.

---

# 2. The Central Problem

We want a way to measure how close the model distribution is to the true distribution:

$$
P_X \quad \text{vs.} \quad P_\theta
$$

That comparison is done using a divergence measure.

Examples include:

- KL divergence
- Reverse KL divergence
- Jensen-Shannon divergence
- Pearson divergence
- Chi-square divergence

These belong to the family of **F-divergences**.

---

# 3. Why F-Divergence Is Difficult

The original divergence depends explicitly on the densities of both distributions.

But in practice we do **not** know the densities of either:

$$
P_X \quad \text{or} \quad P_\theta.
$$

We only have samples.

So the direct computation of the divergence is not possible.

---

# 4. The Key Machine Learning Trick

A central idea in machine learning is that expectations can be estimated from samples.

If

$$
X \sim P_X,
$$

then

$$
\mathbb{E}[h(X)]
$$

can be approximated by the sample average

$$
\frac{1}{N} \sum_{i=1}^N h(x_i).
$$

This is the law of large numbers in action.

## Intuition

If you want the average height of a population, you do not need the full distribution. You measure many people and average their heights.

That is exactly what this approximation does.

---

# 5. Why Convex Conjugates Are Introduced

The goal is to rewrite the F-divergence into a form involving expectations, because expectations can be estimated from samples.

Convexity theory gives us a way to do that using the **convex conjugate**.

This leads to the variational form:

$$
D_f(P_X, P_\theta)
\ge
\sup_T
\left(
\mathbb{E}_{P_X}[T(x)]
-
\mathbb{E}_{P_\theta}[F^*(T(x))]
\right),
$$

where:

- $F^*$ is the convex conjugate of the chosen convex function $F$
- $T(x)$ is a learnable function over the input space

This is the bridge from abstract divergence theory to something computable.

---

# 6. What “Variational” Means

The word **variational** means we turn the problem into an optimization problem.

Instead of computing the divergence directly, we search for the best function $T(x)$ that maximizes the objective above.

That is why the framework is called **Variational Divergence Minimization (VDM)**.

---

# 7. Why the Supremum Appears

The supremum appears because the convex conjugate itself is defined through an optimization problem.

So the maximization survives into the final variational form.

That is why the result is written with

$$
\sup_T
$$

instead of a simple closed-form expression.

---

# 8. Why Equality Becomes Inequality

Ideally, we would like an equality:

$$
D_f = \sup(\cdots)
$$

But in practice we only search over a restricted family of functions.

So the result becomes a lower bound:

$$
D_f \ge \sup(\cdots)
$$

## Intuition

Suppose you want the best singer in the world, but you only search among singers in one city. You may not find the global best. You only find the best inside your restricted search space.

That is what is happening here.

---

# 9. How Do We Get Samples From $P_\theta$?

This is the major generative modeling idea.

Start with a known distribution such as:

$$
z \sim \mathcal{N}(0, I)
$$

or

$$
z \sim U(0,1).
$$

We know how to sample from such distributions using random number generators.

Now define a deterministic neural network:

$$
x = G_\theta(z).
$$

Then the output has its own distribution, which depends on the function $G_\theta$.

---

# 10. Deterministic Functions Change Distributions

If

$$
z \sim P_Z
$$

and

$$
x = G(z),
$$

then $x$ becomes a new random variable with a new distribution.

So a deterministic transformation can map a simple distribution into a more complex one.

## Intuition

Think of a machine that takes random clay balls and reshapes them into faces. The output distribution is now about faces, even though the input was just random noise.

That machine is the neural network $G_\theta$.

---

# 11. Neural Networks as Distribution Transformers

This is one of the most important ideas in generative modeling.

The generator network

$$
G_\theta
$$

takes noise samples $z$ and transforms them into samples from a learned model distribution

$$
P_\theta.
$$

So the generator is a sampler.

Changing $\theta$ changes the output distribution.

That means $\theta$ is the knob we tune.

---

# 12. The GAN Setup

We now have two neural networks:

## Generator

$$
G_\theta
$$

This maps noise to generated samples.

## Discriminator / Critic

$$
T_w
$$

This approximates the variational function $T(x)$ used in the F-divergence bound.

---

# 13. Two Optimization Problems

The VDM objective naturally creates two nested optimization problems:

1. **Inner maximization** over $w$ to estimate the divergence.
2. **Outer minimization** over $\theta$ to make $P_\theta$ closer to $P_X$.

So the overall game is:

$$
\min_\theta \max_w J(\theta, w).
$$

---

# 14. Why GANs Are Called Adversarial

The same objective is being optimized in opposite directions:

- the discriminator tries to maximize the objective
- the generator tries to minimize it

That is why the setup is called **adversarial**.

## Intuition

Think of a counterfeiter and a fraud detector.

- The counterfeiter tries to make fake notes look real.
- The fraud detector tries to spot the fake notes.

Each side forces the other to improve.

---

# 15. Saddle Point Optimization

This is not ordinary minimization.

We are looking for a saddle point:

$$
(\theta^*, w^*)
$$

such that the objective decreases in one direction and increases in another.

That is why GAN training is often unstable.

Most optimization problems avoid saddle points. GANs intentionally search for one.

---

# 16. The Classifier Interpretation

For one particular F-divergence choice, the discriminator output lies in $[0,1]$ and can be interpreted as a probability.

Then the discriminator behaves like a binary classifier:

- output 1 for real samples
- output 0 for generated samples

This gives a useful intuition, but it is **not** the general story for all F-divergences.

The deeper general idea is still distribution matching through divergence minimization.

---

# 17. GAN Training in Practice

Training alternates between two steps.

## Step 1: Train the discriminator

- sample real data from $P_X$
- sample noise $z$
- generate fake samples $G_\theta(z)$
- update $w$ to improve the discriminator objective

## Step 2: Train the generator

- freeze $w$
- sample noise again
- generate fake samples
- update $\theta$ to reduce the objective

Then repeat.

---

# 18. Why the Initial Samples Look Bad

At the beginning, the generator is untrained.

So the fake samples look like noise or junk.

As training proceeds, the divergence decreases and the generated samples become closer to the real data distribution.

---

# 19. How the Expectations Are Computed

The objective contains expectations under two distributions:

- $\mathbb{E}_{P_X}[\cdot]$
- $\mathbb{E}_{P_\theta}[\cdot]$

These are approximated using mini-batches.

For real data, we use a batch of samples from the dataset.

For generated data, we sample $z$ from the noise distribution and pass it through $G_\theta$.

This is exactly why the math becomes implementable.

---

# 20. The Most Important Takeaway

Generative modeling can be summarized as:

1. Start from simple noise.
2. Push it through a neural network.
3. Obtain a model distribution.
4. Compare the model distribution to the true data distribution.
5. Update the parameters until the two distributions become close.

That is the core idea behind GANs.

---

# Final Mental Model

A GAN is best understood as a **distribution matching machine**.

- The generator transforms noise into samples.
- The discriminator estimates how different those samples are from real data.
- The learning process alternates between the two until the generated distribution resembles the real distribution.

The full theory comes from **F-divergence**, **variational optimization**, and **saddle point training**.
