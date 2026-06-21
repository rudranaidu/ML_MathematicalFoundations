> **GANs are not an isolated algorithm. They emerge naturally from the framework of Variational Divergence Minimization (VDM).**

---

# 1. Where We Are in the Story

In the previous note (**07_F_Divergence_and_Variational_Optimization.md**) we derived a variational representation of an F-divergence.

Our goal remains:

Given a dataset

$$
D = {x_1, x_2, \ldots, x_N}
$$

where

$$
x_i \sim P_X
$$

we want to learn a model distribution

$$
P_\theta
$$

such that

$$
P_\theta \approx P_X
$$

and then generate new samples from it.

---

# 2. The Central Challenge

We need a way to measure how different two distributions are:

$$
P_X
$$

and

$$
P_\theta
$$

This is done using a divergence measure.

Examples include:

* KL Divergence
* Reverse KL Divergence
* Jensen-Shannon Divergence
* Pearson Divergence
* Chi-Square Divergence

All belong to a larger family called:

$$
D_f(P_X,P_\theta)
$$

known as **F-Divergences**.

---

# 3. Why Direct Computation Is Impossible

The original divergence formula contains:

$$
P_X
$$

and

$$
P_\theta
$$

explicitly.

The problem:

We do not know either density function.

We only have samples.

Therefore:

We cannot directly compute

$$
D_f(P_X,P_\theta)
$$

---

# 4. The Key Machine Learning Trick

Instead of computing expectations from a distribution,

we compute them from samples.

By the Law of Large Numbers:

$$
\mathbb{E}[h(X)]
\approx
\frac{1}{N}
\sum_{i=1}^{N}
h(x_i)
$$

for sufficiently large (N).

---

## Intuition

Suppose you want average human height.

You do not know the true distribution of human heights.

You simply measure 10,000 people and average them.

Machine learning uses exactly the same idea.

---

# 5. Variational Form of F-Divergence**

Using convex conjugates, the F-divergence can be rewritten as:

```math
D_f(P_X,P_\theta)
=
\sup_T
\left(
\mathbb{E}_{P_X}[T(x)]
-
\mathbb{E}_{P_\theta}[F^*(T(x))]
\right)
```

where:

- \(F^*\) is the convex conjugate of \(F\)
- \(T(x)\) is an auxiliary function being optimized
- \(\sup\) denotes the supremum (maximum)

This equation is the foundation of **Variational Divergence Minimization (VDM)**.

# 6. Why "Variational"?

Variational simply means:

> Representing a quantity through an optimization problem.

Instead of computing divergence directly, we search for the best function (T(x)).

Thus:

Computing divergence itself becomes an optimization problem.

---

# 7. Why Does Supremum Appear?

The convex conjugate is defined through a maximization problem.

Because the conjugate appears inside the derivation,

the maximization survives all the way to the final formula:

$$
\sup_T
$$

This is where the supremum comes from.

---

# 8. Why Equality Becomes Inequality

Ideally:

```math
D_f
=
\sup(\cdots)
```

In practice:

```math
D_f
\ge
\sup(\cdots)
```

because we only search over a restricted family of functions.

### Intuition

Suppose you want the world's best singer.

But you are only allowed to search inside Bangalore.

The best singer in Bangalore may not be the best singer in the world.

Therefore your answer becomes a lower bound.

The same thing happens here.

The supremum is taken over a restricted class of functions rather than all possible functions. Therefore we obtain a lower bound:

```math
D_f
\ge
\sup(\cdots)
```

# 9. The Missing Piece

So far we have:

* Samples from (P_X)
* A way to estimate divergence

But we still need samples from:

$$
P_\theta
$$

How do we obtain them?

---

# 10. Sampling Through Neural Networks

Start with a known distribution:

$$
z \sim \mathcal{N}(0,1)
$$

or

$$
z \sim U(0,1)
$$

We know how to sample from these distributions using random number generators.

Now define:

$$
\hat{x} = G_\theta(z)
$$

where:

$$
G_\theta
$$

is a neural network.

---

# 11. A Fundamental Probability Result

If

$$
z \sim P_Z
$$

and

$$
\hat{x} = G(z)
$$

then

$$
\hat{x}
$$

has a new distribution.

A deterministic transformation changes distributions.

---

## Intuition

Imagine:

Input:

* random clay balls

Transformation:

* sculpting machine

Output:

* human faces

The output distribution differs from the input distribution.

The sculpting machine is:

$$
G_\theta
$$

---

# 12. Neural Networks as Distribution Transformers

This is one of the deepest ideas in Generative AI.

We begin with:

$$
z \sim \mathcal{N}(0,1)
$$

and pass it through a neural network:

$$
G_\theta
$$

which transforms it into:

$$
\hat{x} \sim P_\theta
$$

Thus:

The neural network becomes a sampler.

---

# 13. Theta Is the Only Knob

Changing:

$$
\theta
$$

changes:

$$
P_\theta
$$

Therefore:

**Theta is the only knob we can turn.**

Our objective:

$$
P_\theta
\rightarrow
P_X
$$

---

# 14. Two Neural Networks Appear

To solve the variational optimization problem we introduce:

## Generator

$$
G_\theta
$$

Generates samples from:

$$
P_\theta
$$

---

## Critic / Discriminator

$$
T_w
$$

Approximates the function:

$$
T(x)
$$

appearing in the variational representation.

---

# 15. The GAN Objective

The variational divergence becomes:

$$
J(\theta,w)
===========

## \mathbb{E}_{P_X}[T_w(x)]

\mathbb{E}*{P*\theta}[F^*(T_w(x))]
$$

The discriminator tries to maximize this objective.

The generator tries to minimize it.

---

# 16. Two Optimization Problems

We first solve:

$$
\max_w J(\theta,w)
$$

This computes the divergence estimate.

Then we solve:

$$
\min_\theta J(\theta,w)
$$

This reduces the divergence.

Combining both gives:

$$
\min_\theta
\max_w
J(\theta,w)
$$

---

# 17. Why GANs Are Called Adversarial

The same objective is:

* maximized by one network
* minimized by another network

Generator and discriminator have opposing goals.

Therefore:

The optimization becomes adversarial.

---

## Counterfeit Currency Analogy

Generator:

Creates fake currency.

Discriminator:

Attempts to detect fake currency.

Both improve continuously.

---

# 18. Saddle Point Optimization

This is not ordinary minimization.

We are looking for a saddle point:

$$
(\theta^*, w^*)
$$

such that:

$$
\min_\theta
\max_w
J(\theta,w)
$$

is satisfied.

---

## Horse Saddle Intuition

A saddle point behaves differently depending on direction.

Move along one direction:

* function increases

Move along another direction:

* function decreases

This resembles a horse saddle.

---

# 19. Why GAN Training Is Difficult

Most optimization algorithms try to avoid saddle points.

GANs intentionally seek saddle points.

Therefore training often becomes:

* unstable
* oscillatory
* difficult to converge

This is one of the major limitations of GANs.

---

# 20. Classifier Interpretation of GANs

For the Jensen-Shannon divergence,

the discriminator output lies between:

$$
0
$$

and

$$
1
$$

Therefore it can be interpreted as a probability.

This allows us to view the discriminator as a binary classifier.

---

Output:

$$
1
$$

means:

Real sample.

Output:

$$
0
$$

means:

Generated sample.

---

# 21. Classifier-Guided Generation

Imagine:

A classifier tries to distinguish:

* Real samples
* Generated samples

The generator continuously improves until the classifier fails.

When the classifier can no longer distinguish them:

$$
P_\theta
\approx
P_X
$$

---

# 22. Important Warning

Prof. Prathosh repeatedly emphasizes:

This classifier interpretation is not general.

It works only for specific choices of F-divergence.

For many divergences:

* there is no classifier interpretation
* there is no "fooling the discriminator"

The general framework remains:

**Distribution Matching via Divergence Minimization.**

---

# 23. GAN Training Procedure

## Step 1

Sample real data:

$$
x_i \sim P_X
$$

---

## Step 2

Sample noise:

$$
z_i \sim \mathcal{N}(0,1)
$$

Generate:

$$
\hat{x}*i = G*\theta(z_i)
$$

---

## Step 3

Update discriminator:

$$
w
\leftarrow
w + \eta \nabla_w J
$$

---

## Step 4

Freeze discriminator.

Update generator:

$$
\theta
\leftarrow
\theta - \eta \nabla_\theta J
$$

---

## Step 5

Repeat until convergence.

---

# 24. What Happens During Training?

Initially:

Generated samples are random noise.

The generated distribution:

$$
P_\theta
$$

is far away from:

$$
P_X
$$

---

As training proceeds:

$$
D_f(P_X,P_\theta)
\downarrow
$$

The generated samples become increasingly realistic.

Eventually:

$$
P_\theta
\approx
P_X
$$

---

# 25. Conditional GANs (CGANs)

Suppose we want controlled generation.

Example:

Generate digit "3".

Generate digit "7".

Generate an image from a text description.

We need labeled data:

$$
(x,y)
\sim
P(X,Y)
$$

where:

* (x) = image
* (y) = label or text condition

The generator now learns:

$$
P(X|Y)
$$

instead of:

$$
P(X)
$$

This enables conditional generation.

---

# 26. The Most Important Mental Model

Generative Modeling can be viewed as:

1. Start with noise

$$
z \sim \mathcal{N}(0,1)
$$

2. Pass through a neural network

$$
G_\theta
$$

3. Obtain samples

$$
\hat{x} \sim P_\theta
$$

4. Measure divergence

$$
D(P_X,P_\theta)
$$

5. Update parameters

$$
\theta
$$

6. Repeat until

$$
P_\theta
\approx
P_X
$$

---

# Final Takeaway

GANs are not magic.

GANs are simply:

* Variational Divergence Minimization
* Implemented using Neural Networks
* Solved via Saddle Point Optimization

The Generator:

$$
G_\theta
$$

creates samples.

The Discriminator:

$$
T_w
$$

estimates divergence.

Together they gradually force:

$$
P_\theta
\rightarrow
P_X
$$

which allows us to generate realistic data.

This is the core idea behind Generative Adversarial Networks.
