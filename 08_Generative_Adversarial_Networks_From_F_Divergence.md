# 1. Where We Are in the Story

Recall the goal of Generative Modeling:

Given samples

\[
D = \{x_1,x_2,\ldots,x_N\}
\]

generated from some unknown distribution

\[
P_X
\]

we want to learn a model

\[
P_\theta
\]

such that

\[
P_\theta \approx P_X
\]

and then generate new samples from it.

---

# 2. The Central Problem

We need a way to measure:

\[
P_X
\]

vs

\[
P_\theta
\]

How close are they?

This is done using a **divergence measure**.

Examples:

- KL Divergence
- Reverse KL
- Jensen-Shannon Divergence
- Pearson Divergence
- Chi-Square Divergence

All belong to the family:

\[
D_f(P_X,P_\theta)
\]

called **F-Divergences**.

---

# 3. Why F-Divergence Is Difficult

The original divergence contains:

\[
P_X
\]

and

\[
P_\theta
\]

explicitly.

Problem:

We do not know either density.

We only have samples.

Therefore:

We cannot directly compute

\[
D_f(P_X,P_\theta)
\]

---

# 4. The Key Machine Learning Trick

Machine learning repeatedly uses one powerful idea:

## Law of Large Numbers

Instead of computing expectations from distributions,

compute them using sample averages.

If

\[
X \sim P_X
\]

then

\[
E[h(X)]
\]

can be approximated as

\[
\frac{1}{N}
\sum_{i=1}^{N}
h(x_i)
\]

for sufficiently large N.

---

## Real World Intuition

Suppose you want the average human height.

You do not know the true distribution of human heights.

Measure 10,000 people.

Average them.

Done.

Machine learning works exactly the same way.

---

# 5. Why Convex Conjugates Are Introduced

The original F-divergence is impossible to compute because it requires density functions.

The goal is:

Convert

\[
D_f(P_X,P_\theta)
\]

into a form involving expectations.

Expectations can then be estimated using samples.

To achieve this, convexity theory introduces:

- Convex functions
- Convex conjugates

which allow the divergence to be rewritten as:

\[
D_f(P_X,P_\theta)
\ge
\sup_T
\left(
E_{P_X}[T(x)]
-
E_{P_\theta}[F^*(T(x))]
\right)
\]

where

\[
F^*
\]

is the convex conjugate.

---

# 6. What Does "Variational" Mean?

The word sounds scary.

It actually means:

> Represent a quantity as an optimization problem.

Instead of directly computing divergence,

we search for the best function

\[
T(x)
\]

that maximizes the above expression.

This is why the method is called:

**Variational Divergence Minimization (VDM)**.

---

# 7. Why Does Supremum Appear?

Students asked this question during class.

The answer:

The convex conjugate itself contains a maximization problem.

That maximization survives throughout the derivation and appears as

\[
\sup_T
\]

in the final equation.

---

# 8. Why Equality Becomes Inequality

Ideally:

\[
D_f
=
\sup(...)
\]

However in practice:

\[
D_f
\ge
\sup(...)
\]

because we search only inside a restricted family of functions.

---

## Real World Analogy

Suppose you want the world's best singer.

But you are allowed to search only inside Bangalore.

The best singer in Bangalore may not be the best singer in the world.

Therefore your result becomes a lower bound.

The same thing happens here.

---

# 9. The Biggest Question

How do we get samples from

\[
P_\theta
\]

?

Until now we only talked about distributions.

We still need a mechanism that generates samples.

---

# 10. Sampling Through Neural Networks

Start with a known distribution:

\[
z \sim N(0,1)
\]

or

\[
z \sim U(0,1)
\]

We know how to sample these using random number generators.

Now define:

\[
x = G_\theta(z)
\]

where

\[
G_\theta
\]

is a neural network.

---

# 11. A Deep Probability Result

If

\[
z \sim P_Z
\]

and

\[
x = G(z)
\]

then

\[
x
\]

has a new distribution.

A deterministic transformation changes distributions.

---

## Real World Analogy

Imagine:

Input:

- random clay balls

Machine:

- reshapes them

Output:

- human faces

The output distribution differs from the input distribution.

The reshaping machine is:

\[
G_\theta
\]

---

# 12. Neural Networks as Distribution Transformers

This is one of the most important ideas in Generative AI.

Neural network:

\[
G_\theta
\]

takes

\[
z \sim N(0,1)
\]

and transforms it into

\[
x \sim P_\theta
\]

Thus:

\[
G_\theta
\]

acts as a sampler.

---

# 13. Why Theta Matters

Changing

\[
\theta
\]

changes

\[
P_\theta
\]

Therefore:

Theta is the only knob we have.

Adjusting theta changes the generated distribution.

The goal is:

\[
P_\theta \rightarrow P_X
\]

---

# 14. The GAN Idea

Now we have:

## Generator

\[
G_\theta
\]

Generates samples from

\[
P_\theta
\]

---

## Critic / Discriminator

\[
T_w
\]

Measures how different

\[
P_\theta
\]

is from

\[
P_X
\]

---

# 15. Two Optimization Problems

The F-divergence itself requires solving:

\[
\max_w
\]

to compute the divergence.

After computing divergence:

\[
\min_\theta
\]

to reduce it.

Thus:

\[
\min_\theta
\max_w
J(\theta,w)
\]

---

# 16. Why GANs Are Called Adversarial

The same objective is:

Maximized by one network.

Minimized by another network.

---

Generator:

tries to minimize divergence.

Discriminator:

tries to maximize divergence estimate.

---

## Real World Analogy

Counterfeit money factory:

Generator:
creates fake notes.

Discriminator:
tries to detect fake notes.

Both continuously improve.

---

# 17. Saddle Point Optimization

GAN training is not ordinary optimization.

It seeks:

\[
(\theta^*,w^*)
\]

such that

\[
\min_\theta
\max_w
J(\theta,w)
\]

This point is called a saddle point.

---

## Horse Saddle Intuition

Move in one direction:

Function increases.

Move in another direction:

Function decreases.

This shape resembles a horse saddle.

---

# 18. Why GAN Training Is Difficult

Most optimization algorithms try to avoid saddle points.

GANs intentionally seek them.

Therefore GAN training is often:

- unstable
- oscillatory
- difficult to converge

---

# 19. Discriminator as a Classifier

For Jensen-Shannon Divergence:

The discriminator output lies between

0 and 1.

Therefore it can be interpreted as a probability.

This allows us to view it as a binary classifier.

---

Output:

1 → Real sample

0 → Fake sample

---

# 20. Classifier-Guided Generation

Imagine:

Classifier distinguishes:

- Real images
- Generated images

Generator keeps changing until classifier fails.

When classifier can no longer distinguish:

\[
P_\theta
\approx
P_X
\]

---

# 21. Why This Interpretation Is Limited

Prof. Prathosh repeatedly warns:

This classifier interpretation works only for specific divergences.

For other F-divergences:

- No classifier interpretation exists.
- No "fooling the discriminator" story exists.

The general viewpoint remains:

**Distribution Matching via Divergence Minimization.**

---

# 22. GAN Training Algorithm

Step 1:

Sample real data

\[
x_i \sim P_X
\]

---

Step 2:

Sample noise

\[
z_i \sim N(0,1)
\]

Generate

\[
x_i^{fake}
=
G_\theta(z_i)
\]

---

Step 3:

Train discriminator.

---

Step 4:

Freeze discriminator.

Train generator.

---

Step 5:

Repeat.

---

# 23. What Happens During Training?

Initially:

Generated samples are junk.

Random blobs.

Noise.

---

As divergence decreases:

Generated samples become increasingly realistic.

Eventually:

\[
P_\theta
\approx
P_X
\]

and generated samples resemble real data.

---

# 24. The Most Important Takeaway

Generative Modeling can be viewed as:

1. Start with noise

\[
z \sim N(0,1)
\]

2. Pass through a neural network

\[
G_\theta
\]

3. Obtain samples

\[
x \sim P_\theta
\]

4. Measure divergence

\[
D(P_X,P_\theta)
\]

5. Update parameters

\[
\theta
\]

6. Repeat until

\[
P_\theta \approx P_X
\]

---

# Final Mental Model

Generative AI is fundamentally:

> Learning a transformation that converts simple noise into samples from a complex real-world distribution.

GANs achieve this through:

- F-Divergence Minimization
- Variational Optimization
- Adversarial Training
- Saddle Point Optimization

Everything else is implementation detail.
