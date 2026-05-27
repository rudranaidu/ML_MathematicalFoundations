# Generative Modeling — Intuitive Explanation for Beginners

## Core Idea

Given a dataset D, generative modeling tries to:

1. Learn the unknown probability distribution P(X)
2. Generate new samples from it.

This is the mathematical foundation of:
- GANs
- VAEs
- Diffusion Models
- ChatGPT
- Image Generation Models

## Connecting With Probability Basics

Earlier we learned:

$$
X : \Omega \rightarrow \mathbb{R}^d
$$

This means:
- real-world objects become vectors.

Examples:
- image → vector
- sentence → embedding
- speech → signal vector

We also learned:

$$
P_X
$$

describes how data is distributed in vector space.

## The Core Problem

Suppose we collect one million cat images.

These images came from some unknown probability distribution:

$$
P_X
$$

But we do NOT know the exact mathematical formula for this distribution.

We only observe samples:

$$
D = \{x_1, x_2, ..., x_n\}
$$

## Goal of Generative AI

We want to:

### Step 1
Learn the hidden distribution:

$$
P_X
$$

### Step 2
Generate new samples from it.

## Main Recipe of Generative Modeling

Given dataset D,
estimate and sample from unknown P(X)

## Step 1 — Assume a Parametric Model

We introduce a model distribution:

$$
P_\theta
$$

This is our approximation to reality.

## What is Theta?

$$
\theta
$$

represents model parameters.

In neural networks these parameters are:
- weights
- biases
- matrices

Example:

$$
\theta = \{W_1, W_2, W_3\}
$$

## Neural Network as a Function

$$
F_\theta(x)=W_3\sigma(W_2\sigma(W_1x))
$$

where:
- W1, W2, W3 are matrices
- sigma is a nonlinear activation function

Examples of activation functions:
- sigmoid
- ReLU
- tanh

## Why Neural Networks?

Real-world distributions are extremely complicated.

Simple equations cannot model:
- language
- images
- speech
- semantic meaning

Neural networks are flexible enough to approximate highly complex distributions.

## Important Insight

### Real Distribution

$$
P_X
$$

= actual unknown data distribution.

### Model Distribution

$$
P_\theta
$$

= neural network’s approximation to reality.

Goal:

$$
P_\theta \approx P_X
$$

## Step 2 — Measure Distance Between Distributions

$$
D(P_X, P_\theta)
$$

This measures how different the model distribution is from the real distribution.

## Common Distribution Distances

- KL Divergence
- Jensen-Shannon Divergence
- Wasserstein Distance
- Cross Entropy

## Step 3 — Optimize Parameters

Adjust:

$$
\theta
$$

Goal:

$$
D(P_X, P_\theta) \rightarrow \text{minimum}
$$

Eventually:

$$
P_\theta \approx P_X
$$

## Entire Generative Modeling Pipeline

```text
Real World
    ↓
Random Experiments
    ↓
Random Variables
    ↓
Vectors in R^d
    ↓
Unknown Distribution P(X)
    ↓
Dataset D
    ↓
Neural Network Pθ
    ↓
Minimize Distance
    ↓
Pθ ≈ P(X)
    ↓
Generate New Samples
```

## Final Insight

Modern ML does not just learn functions.

Modern Generative AI learns probability distributions over high-dimensional data.
