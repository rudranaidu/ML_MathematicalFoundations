# Mathematical Foundations of Machine Learning
# 06 - Generative Modeling Foundations

---

# 1. What is Generative Modeling?

Modern AI systems such as:

- ChatGPT
- Gemini
- Claude
- DALL-E
- Stable Diffusion
- Midjourney
- ElevenLabs

are all examples of generative models.

Their goal is not merely prediction.

Their goal is:

> Learn the underlying data distribution and generate new samples from it.

---

# Real World Intuition

Imagine you are shown 1 million cat images.

After years of observing them, you develop an internal understanding of:

- cat faces
- cat ears
- cat fur
- cat body structure

Now if someone asks:

"Draw a cat"

you can create a new cat you've never seen before.

You learned a distribution of cats.

Generative AI attempts to do exactly the same thing.

---

# 2. Conditional Generators

Large Language Models are conditional generators.

Example:

Prompt:

```text
The capital of France is
```

Model predicts:

```text
Paris
```

Mathematically:

$$
P(Y|X)
$$

where

- X = prompt
- Y = generated continuation

---

# Image Generation

Prompt:

```text
A cat sitting on the moon
```

Model generates:

- image pixels

Again:

$$
P(\text{Image}|\text{Prompt})
$$

---

# Speech Generation

Input:

```text
Hello World
```

Output:

- waveform
- audio signal

Again:

$$
P(\text{Speech}|\text{Text})
$$

---

# 3. Dataset

Suppose:

$$
D=\{x_1,x_2,\dots,x_n\}
$$

where:

$$
x_i \overset{iid}{\sim} P_X
$$

meaning:

- Independent
- Identically Distributed

samples from an unknown distribution.

---

# Real World Example

Suppose:

- 10 million images from Instagram

Each image is:

$$
x_i
$$

All images together define:

$$
P_X
$$

which we do not know.

---

# 4. Goal of Generative Modeling

Given:

$$
D \sim P_X
$$

unknown.

Learn:

$$
P_\theta
$$

such that:

$$
P_\theta \approx P_X
$$

and generate new samples.

---

# 5. Discriminative vs Generative Models

## Discriminative Models

Learn:

$$
P(Y|X)
$$

Example:

- Spam Detection
- Image Classification

Goal:

Prediction only.

---

## Generative Models

Learn:

$$
P_X
$$

or

$$
P(X,Y)
$$

Goal:

- understand data
- generate new samples

---

# Real World Analogy

Discriminative model:

Teacher grades answers.

Generative model:

Teacher writes new answers.

---

# 6. The Central Idea Behind Modern Generative Models

Instead of generating images directly:

Start with simple noise.

---

# Step 1

Sample:

$$
Z \sim N(0,I)
$$

Standard Gaussian.

---

# Intuition

Think of Gaussian noise as:

```text
TV static
Random pixels
Pure randomness
```

Easy to sample from.

---

# Step 2

Apply a neural network:

$$
g_\theta(Z)
$$

---

# Step 3

Generate:

$$
\hat X=g_\theta(Z)
$$

---

# Real World Analogy

Imagine clay.

Noise:

```text
random lump of clay
```

Neural network:

```text
sculptor
```

Output:

```text
cat sculpture
```

The sculptor reshapes randomness into meaningful structure.

---

# 7. Why Does a New Distribution Appear?

Initially:

$$
Z \sim N(0,I)
$$

After transformation:

$$
\hat X=g_\theta(Z)
$$

Distribution changes.

---

# Example

Suppose:

$$
g(z)=z^2
$$

Then:

$$
X=z^2
$$

is no longer Gaussian.

The transformation creates a new distribution.

---

# Intuition

The neural network bends and warps probability mass.

This is the heart of generative modeling.

---

# 8. Learning Objective

Generated distribution:

$$
P_\theta
$$

True distribution:

$$
P_X
$$

We want:

$$
P_\theta \approx P_X
$$

---

# Optimization Problem

$$
\hat\theta
=
\arg\min_\theta
D(P_X||P_\theta)
$$

---

# Interpretation

Find neural network parameters such that:

generated data looks indistinguishable from real data.

---

# 9. Four Fundamental Questions

Every generative model answers:

1. How do we compare distributions?
2. How do we choose the neural network?
3. How do we optimize?
4. How do we generate samples?

Different answers produce:

- GANs
- VAEs
- Diffusion Models
- Normalizing Flows

---

# Final Takeaway

Generative Modeling:

1. Start with simple noise

$$
Z\sim N(0,I)
$$

2. Transform it

$$
\hat X=g_\theta(Z)
$$

3. Match distributions

$$
P_\theta \approx P_X
$$

4. Generate new samples

Everything in modern generative AI follows this recipe.
