
# P02. DDPM Implementation in PyTorch (Part 1)

> **Core Theme:** Learn how to implement a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch and understand how every component of the implementation maps to the mathematical formulation derived in Theory Chapter 13.

---

# Learning Objectives

After completing this chapter, you will be able to:

✅ Explain the overall DDPM training pipeline.

✅ Understand why DDPM uses a U-Net architecture.

✅ Explain why Time Embeddings are necessary.

✅ Understand how Alpha and Beta schedules are created.

✅ Explain the Forward Diffusion process in code.

✅ Relate every important line of code to the mathematical equations from Chapter 13.

---

# Prerequisites

Before reading this chapter, you should be comfortable with:

- PyTorch tensors
- PyTorch Modules (`nn.Module`)
- CNNs
- U-Net architecture (basic intuition)
- DDPM theory (Chapter 12)
- DDPM ELBO derivation (Chapter 13)

---

# 1. From Theory to Practice

In Theory Chapter 13, we derived the DDPM objective mathematically.

We learned that training a DDPM ultimately becomes a regression problem.

The practical implementation follows exactly the same pipeline.

```text
Training Image

↓

Randomly Choose Time Step (t)

↓

Forward Diffusion

↓

Generate Noisy Image (Xt)

↓

Neural Network

↓

Predict Noise (or Mean)

↓

Compute Loss

↓

Backpropagation
```

Every line of code that Chandan writes implements one of these steps.

Our goal in this chapter is to connect the mathematics with the implementation.

---

# 2. The Overall DDPM Pipeline

A DDPM consists of four major components.

```text
Dataset

↓

Forward Diffusion

↓

Neural Network (U-Net)

↓

Loss Function

↓

Optimizer
```

Notice something important.

Unlike a GAN,

there is

- no discriminator,
- no adversarial training,
- no two-player optimization.

Unlike a VAE,

there is

- no encoder network.

The only trainable component is the **U-Net**.

Everything else is fixed.

---

# 3. Why Does DDPM Use a U-Net?

One of the first things Chandan discusses is the U-Net architecture.

A natural question is:

> Why don't we simply use an ordinary CNN?

The answer comes from the DDPM objective.

Recall that the network receives

```text
Xt
```

and predicts

```text
Noise

(or)

Mean

(or)

Original Image
```

Notice something.

The input image and the output image have exactly the same dimensions.

For example,

```text
Input

64 × 64 × 3

↓

Output

64 × 64 × 3
```

A traditional CNN usually reduces the spatial dimensions.

That is useful for classification.

However,

DDPM is **not** a classification problem.

It is an **image-to-image prediction problem**.

Therefore,

the network must preserve the spatial dimensions.

This is exactly what a U-Net is designed to do.

---

# 4. Understanding the U-Net Architecture

A U-Net consists of three parts.

```text
Encoder

↓

Bottleneck

↓

Decoder
```

---

## Encoder

The encoder gradually extracts higher-level visual features.

At every stage:

- Spatial dimensions decrease.
- Number of feature channels increases.

Example:

```text
64 × 64 × 3

↓

32 × 32 × 64

↓

16 × 16 × 128

↓

8 × 8 × 256
```

The network is learning increasingly abstract representations.

---

## Bottleneck

The bottleneck is the smallest spatial representation.

It contains the most compressed information.

Conceptually,

```text
Image

↓

Compressed Representation

↓

Image
```

Think of it as the point where the network has extracted the most important information from the image.

---

## Decoder

The decoder performs the opposite operation.

Instead of reducing spatial dimensions,

it increases them.

Example:

```text
8 × 8 × 256

↓

16 × 16 × 128

↓

32 × 32 × 64

↓

64 × 64 × 3
```

Eventually,

the output image has the same dimensions as the input.

---

# 5. Why Does the Decoder Increase Resolution?

Suppose we only used an encoder.

The network would produce:

```text
8 × 8
```

feature maps.

But DDPM needs to predict an image having exactly the same resolution as the input.

Therefore,

the decoder gradually reconstructs the original spatial dimensions.

This process is often called

**Upsampling**.

---

# 6. Skip Connections

One of the defining features of U-Net is the use of **Skip Connections**.

Instead of throwing away the features learned by the encoder,

the decoder reuses them.

```text
Encoder

↓

64 Channels

───────────────┐
               │
               ▼

Decoder
```

These skip connections preserve fine image details that would otherwise be lost.

Without them,

generated images become blurry.

---

# 7. Why Skip Connections Matter for DDPM

Remember what the network is trying to do.

It is not trying to classify an image.

It is trying to remove noise while preserving every small visual detail.

Examples include:

- edges,
- textures,
- fine boundaries,
- small objects.

Skip connections allow these details to flow directly from the encoder to the decoder.

This dramatically improves image quality.

---

# 8. Why Does the U-Net Need Time Information?

Suppose we provide the network with a noisy image.

Can the network determine how much noise has been added?

Not always.

Two images may look similar,

yet one may correspond to

```text
t = 20
```

while another corresponds to

```text
t = 700
```

The denoising operation required in these two cases is completely different.

Therefore,

the network must know the current diffusion step.

This is why every DDPM implementation provides the current

```text
Time Step (t)
```

as an additional input.

---

# 9. Time Embeddings

The simplest idea would be to give the network

```text
t = 57
```

as an integer.

Unfortunately,

this is not sufficient.

Neural networks work much better with high-dimensional feature vectors.

Therefore,

the scalar time value is first converted into a vector.

This vector is called a

**Time Embedding**.

Conceptually,

```text
Time Step

↓

Embedding Layer

↓

Time Embedding Vector

↓

Injected into U-Net
```

This allows the network to understand how much diffusion has already occurred.

---

# 10. Why Time Embeddings are Similar to Positional Embeddings

Chandan points out that this idea is identical to the **Positional Encoding** used in Transformers.

In both cases,

we begin with a scalar quantity.

Examples:

Transformer

```text
Token Position

↓

Position Embedding
```

DDPM

```text
Time Step

↓

Time Embedding
```

Both techniques convert a simple integer into a richer representation that the neural network can learn from.

---

# Key Insights

- DDPM is fundamentally an image-to-image prediction problem.
- The input and output images have identical dimensions.
- A U-Net naturally preserves image resolution.
- Skip Connections preserve fine visual details.
- The network must know the current diffusion step.
- Time Embeddings provide this information.
- Time Embeddings play a role similar to Positional Embeddings in Transformers.

---

# What's Next?

Now that we understand **why a U-Net is used**,

we will examine Chandan's PyTorch implementation.

In the next section, we will study:

- U-Net blocks
- Convolution Blocks
- Time Embedding implementation
- Downsampling
- Upsampling
- Skip Connections in code

and explain every important line of the implementation.
