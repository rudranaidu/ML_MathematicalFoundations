# 11. Building the U-Net Step by Step

Now that we understand **why** DDPM uses a U-Net,

let us examine how the network is implemented in PyTorch.

Instead of looking at the entire network at once,

we will build it one block at a time.

The complete U-Net is constructed using three building blocks:

- Convolution Blocks
- Downsampling Blocks
- Upsampling Blocks

---

# 12. The Convolution Block

Every U-Net is built from a small reusable component called a **Convolution Block**.

Instead of writing

```python
Conv

↓

BatchNorm

↓

ReLU

↓

Conv

↓

BatchNorm

↓

ReLU
```

again and again,

we combine them into one reusable module.

Conceptually,

```text
Input

↓

Conv2D

↓

Batch Normalization

↓

ReLU

↓

Time Embedding

↓

Conv2D

↓

Batch Normalization

↓

ReLU

↓

Output
```

This becomes one **ConvBlock**.

---

# 13. Why Two Convolutions?

A natural question is:

> Why not use only one convolution?

Using two convolutions allows the network to learn more expressive features.

The first convolution learns simple patterns.

The second convolution combines those patterns into richer representations.

This is why almost every modern CNN architecture uses multiple convolutions inside one block.

---

# 14. Why Batch Normalization?

After every convolution,

Chandan applies

```python
nn.BatchNorm2d(...)
```

Why?

Batch Normalization provides two important advantages.

### 1. Stable Training

Feature values remain within a reasonable range.

This prevents exploding or vanishing activations.

---

### 2. Faster Convergence

The optimizer can use larger learning rates.

Training becomes much more stable.

---

From our Theory Track,

we also know another interpretation.

Batch Normalization acts as a **regularizer**.

---

# 15. Why ReLU?

After every convolution,

the network applies

```python
nn.ReLU()
```

Without ReLU,

multiple convolution layers collapse into one equivalent linear transformation.

The network would lose its ability to learn complex functions.

ReLU introduces non-linearity.

This allows the network to approximate complicated mappings such as

```text
Noisy Image

↓

Clean Image
```

---

# 16. Injecting the Time Embedding

One of the biggest differences between an ordinary CNN and a DDPM U-Net is the **Time Embedding**.

The ConvBlock does something unusual.

After the first convolution,

the Time Embedding is added.

Conceptually,

```text
Image Features

+

Time Embedding

↓

Second Convolution
```

The image representation now contains two types of information:

- Visual Features
- Diffusion Time

Both are needed for denoising.

---

# 17. Why Add the Time Embedding Here?

Imagine two noisy images.

They look almost identical.

However,

one corresponds to

```text
t = 50
```

while another corresponds to

```text
t = 800
```

The denoising operation required is completely different.

Therefore,

every convolution block must know

how much diffusion has already occurred.

Adding the Time Embedding after the first convolution provides this information.

---

# 18. Downsampling Blocks

After each ConvBlock,

the spatial dimensions are reduced.

Example:

```text
64 × 64

↓

32 × 32

↓

16 × 16

↓

8 × 8
```

At the same time,

the number of channels increases.

Example:

```text
64 Channels

↓

128 Channels

↓

256 Channels

↓

512 Channels
```

This allows the network to learn increasingly abstract visual features.

---

# 19. Why Reduce Spatial Dimensions?

Reducing the spatial dimensions has two advantages.

### Computational Efficiency

Smaller feature maps require fewer computations.

Training becomes much faster.

---

### Larger Receptive Field

Every neuron gradually sees a larger portion of the image.

This helps the network understand global image structure.

---

# 20. The Bottleneck

Eventually,

the encoder reaches the smallest spatial representation.

This layer is called the **Bottleneck**.

Conceptually,

```text
Input Image

↓

Encoder

↓

Bottleneck

↓

Decoder

↓

Output Image
```

The bottleneck contains the most compressed representation of the noisy image.

All important information must pass through this layer.

---

# 21. Upsampling Blocks

After the bottleneck,

the decoder begins reconstructing the image.

Instead of reducing spatial dimensions,

it now increases them.

Example:

```text
8 × 8

↓

16 × 16

↓

32 × 32

↓

64 × 64
```

This process is called **Upsampling**.

---

# 22. Transposed Convolution

To increase image resolution,

the decoder uses

```python
nn.ConvTranspose2d(...)
```

This operation is often called:

- Transposed Convolution
- Deconvolution
- Up Convolution

Its purpose is simple.

Increase the spatial dimensions while learning useful image features.

---

# 23. Skip Connections in Code

Remember the skip connections from Part 1.

During implementation,

the feature maps produced by the encoder are saved.

Later,

the decoder concatenates them with its own feature maps.

Conceptually,

```text
Encoder Feature Map

───────────────┐
               │
               ▼

Decoder Feature Map

↓

Concatenate

↓

Next Convolution
```

This allows the decoder to recover fine image details.

---

# 24. Why Concatenate Instead of Replace?

Suppose the decoder only used its own features.

Many small details would already have been lost.

By concatenating the encoder features,

the decoder receives both:

- High-level semantic information
- Fine spatial details

This significantly improves reconstruction quality.

---

# 25. The Overall U-Net

Putting everything together,

the U-Net becomes

```text
Input Image

↓

ConvBlock

↓

Downsample

↓

ConvBlock

↓

Downsample

↓

ConvBlock

↓

Bottleneck

↓

Upsample

↓

Concatenate Skip Connection

↓

ConvBlock

↓

Upsample

↓

Concatenate Skip Connection

↓

ConvBlock

↓

Output
```

Notice that the architecture is perfectly symmetric.

The decoder mirrors the encoder.

---

# 26. Why Only One U-Net?

One question raised during the lecture was:

> Should we train a different U-Net for every diffusion step?

The answer is:

**No.**

The same U-Net is reused for all diffusion steps.

The only thing that changes is the **Time Embedding**.

Conceptually,

```text
XT

+

Time Embedding

↓

Same U-Net

↓

Prediction
```

This dramatically reduces the number of learnable parameters.

---

# 27. Why Randomly Sample Time Steps?

Another important implementation detail.

During training,

we do **not** process the diffusion steps sequentially.

Instead,

every iteration randomly samples

```text
t

∈

{1, 2, 3, ..., T}
```

Why?

Because every diffusion step represents a valid supervised learning example.

Random sampling allows the network to learn all diffusion stages simultaneously.

This also makes training much faster.

---

# Key Insights

- The U-Net is constructed from reusable ConvBlocks.
- Every ConvBlock contains two convolutions.
- Batch Normalization stabilizes training.
- ReLU introduces non-linearity.
- Time Embeddings are injected after the first convolution.
- Downsampling reduces spatial dimensions while increasing feature channels.
- Upsampling reconstructs the original image resolution.
- Skip Connections preserve fine image details.
- A single U-Net is shared across all diffusion steps.
- Random time-step sampling makes DDPM training efficient.

---

# What's Next?

Now that we understand the U-Net,

we are ready to study the actual DDPM implementation.

In the next section,

we will implement:

- Alpha Schedule
- Beta Schedule
- Forward Diffusion
- Sampling Noisy Images
- Reverse Diffusion
- DDPM Training Loop

and connect every line of PyTorch code directly to the mathematical equations derived in Theory Chapter 13.
