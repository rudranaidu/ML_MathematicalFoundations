# 28. Building the DDPM Class

Now that we have implemented the U-Net,

we can build the actual **DDPM model**.

Think of the DDPM class as the component that manages:

- Forward Diffusion
- Reverse Diffusion
- Alpha Schedule
- Beta Schedule
- Noise Sampling

The U-Net is only the neural network.

The DDPM class defines **how the neural network is used**.

Conceptually,

```text
                 DDPM

        ┌──────────────────┐
        │                  │
        │   Forward Noise  │
        │                  │
        │   Reverse Noise  │
        │                  │
        │  Sampling Logic  │
        │                  │
        └────────┬─────────┘
                 │
                 ▼
               U-Net
```

---

# 29. Initializing the DDPM

The DDPM constructor typically looks like

```python
DDPM(
    model,
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02
)
```

Notice something important.

The U-Net is passed as an argument.

This means the DDPM itself does **not** define the neural network.

Instead,

it simply uses the U-Net to perform denoising.

This separation makes the implementation clean and modular.

---

# 30. Why Do We Need Beta?

Recall the forward diffusion equation.

`Xₜ = √αₜ Xₜ₋₁ + √(1−αₜ) ε`

Instead of directly specifying α,

implementations usually define

β.

Why?

Because

`α = 1 − β`

Beta directly controls

> "How much noise should be added at this diffusion step?"

Small β

↓

Little noise

Large β

↓

More noise

---

# 31. Creating the Beta Schedule

Instead of using one fixed β,

we create one value for every diffusion step.

Example:

```text
β₁

β₂

β₃

⋮

β₁₀₀₀
```

Chandan creates these values using

```python
torch.linspace(...)
```

Conceptually,

```python
betas = torch.linspace(
    beta_start,
    beta_end,
    timesteps
)
```

This simply creates evenly spaced beta values.

Example

```text
0.0001

0.00012

0.00014

...

0.02
```

The exact numbers are less important.

The important idea is that

noise increases gradually.

---

# 32. Computing Alpha

Once β is known,

computing α becomes trivial.

```text
α = 1 − β
```

In code,

```python
alphas = 1 - betas
```

Now every diffusion step has

- β
- α

already computed.

---

# 33. What is Alpha Bar (ᾱ)?

This is one of the most important variables in DDPM.

Recall from Theory Chapter 13.

Instead of repeatedly applying

```text
X₀

↓

X₁

↓

X₂

↓

⋮

↓

Xₜ
```

we derived the closed-form equation

`Xₜ = √ᾱₜ X₀ + √(1−ᾱₜ) ε`

where

ᾱₜ is the cumulative product

`ᾱₜ = α₁ × α₂ × ... × αₜ`

This quantity appears throughout the implementation.

---

# 34. Computing Alpha Bar

PyTorch provides exactly the operation we need.

```python
torch.cumprod(...)
```

Conceptually,

```python
alpha_bar = torch.cumprod(
    alphas,
    dim=0
)
```

Suppose

```text
α

=

0.99

0.98

0.97
```

Then

```text
ᾱ₁

=

0.99

ᾱ₂

=

0.99 × 0.98

ᾱ₃

=

0.99 × 0.98 × 0.97
```

This matches the mathematics exactly.

---

# 35. Why Compute Alpha Bar?

Without ᾱ,

we would have to simulate

```text
X₀

↓

X₁

↓

X₂

↓

⋮

↓

Xₜ
```

every time we wanted one noisy image.

That would be extremely slow.

Instead,

ᾱ allows us to generate

Xₜ

directly from

X₀.

This is why every DDPM implementation computes ᾱ immediately.

---

# 36. Computing the Square Roots

The forward equation contains

`√ᾱₜ`

and

`√(1−ᾱₜ)`

Therefore,

the implementation precomputes

```python
sqrt_alpha_bar

sqrt_one_minus_alpha_bar
```

These values are reused throughout training.

Precomputing them makes the implementation faster.

---

# 37. Forward Diffusion

Now we implement the equation

`Xₜ = √ᾱₜ X₀ + √(1−ᾱₜ) ε`

This is exactly the equation derived in Theory Chapter 13.

Notice something remarkable.

The code is almost identical to the mathematics.

We already have

- X₀
- √ᾱₜ
- √(1−ᾱₜ)

We simply sample

ε

from a standard Gaussian.

Then combine everything.

---

# 38. Sampling Gaussian Noise

Sampling ε is extremely simple.

```python
noise = torch.randn_like(x)
```

This generates

```text
ε ~ N(0, I)
```

with the same shape as the input image.

This line implements exactly the Gaussian noise variable from the mathematical derivation.

---

# 39. Computing Xt

Now everything is available.

The implementation becomes

```python
Xt =
sqrt_alpha_bar * X0
+
sqrt_one_minus_alpha_bar * noise
```

Notice how closely this resembles the theoretical equation.

This is one of the nicest examples where probability theory translates almost directly into code.

---

# 40. Why Randomly Sample Time Steps?

A common beginner question is:

> Why don't we train sequentially?

For example,

```text
Step 1

↓

Step 2

↓

Step 3
```

Instead,

Chandan randomly samples

```text
t

∈

{1,...,1000}
```

every iteration.

Why?

Because every diffusion step is a valid training example.

Random sampling allows the network to learn all diffusion stages simultaneously.

This significantly speeds up training.

---

# 41. Sampling Time Steps in PyTorch

Conceptually,

the implementation is

```python
t = RandomInteger(
    1,
    timesteps
)
```

For an entire batch,

every image receives a different random timestep.

Example

```text
Image 1

↓

t = 43

Image 2

↓

t = 712

Image 3

↓

t = 188
```

This provides a much richer set of training examples.

---

# 42. The Complete Forward Diffusion Pipeline

Putting everything together,

the implementation becomes

```text
Training Image

↓

Randomly Sample t

↓

Compute ᾱₜ

↓

Sample Gaussian Noise

↓

Generate Xt

↓

Pass Xt to U-Net
```

Everything before the U-Net is completely deterministic except for the Gaussian noise.

No learning occurs during this stage.

---

# Key Insights

- The DDPM class manages the diffusion process.
- Beta controls how much noise is added.
- Alpha is computed as `1 − β`.
- Alpha Bar is the cumulative product of all previous alphas.
- Alpha Bar enables direct computation of Xt from X₀.
- Gaussian noise is sampled using `torch.randn_like`.
- Random timestep sampling greatly improves training efficiency.
- The forward diffusion code almost exactly matches the mathematical equation.

---

# What's Next?

Now that we know how to generate noisy images,

we are ready to train the U-Net.

In the next section,

we will study:

- Reverse Diffusion
- Predicting X₀
- Predicting Noise
- DDPM Loss Function
- Training Loop
- Sampling New Images

This is where the complete DDPM implementation comes together.
