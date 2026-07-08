
# 13. DDPM ELBO Derivation and Training (Part 1)

> **Core Theme:** Learn how the Evidence Lower Bound (ELBO) is derived for Diffusion Models and understand why the DDPM training objective ultimately becomes a simple regression problem.

---

# 1. Where We Are in the Story

In the previous chapter, we introduced **Denoising Diffusion Probabilistic Models (DDPMs)** as a special class of **Hierarchical Latent Variable Models**.

We learned that a DDPM consists of two parts.

## Forward Process (Encoder)

The encoder gradually destroys information.

```text
Original Image (X₀)

↓

X₁

↓

X₂

↓

⋮

↓

XT

↓

Pure Gaussian Noise
```

Unlike a Variational Autoencoder,

this encoder is completely **fixed**.

No neural network is involved.

---

## Reverse Process (Decoder)

The decoder performs the opposite task.

```text
Pure Gaussian Noise

↓

XT

↓

XT−1

↓

XT−2

↓

⋮

↓

X₀
```

Unlike the encoder,

this reverse process **must be learned**.

---

# 2. The Remaining Question

At the end of Chapter 12, we understood:

- how the encoder works,
- how the decoder is defined,
- what probability distributions are involved,
- and what parameters must be learned.

One important question still remained.

> **How do we train the reverse process?**

This chapter answers that question.

---

# 3. A Quick Recap of Variational Autoencoders

Before deriving the DDPM objective,

Prathosh reminds us that DDPMs are still **latent-variable models**.

Therefore,

the same mathematical framework developed for Variational Autoencoders still applies.

Recall the VAE objective.

We wanted to maximize

```text
log Pθ(X)
```

However,

this quantity was difficult to optimize directly.

Therefore,

we introduced a lower bound:

```text
ELBO
```

and optimized that instead.

Exactly the same strategy is used for DDPMs.

---

# 4. Why Do We Need ELBO Again?

Our ultimate objective has not changed.

We still want the model distribution to match the true data distribution.

Conceptually,

```text
Pθ(X₀)

≈

PX₀
```

where

- `PX₀` is the true data distribution.
- `Pθ(X₀)` is the model distribution.

Therefore,

the optimization problem is still:

```text
Maximize

log Pθ(X₀)
```

Since DDPMs are latent-variable models,

this quantity is again difficult to compute exactly.

So,

just like a VAE,

we maximize a lower bound instead.

---

# 5. The Difference Between VAE and DDPM

Although both optimize an ELBO,

there is one major difference.

## Variational Autoencoder

Only one latent variable exists.

```text
X

↓

Z

↓

X̂
```

---

## Diffusion Model

There are many latent variables.

```text
X₀

↓

X₁

↓

X₂

↓

⋮

↓

XT
```

Instead of one latent variable,

the ELBO must now account for an entire sequence of latent variables.

This is the main reason why the derivation becomes longer.

---

# 6. The Forward Distribution

Recall that the forward diffusion process was fixed.

Every transition followed a Gaussian distribution.

Conceptually,

```text
Q(Xt | Xt−1)
```

which reads as:

> "The probability distribution of the current noisy image given the previous noisy image."

Each transition is Gaussian.

Each transition is known.

Nothing about the forward process needs to be learned.

---

# 7. The Reverse Distribution

The reverse process is different.

Instead of

```text
Q(Xt | Xt−1)
```

we model

```text
Pθ(Xt−1 | Xt)
```

which reads as:

> "The probability distribution of the previous (less noisy) image given the current noisy image."

Unlike the forward process,

the parameters of this distribution are unknown.

They must be learned.

---

# 8. The Two Distributions We Need

To derive the ELBO,

we need two probability distributions.

## Forward Distribution

```text
Q(X₁, X₂, ..., XT | X₀)
```

This is completely known.

---

## Reverse Distribution

```text
Pθ(X₀, X₁, ..., XT)
```

This is our model.

Its parameters must be learned.

Notice something important.

The ELBO is always constructed using:

- the model distribution,
- the variational (forward) distribution.

Exactly the same idea appeared in Variational Autoencoders.

---

# 9. Defining the DDPM Model

A latent-variable model is defined by its **joint probability distribution**.

For a DDPM,

the joint distribution contains:

- the observed data,
- every latent variable.

Conceptually,

```text
Pθ(X₀, X₁, ..., XT)
```

Using the Chain Rule of Probability,

this becomes

```text
P(XT)

×

Pθ(XT−1 | XT)

×

Pθ(XT−2 | XT−1)

×

⋯

×

Pθ(X₀ | X₁)
```

Notice the direction.

The chain starts from

```text
XT
```

because inference begins with pure Gaussian noise.

Then,

one reverse transition is performed at a time until the model reaches

```text
X₀
```

This chain completely defines the DDPM.

---

# 10. Why Does the Chain Start at XT?

At first,

this looks backwards.

Why not start with the original image?

The reason is simple.

During inference,

we do **not** have an image.

We only know that

```text
XT

≈

N(0, I)
```

Therefore,

generation always starts from random Gaussian noise.

The model then repeatedly samples:

```text
XT

↓

XT−1

↓

XT−2

↓

⋮

↓

X₀
```

Eventually,

a realistic image appears.

---

# 11. The Forward Distribution

Now consider the encoder.

Since the forward process is a Markov Chain,

the entire encoder distribution can also be written using the Chain Rule.

Conceptually,

```text
Q(X₁, X₂, ..., XT | X₀)
```

becomes

```text
Q(X₁ | X₀)

×

Q(X₂ | X₁)

×

Q(X₃ | X₂)

×

⋯

×

Q(XT | XT−1)
```

Notice the difference.

The encoder moves

```text
Forward
```

while the decoder moves

```text
Backward.
```

---

# 12. Why Is This Simplification Possible?

This simplification relies on one important property.

The forward diffusion process is a **First-Order Markov Chain**.

That means

```text
Xt

depends only on

Xt−1
```

It does **not** depend directly on:

- X₀
- X₁
- X₂
- ...

Therefore,

the complicated joint probability naturally decomposes into a product of simple conditional distributions.

Without the Markov property,

the ELBO derivation would become much more difficult.

---

# 13. The Two Ingredients of the ELBO

We now have everything required to build the ELBO.

We know:

## Model Distribution

```text
Pθ(X₀, X₁, ..., XT)
```

---

## Forward Distribution

```text
Q(X₁, ..., XT | X₀)
```

The derivation now follows exactly the same roadmap we used for Variational Autoencoders.

The only difference is that instead of one latent variable,

we now have an entire hierarchy of latent variables.

---

# Key Insights

- DDPMs are still latent-variable models.
- Therefore, DDPMs are trained by maximizing an ELBO.
- The forward process defines the variational distribution.
- The reverse process defines the model distribution.
- Both distributions can be factorized using the Chain Rule.
- The Markov property makes this factorization simple.
- The forward chain and reverse chain together provide the two ingredients needed to derive the DDPM ELBO.

---

# What's Next?

Now that we have defined:

- the forward distribution,
- the reverse distribution,
- and the DDPM joint probability,

we are finally ready to construct the **DDPM ELBO**.

In the next section, we will derive the ELBO step by step using:

- Bayes' Rule,
- the Markov Property,
- conditional expectations,
- and KL divergence,

eventually transforming a seemingly complicated probabilistic objective into an elegant optimization problem.
