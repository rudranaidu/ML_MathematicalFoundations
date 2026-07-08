# 26. Applying Bayes' Rule

At the end of the previous section, we arrived at an important expression inside the ELBO that contained the forward transition distribution:

`Q(Xₜ | Xₜ₋₁, X₀)`

Although this expression is mathematically correct, it is not the most useful form for optimization.

Prathosh now applies **Bayes' Rule** to rewrite this distribution in terms of quantities that are either:

- already known,
- easy to compute,
- or directly related to the reverse diffusion process.

This is one of the most important transformations in the DDPM derivation.

---

# 27. Bayes' Rule for Three Variables

Recall the familiar Bayes' Rule:

`P(A|B) = P(B|A) P(A) / P(B)`

In DDPM, we work with three random variables:

- X₀ (original image)
- Xₜ₋₁ (previous noisy image)
- Xₜ (current noisy image)

Applying Bayes' Rule gives:

`Q(Xₜ₋₁ | Xₜ, X₀) = Q(Xₜ | Xₜ₋₁, X₀) × Q(Xₜ₋₁ | X₀) / Q(Xₜ | X₀)`

This equation looks complicated, but it has a very important advantage.

Every term on the right-hand side is either already known or can be computed analytically.

---

# 28. Simplifying the First Term

Notice the first term:

`Q(Xₜ | Xₜ₋₁, X₀)`

Recall the Markov Property.

Once Xₜ₋₁ is known,

Xₜ becomes independent of X₀.

Therefore,

`Q(Xₜ | Xₜ₋₁, X₀) = Q(Xₜ | Xₜ₋₁)`

This simplification is possible **only because the forward process is a First-Order Markov Chain**.

This is why Prathosh repeatedly emphasizes the importance of the Markov assumption.

---

# 29. What Does Bayes' Rule Achieve?

Before applying Bayes' Rule, we had a difficult conditional probability.

After applying Bayes' Rule, we obtain three simpler terms:

`Q(Xₜ | Xₜ₋₁)`

`Q(Xₜ₋₁ | X₀)`

`Q(Xₜ | X₀)`

These three distributions are much easier to work with.

The remaining derivation becomes a matter of simplifying these Gaussian distributions.

---

# 30. Why Are These Distributions Easy?

Recall the forward diffusion equation:

`Xₜ = √αₜ · Xₜ₋₁ + √(1−αₜ) · ε`

where

- ε ~ N(0, I)

Since ε is Gaussian,

and linear transformations of Gaussian variables remain Gaussian,

every one of these distributions is Gaussian.

Therefore,

`Q(Xₜ | Xₜ₋₁)` is Gaussian.

Similarly,

`Q(Xₜ | X₀)` is also Gaussian.

And,

`Q(Xₜ₋₁ | X₀)` is Gaussian as well.

This observation is one of the reasons DDPMs are mathematically elegant.

---

# 31. Computing Q(Xₜ | X₀)

At first glance,

it seems that obtaining Xₜ requires repeatedly applying the forward diffusion process.

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

This would require t separate computations.

Fortunately,

there is a much better approach.

---

# 32. The Recursive Expansion

Recall the forward diffusion equation:

`Xₜ = √αₜ · Xₜ₋₁ + √(1−αₜ) · ε`

Now expand Xₜ₋₁.

Then expand Xₜ₋₂.

Continue recursively.

Eventually,

everything can be expressed directly in terms of X₀.

After simplification,

the result becomes:

`Xₜ = √ᾱₜ · X₀ + √(1−ᾱₜ) · ε`

where

`ᾱₜ = α₁ × α₂ × α₃ × ... × αₜ`

This is one of the most beautiful equations in DDPM.

---

# 33. Why Is This Equation Important?

Instead of computing

```text
X₀ → X₁ → X₂ → ... → Xₜ
```

we can now compute Xₜ directly from X₀.

Only one equation is needed.

This greatly simplifies both:

- mathematical analysis,
- practical implementation.

This is exactly what Chandan implements later in the PyTorch code.

---

# 34. Intuition Behind ᾱₜ

Recall that

α controls how much information survives each diffusion step.

ᾱ represents the cumulative effect of all previous diffusion steps.

Suppose

α₁ = 0.99

α₂ = 0.98

α₃ = 0.97

Then

ᾱ₃ = 0.99 × 0.98 × 0.97

As diffusion progresses,

ᾱ gradually decreases.

Consequently,

the contribution of X₀ becomes smaller,

while the contribution of Gaussian noise becomes larger.

Eventually,

almost all information comes from the noise term.

This explains why

`Xₜ ≈ N(0, I)`

for sufficiently large t.

---

# 35. Computing Q(Xₜ | X₀)

Since

`Xₜ = √ᾱₜ · X₀ + √(1−ᾱₜ) · ε`

and ε is Gaussian,

the distribution immediately becomes

`Q(Xₜ | X₀) = N( √ᾱₜ X₀ , (1−ᾱₜ) I )`

Notice something remarkable.

This distribution is completely known.

Nothing needs to be learned.

Everything depends only on:

- X₀
- ᾱₜ

both of which are already available.

---

# 36. Why Is This So Powerful?

This result gives us a **closed-form expression** for any noisy image.

Instead of repeatedly simulating thousands of diffusion steps,

we can directly sample any intermediate noisy image.

This dramatically speeds up training.

In practice,

every training iteration simply:

1. Picks a random timestep t.
2. Computes ᾱₜ.
3. Samples Gaussian noise ε.
4. Computes Xₜ directly using the closed-form equation.

No recursive simulation is required.

---

# 37. Connection to Chandan's Implementation

Later,

when we implement DDPM in PyTorch,

you will notice variables such as:

- `alpha`
- `alpha_bar`
- `sqrt_alpha_bar`
- `sqrt_one_minus_alpha_bar`

These variables come directly from the equation we just derived.

This is one of the nicest examples where the mathematics maps almost one-to-one into PyTorch code.

---

# Key Insights

- Bayes' Rule converts difficult conditional probabilities into computable Gaussian distributions.
- The Markov Property allows us to simplify `Q(Xₜ | Xₜ₋₁, X₀)` into `Q(Xₜ | Xₜ₋₁)`.
- Recursive expansion leads to a closed-form expression for Xₜ.
- The cumulative product `ᾱₜ` summarizes the effect of all previous diffusion steps.
- Every noisy image can be generated directly from the original image without simulating every intermediate step.
- This closed-form equation is one of the most important equations in DDPM implementation.

---

# What's Next?

Now that every forward distribution can be written analytically,

we can finally simplify the KL divergence inside the ELBO.

This simplification reveals one of the most surprising results in modern generative modeling:

> **Training a DDPM ultimately becomes a simple regression problem.**
