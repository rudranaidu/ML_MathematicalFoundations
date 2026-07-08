# 14. Constructing the DDPM ELBO

Now that we have defined both probability distributions,

- the forward distribution
- the reverse distribution

we are finally ready to construct the ELBO for DDPM.

Recall that our objective is still

```text
Maximize

log Pθ(X₀)
```

Since this quantity is difficult to optimize directly,

we maximize its lower bound instead.

Exactly the same strategy was used in Variational Autoencoders.

---

# 15. The DDPM ELBO

For a VAE, the ELBO was

```text
ELBO

=

EQ(Z|X)

[

log

Pθ(X,Z)

─────────

Q(Z|X)

]
```

For a DDPM,

we simply replace the single latent variable `Z`

with the entire sequence of latent variables.

Conceptually,

```text
ELBO

=

EQ(X₁,...,XT | X₀)

[

log

Pθ(X₀,...,XT)

──────────────────────

Q(X₁,...,XT | X₀)

]
```

Notice the only difference.

Instead of

```text
One Latent Variable
```

we now have

```text
Many Latent Variables.
```

Everything else remains exactly the same.

---

# 16. Why Does the ELBO Look So Complicated?

At first glance,

the numerator and denominator appear enormous.

However,

they are simply the two probability distributions that we already defined.

The numerator is

```text
Pθ(X₀,...,XT)
```

which represents our generative model.

The denominator is

```text
Q(X₁,...,XT | X₀)
```

which represents the fixed forward diffusion process.

The derivation now becomes a process of replacing these two expressions with
their factorized forms.

---

# 17. Expanding the Numerator

Earlier,

we used the Chain Rule of Probability to factorize the model.

Conceptually,

```text
Pθ(X₀,...,XT)

=

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

Nothing new has happened.

We are simply substituting this expression into the ELBO.

---

# 18. Expanding the Denominator

Similarly,

the forward distribution becomes

```text
Q(X₁,...,XT | X₀)

=

Q(X₁ | X₀)

×

Q(X₂ | X₁)

×

⋯

×

Q(XT | XT−1)
```

Again,

this is simply the Chain Rule combined with the Markov Property.

Now,

both the numerator and denominator are written entirely as products of
conditional distributions.

---

# 19. The ELBO Before Simplification

After substitution,

the ELBO conceptually becomes

```text
ELBO

=

Expectation

of

log

{

P(XT)

×

Pθ(XT−1 | XT)

×

⋯

×

Pθ(X₀ | X₁)

──────────────────────────────

Q(X₁ | X₀)

×

Q(X₂ | X₁)

×

⋯

×

Q(XT | XT−1)

}
```

Although this expression looks intimidating,

it is actually much easier than it appears.

The rest of the derivation simply simplifies this fraction.

---

# 20. Why Does Prathosh Separate the First Term?

One of the first algebraic manipulations performed in the lecture is to isolate
the first reconstruction term.

Instead of writing everything inside one giant product,

he separates

```text
Pθ(X₀ | X₁)
```

from the remaining terms.

Why?

Because this term directly involves the original data.

Later,

it naturally becomes the **reconstruction term** of the DDPM objective.

Nothing mathematical changes.

It is simply reorganized to make the later interpretation clearer.

---

# 21. Applying the Logarithm Rule

The next simplification is based on a simple logarithm identity.

Recall

```text
log(a × b)

=

log(a)

+

log(b)
```

Similarly,

```text
log

a
──
b

=

log(a)

−

log(b)
```

Since both the numerator and denominator are products,

the logarithm converts them into sums.

This is an extremely useful trick because

products are difficult to optimize,

whereas sums are much easier.

---

# 22. Why Logarithms Help

Suppose we have

```text
A × B × C × D
```

Optimizing this directly is difficult.

After taking the logarithm,

it becomes

```text
log(A)

+

log(B)

+

log(C)

+

log(D)
```

Each term can now be analyzed independently.

This is exactly why almost every probabilistic model uses log-likelihood rather
than likelihood.

---

# 23. A New Challenge Appears

After expanding the logarithm,

one particular term becomes problematic.

Conceptually,

we encounter

```text
Q(XT | XT−1)
```

inside the denominator.

Unfortunately,

this form is not immediately useful for the derivation.

We would prefer a distribution that contains

```text
X₀
```

because our objective ultimately depends on the observed data.

This motivates the next mathematical tool.

---

# 24. Why Introduce X₀ Again?

Prathosh performs a seemingly strange step.

He rewrites

```text
Q(XT | XT−1)
```

as

```text
Q(XT | XT−1 , X₀)
```

At first,

this looks unnecessary.

Why add another conditioning variable?

The reason is subtle.

Because the forward process is a First-Order Markov Chain,

adding

```text
X₀
```

does **not** change the distribution.

The Markov Property tells us that

```text
XT

is conditionally independent of X₀

given XT−1.
```

Therefore,

both expressions represent exactly the same probability distribution.

Mathematically,

nothing has changed.

However,

this new form allows us to apply **Bayes' Rule** in the next step.

---

# 25. Why Is This Important?

Without introducing

```text
X₀
```

the next algebraic manipulation would not be possible.

Once

```text
X₀
```

appears,

Bayes' Rule connects quantities that are:

- already known,
- easy to compute,
- or already defined by the forward diffusion process.

This is the key idea behind the remaining derivation.

---

# Key Insights

- The DDPM ELBO is structurally identical to the VAE ELBO.
- The only difference is that DDPMs contain many latent variables instead of one.
- Both the numerator and denominator are expanded using the Chain Rule.
- Logarithms convert products into sums, greatly simplifying the objective.
- The reconstruction term is isolated because it directly involves the observed data.
- Adding `X₀` to the conditioning set does not change the distribution because of the Markov Property.
- This prepares the derivation for the application of Bayes' Rule.

---

# What's Next?

The next step is one of the most important in the entire DDPM derivation.

We will apply **Bayes' Rule** to rewrite the difficult forward conditional into
quantities that are already known.

This transformation introduces the reverse posterior

```text
Q(XT−1 | XT , X₀)
```

which eventually leads to the elegant KL-divergence formulation of the DDPM
training objective.
