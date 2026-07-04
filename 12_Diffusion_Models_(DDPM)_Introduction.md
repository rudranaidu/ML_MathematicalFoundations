# 12. Diffusion Models (DDPM) – Introduction

> **Core Theme:** Learn how Diffusion Models gradually transform data into noise and then learn how to reverse that process to generate entirely new data.

---

# 1. Where We Are in the Story

Over the past few chapters, we have gradually built the mathematical foundation for modern generative models.

Our journey has been:

```text
Probability Theory

↓

Generative Models

↓

Latent Variable Models

↓

Variational Inference

↓

Variational Autoencoders (VAEs)

↓

Regularized Autoencoders

↓

Diffusion Models (DDPMs)
```

Notice that every chapter builds upon the previous one.

We first learned how to model probability distributions.

Then we introduced hidden (latent) variables.

Next, we learned how Variational Autoencoders approximate complex latent-variable models.

Now we move to the current state-of-the-art family of image generation models:

**Denoising Diffusion Probabilistic Models (DDPMs).**

---

# 2. Why Do We Need Another Generative Model?

At this point, a natural question arises.

> If we already have GANs and VAEs, why do we need Diffusion Models?

The answer lies in the limitations of previous models.

### Limitations of GANs

Although GANs generate highly realistic images,

they suffer from problems such as:

* Mode Collapse
* Difficult training
* Instability
* Sensitive hyperparameters

Training GANs often becomes an optimization challenge.

---

### Limitations of VAEs

VAEs solve many optimization problems elegantly.

They are mathematically beautiful and stable.

However,

their generated images are often blurrier than GANs.

Why?

Because the latent space is forced to follow a Gaussian prior,

which sometimes limits the quality of generated samples.

---

### Motivation for DDPMs

Researchers wanted a model that:

* Generates high-quality images.
* Is mathematically well-founded.
* Trains stably.
* Covers the entire data distribution.
* Avoids mode collapse.

Diffusion Models satisfy many of these requirements.

Today,

they power many modern image-generation systems.

---

# 3. The Big Idea Behind Diffusion Models

The central idea behind a Diffusion Model is surprisingly simple.

Instead of directly learning how to generate an image,

the model learns something much easier.

It learns how to **remove noise**.

Imagine the following experiment.

Take a clean photograph.

```text
😀
```

Now add a tiny amount of random noise.

```text
😀▒
```

Add more noise.

```text
▒▒😀▒▒
```

Keep adding noise.

```text
▓▒▓▒▒▓
```

Eventually,

the original image disappears completely.

Only random noise remains.

The forward process can therefore be visualized as:

```text
Original Image

↓

Slightly Noisy Image

↓

More Noisy Image

↓

Very Noisy Image

↓

Almost Pure Noise

↓

Pure Gaussian Noise
```

Diffusion Models learn the reverse process.

Starting from pure noise,

they gradually remove noise step by step until a realistic image emerges.

---

# 4. DDPMs are Hierarchical Latent Variable Models

One of the most important insights from Prathosh's lecture is:

> **A DDPM is a special kind of Hierarchical Variational Autoencoder.**

To understand this,

let us compare the architectures.

---

## Ordinary VAE

A Variational Autoencoder contains only one latent representation.

```text
Input (X)

↓

Encoder

↓

Latent Variable (Z)

↓

Decoder

↓

Reconstructed Input
```

Everything passes through a single latent space.

---

## Hierarchical VAE

A Hierarchical VAE introduces multiple latent spaces.

Instead of

```text
X

↓

Z
```

we now have

```text
X

↓

Z₁

↓

Z₂

↓

Z₃

↓

⋮

↓

ZT
```

Each latent space represents another level of abstraction.

Information gradually flows through many latent variables instead of one.

---

## DDPM

A Diffusion Model is exactly this idea,

with a few additional constraints.

Instead of learning one latent representation,

the model learns a sequence of latent variables.

Each latent variable is simply a slightly noisier version of the previous one.

Eventually,

the final latent variable becomes pure Gaussian noise.

---

# 5. Three Important Differences Between VAEs and DDPMs

Although DDPMs are closely related to VAEs,

they differ in three important ways.

| Variational Autoencoder             | Diffusion Model (DDPM)                  |
| ----------------------------------- | --------------------------------------- |
| One latent variable                 | Many latent variables                   |
| Encoder is learned                  | Encoder is fixed                        |
| Latent dimension is usually smaller | Latent dimension equals input dimension |

These three differences completely change how the model is trained.

---

# 6. A Change in Notation

Before studying DDPMs,

Prathosh introduces an important change in notation.

Until now,

our notation has been:

```text
Observed Data      → X

Latent Variable    → Z
```

However,

almost every research paper on Diffusion Models uses a different notation.

Instead,

they write:

```text
Observed Data      → X₀

Latent Variables   → X₁

                    X₂

                    X₃

                    ⋮

                    XT
```

This notation can initially feel confusing.

Remember:

* X₀ is **not** a latent variable.
* X₀ is the original data.

Everything after X₀ is a latent representation.

Prathosh deliberately switches notation so that the course matches the original DDPM literature.

---

# 7. Why Does the Literature Use X Instead of Z?

In previous chapters,

we used:

```text
Data      → X

Latent    → Z
```

Now,

both the data and latent variables use the same symbol X.

Why?

Because every latent representation in a DDPM has exactly the same dimensionality as the original data.

Each latent variable is simply another noisy version of the image.

Conceptually,

they are all images.

Only the amount of noise changes.

This is why researchers use

X₀

X₁

X₂

...

XT

instead of introducing a different symbol such as Z.

---

# 8. Intuition Behind Multiple Latent Variables

Imagine taking a photograph and slowly adding fog.

Initially,

the image is perfectly clear.

After a little fog,

the object is still recognizable.

After more fog,

only rough shapes remain.

Eventually,

everything disappears into white noise.

Each stage can be viewed as a different latent representation.

```text
Clear Image

↓

Light Fog

↓

Medium Fog

↓

Heavy Fog

↓

Only Noise
```

Instead of having a single latent representation,

a DDPM gradually transforms the data through many intermediate latent states.

This gradual transformation is the key idea behind diffusion models.

---

# Key Insights

* Diffusion Models are the next evolution of latent-variable generative models.
* They solve many practical limitations of GANs and VAEs.
* A DDPM can be viewed as a **Hierarchical Variational Autoencoder**.
* Instead of one latent variable, DDPMs use many latent variables.
* Every latent variable has the same dimensionality as the original data.
* The DDPM literature uses the notation **X₀, X₁, …, XT** instead of **X** and **Z**.
* The forward process gradually converts data into Gaussian noise.

---

# What's Next?

Now that we understand **why Diffusion Models use multiple latent variables**, the next question naturally arises:

> **How exactly is one latent variable transformed into the next?**

The answer is the **Forward Diffusion Process**, where we gradually add Gaussian noise according to a carefully designed mathematical rule.


# 9. The Forward Diffusion Process

Now that we understand **what a Diffusion Model is**, the next question is:

> **How do we transform an image into pure Gaussian noise?**

This transformation is called the **Forward Diffusion Process**.

Unlike a Variational Autoencoder,

the forward process in a DDPM is **not learned**.

It is a completely **fixed mathematical process**.

This is one of the biggest differences between VAEs and Diffusion Models.

---

# 10. Learned Encoder vs Fixed Encoder

Let's compare the two models.

## Variational Autoencoder

```text id="4wcpwx"
Image

↓

Encoder Neural Network

↓

Latent Vector
```

The encoder itself is learned during training.

Both the encoder and decoder contain trainable parameters.

---

## Diffusion Model

```text id="pwubab"
Image

↓

Fixed Noise Process

↓

Latent Variable X₁

↓

Fixed Noise Process

↓

Latent Variable X₂

↓

⋮

↓

Pure Noise XT
```

Nothing is learned here.

The forward process is completely predefined.

The model never changes it.

Instead,

the model only learns how to reverse it.

This is a major conceptual difference.

---

# 11. The Goal of the Forward Process

Suppose we start with an image.

```text id="d5rj0l"
🐱
```

Our objective is to slowly destroy the information contained in that image.

We do **not** destroy it all at once.

Instead,

we add a very small amount of Gaussian noise repeatedly.

Eventually,

after many steps,

the original image disappears completely.

The final result becomes indistinguishable from random Gaussian noise.

---

# 12. Why Add Noise Gradually?

You might wonder:

> Why not simply replace the image with Gaussian noise in one step?

That would certainly produce noise,

but it creates another problem.

The reverse process would become extremely difficult.

Imagine tearing an entire book into tiny pieces.

Reconstructing it is almost impossible.

Instead,

imagine erasing only one sentence at a time.

Recovering the original book now becomes much easier.

Diffusion Models follow exactly the same philosophy.

Many tiny reversible changes are much easier to learn than one huge irreversible change.

---

# 13. Building the Forward Process

Suppose

X₀

is the original image.

The first latent variable is obtained by adding a small amount of Gaussian noise.

Conceptually,

```text id="f5cgy0"
X₁

=

Original Image

+

Small Amount of Noise
```

The second latent variable is obtained by adding another small amount of noise.

```text id="x0r3n3"
X₂

=

X₁

+

Small Amount of Noise
```

The process continues:

```text id="frrm61"
X₃

=

X₂

+

Small Amount of Noise
```

Eventually,

```text id="ku1zh5"
XT

≈

Pure Gaussian Noise
```

Notice that every latent variable is simply a noisier version of the previous one.

---

# 14. The Mathematical Rule

The forward process follows a simple equation.

For every diffusion step,

Xt is computed from Xt−1 as:

```text id="1kmtr8"
Xt = √αt · Xt−1 + √(1 − αt) · εt
```

where

* Xt−1 is the previous latent variable.
* Xt is the new latent variable.
* εt is Gaussian noise sampled from N(0, I).
* αt controls how much of the previous image is preserved.

This single equation defines the entire forward diffusion process.

---

# 15. Understanding Every Symbol

Let us understand each component separately.

---

## Xt−1

This is simply the previous latent representation.

Initially,

```text id="fbsvde"
X₀ = Original Image
```

Later,

Xt−1 becomes a progressively noisier image.

---

## εt

ε (epsilon) represents random Gaussian noise.

Every diffusion step samples a completely new noise image.

Conceptually,

```text id="4qm4zr"
εt

↓

Random Noise

↓

Different at every step
```

Each diffusion step therefore introduces fresh randomness.

---

## αt

α controls how much information survives.

Suppose

α = 0.99

Almost the entire image is preserved.

Very little noise is added.

Suppose

α = 0.30

Only a small portion of the original image remains.

Much more noise is introduced.

Therefore,

α controls the speed of diffusion.

---

# 16. Why Are There Square Roots?

The equation uses:

√α

and

√(1 − α)

instead of simply α and (1 − α).

Why?

Because we want the total variance of the image to remain mathematically well behaved throughout the diffusion process.

The square roots guarantee that the contributions of:

* the previous image
* the new noise

combine correctly.

Although the derivation comes from probability theory,

the important intuition is simply:

The image and the noise are carefully balanced at every step.

---

# 17. The Noise Schedule

Prathosh mentions that

α is not a single constant.

Instead,

every diffusion step has its own value:

```text id="1rzbwv"
α₁

α₂

α₃

⋮

αT
```

These values are called the **Noise Schedule**.

Usually,

they decrease gradually.

Early diffusion steps preserve most of the image.

Later diffusion steps add progressively larger amounts of noise.

The exact schedule is chosen before training.

It is a hyperparameter,

not something the model learns.

---

# 18. Why Is the Forward Process Fixed?

One of the biggest conceptual questions is:

> Why don't we learn the encoder like we did in a VAE?

The answer is surprisingly elegant.

The purpose of the encoder is simply to convert any image into Gaussian noise.

We already know how to do this.

We simply keep adding Gaussian noise.

There is nothing to learn.

By fixing the encoder,

the optimization problem becomes much simpler.

The model now focuses entirely on learning the reverse process.

---

# 19. A Simple Analogy

Imagine placing a clean photograph under a sheet of frosted glass.

Each minute,

someone sprays a little more white paint onto the glass.

Initially,

the photograph is still recognizable.

After several sprays,

only rough outlines remain.

Eventually,

the entire photograph disappears behind white paint.

The forward diffusion process behaves exactly the same way.

It does not destroy the image immediately.

Instead,

it slowly hides the information behind increasing amounts of Gaussian noise.

---

# 20. Visualizing the Forward Process

The complete process can be visualized as:

```text id="3esck7"
X₀

Original Image

↓

X₁

Very Slight Noise

↓

X₂

More Noise

↓

X₃

Even More Noise

↓

⋮

↓

XT

Pure Gaussian Noise
```

Every stage is another latent representation.

The only difference is the amount of noise present.

---

# 21. Why This Is a Hierarchical Latent Variable Model

Notice something interesting.

Each latent variable depends only on the previous one.

```text id="gm50jb"
X₀

↓

X₁

↓

X₂

↓

X₃

↓

⋮

↓

XT
```

Instead of one latent space,

we now have a sequence of latent spaces.

Each latent variable contains slightly less information than the previous one.

This gradual hierarchy is why DDPMs are considered **Hierarchical Latent Variable Models**.

---

# Key Insights

* The forward diffusion process is completely fixed.
* No neural network is used for encoding.
* Every step adds a small amount of Gaussian noise.
* α controls how much information survives each diffusion step.
* ε represents newly sampled Gaussian noise.
* Repeatedly adding small amounts of noise eventually transforms every image into pure Gaussian noise.
* The forward process creates a hierarchy of latent variables that gradually lose information.

---

# What's Next?

Now that we know **how the forward diffusion process generates latent variables**, the next question is:

> **What probability distribution does each latent variable follow?**

To answer that,

we need to study the **Markov Chain formulation** of DDPMs and understand why every forward transition follows a Gaussian distribution.


# 22. The Forward Process is a First-Order Markov Chain

Now that we know how the forward diffusion process adds noise, we can study it from a probabilistic point of view.

Prathosh points out that the forward diffusion process is actually a **First-Order Markov Chain**.

This is one of the most important mathematical ideas behind DDPMs.

---

# 23. What is a Markov Chain?

A **Markov Chain** is a sequence of random variables where the future depends only on the present.

It does **not** depend on the entire history.

In simple words,

> **The next state depends only on the current state.**

---

## Real-World Example: Weather Prediction

Suppose today's weather is:

**Sunny**

To predict tomorrow's weather,

we usually use today's weather.

We do not necessarily need to know what the weather was:

* one month ago
* one year ago
* five years ago

Today's weather already contains enough information.

This is exactly the Markov Property.

---

## Another Example: Driving on a Road

Imagine driving from Bengaluru to Mysuru.

Your next location depends only on your **current location**.

It does not matter where you started your journey.

```text
Home

↓

Road A

↓

Road B

↓

Road C

↓

Destination
```

At every step,

your next position depends only on your current position.

---

# 24. How Does This Apply to DDPM?

The same idea is used in Diffusion Models.

Instead of locations,

our "states" are latent variables.

```text
X₀

↓

X₁

↓

X₂

↓

X₃

↓

⋮

↓

XT
```

Notice something important.

To compute **X₃**,

we only need **X₂**.

We do **not** directly use:

* X₀
* X₁

Similarly,

to compute **XT**,

we only need **XT−1**.

Therefore,

the forward process satisfies the **Markov Property**.

---

# 25. Why is it Called a First-Order Markov Chain?

The phrase **First-Order** simply means:

The next state depends on **only one previous state**.

Mathematically,

instead of saying

```text
XT depends on

X₀

X₁

X₂

...

XT−1
```

we simply say

```text
XT depends only on XT−1
```

That is exactly what our forward equation does:

```text
Xt = √αt · Xt−1 + √(1 − αt) · εt
```

The previous latent variable is sufficient.

Nothing else is required.

---

# 26. The Forward Transition Distribution

Since every diffusion step introduces randomness,

we can describe it using a probability distribution.

The distribution of

**Xt given Xt−1**

is written as

**Q(Xt | Xt−1)**

Read this as:

> "The probability distribution of the next latent variable given the current latent variable."

Every diffusion step has one such conditional distribution.

For example,

```text
Q(X₁ | X₀)

Q(X₂ | X₁)

Q(X₃ | X₂)

⋮

Q(XT | XT−1)
```

Together,

these completely describe the forward diffusion process.

---

# 27. Why is Every Forward Distribution Gaussian?

This is one of the most elegant observations in DDPMs.

Recall the forward equation:

```text
Xt = √αt · Xt−1 + √(1 − αt) · εt
```

Notice what is random here.

Xt−1 is already known.

α is a fixed constant.

The only random quantity is

ε,

which is sampled from a Gaussian distribution.

Therefore,

Xt is simply a transformed Gaussian random variable.

Whenever we:

* scale a Gaussian random variable,
* then shift it,

the result is still Gaussian.

This is the same idea we learned earlier while studying the **Reparameterization Trick** in Variational Autoencoders.

---

# 28. Connection to the Reparameterization Trick

Recall the VAE equation:

```text
Z = μ + σ × ε
```

where

ε follows a Standard Gaussian distribution.

The DDPM forward equation has exactly the same structure.

```text
Xt = √αt · Xt−1 + √(1 − αt) · εt
```

Both equations follow the same probabilistic principle:

Start with Gaussian noise,

apply scaling,

apply shifting,

the resulting variable is still Gaussian.

This is why every forward transition distribution is Gaussian.

---

# 29. The Forward Distribution

Therefore,

every transition in the forward process can be written as

```text
Q(Xt | Xt−1)

=

Gaussian Distribution
```

whose

Mean is

```text
√αt × Xt−1
```

and whose

Variance is

```text
1 − αt
```

This is an extremely important result.

It means that the entire forward process is completely known before training even begins.

Nothing about it has to be learned.

---

# 30. Why Is This Useful?

Because the encoder is completely known,

we immediately know:

* how latent variables are generated,
* what probability distributions they follow,
* how much noise is added at every step.

This dramatically simplifies the learning problem.

Instead of learning:

Encoder

*

Decoder

DDPM only needs to learn the reverse process.

---

# 31. What Happens After Many Steps?

Suppose we continue adding Gaussian noise.

```text
Image

↓

Image + Noise

↓

More Noise

↓

Even More Noise

↓

Almost Random

↓

Pure Noise
```

Eventually,

the original image disappears completely.

The remarkable result from probability theory is:

> If we continue this process for sufficiently many steps,

the final latent variable

XT

approaches a **Standard Gaussian Distribution**.

Conceptually,

```text
XT ≈ N(0, I)
```

Prathosh refers to this as the **stationary distribution** of the Markov Chain.

---

# 32. Intuition Behind the Stationary Distribution

Imagine repeatedly stirring a cup of coffee after adding milk.

Initially,

you can clearly distinguish:

* coffee
* milk

As you continue stirring,

the mixture becomes more and more uniform.

Eventually,

the cup reaches a stable state.

No matter how much longer you stir,

its appearance hardly changes.

This stable state is called the **stationary distribution**.

Similarly,

after repeatedly adding Gaussian noise,

every image eventually becomes indistinguishable from pure Gaussian noise.

---

# 33. Why Doesn't This Happen Immediately?

A common question is:

> Why don't we simply jump directly to Gaussian noise?

The answer is that gradual diffusion makes the reverse process learnable.

If we instantly replaced every image with random noise,

recovering the original image would become almost impossible.

Instead,

each diffusion step destroys only a tiny amount of information.

Therefore,

the reverse process only needs to learn tiny corrections at every step.

This is much easier than reconstructing an image from completely unrelated random noise in a single step.

---

# 34. The Entire Forward Process

Putting everything together,

the forward process looks like this.

```text
Original Image

X₀

↓

Q(X₁ | X₀)

↓

X₁

↓

Q(X₂ | X₁)

↓

X₂

↓

Q(X₃ | X₂)

↓

⋮

↓

Q(XT | XT−1)

↓

XT

≈ Pure Gaussian Noise
```

Every transition is:

* Gaussian
* Markovian
* Fixed
* Completely known

No learning happens anywhere in this pipeline.

---

# Key Insights

* The forward diffusion process is a **First-Order Markov Chain**.
* Every latent variable depends only on the previous latent variable.
* Every forward transition follows a Gaussian distribution.
* The DDPM forward equation has the same probabilistic structure as the VAE Reparameterization Trick.
* The encoder is completely fixed and known before training.
* Repeatedly adding Gaussian noise eventually transforms every image into a Standard Gaussian distribution.
* Because the forward process is fixed, DDPM training focuses entirely on learning the reverse (denoising) process.

---

# What's Next?

So far we have completely understood the **encoder** of a Diffusion Model.

However,

there is still no trainable model.

The actual learning begins with the **Reverse Diffusion Process**, where the model learns to remove noise one step at a time.

This reverse process defines the DDPM model itself and becomes the foundation for ELBO optimization.


# 35. The Reverse Diffusion Process

So far, we have completely defined the **forward diffusion process**.

We know:

* how the data is converted into noise,
* what probability distributions are involved,
* and why the forward process is fixed.

However, notice something important.

The forward process does **not** generate anything.

It only destroys information.

To build a generative model, we need the opposite process.

We need to learn:

> **How can we start from pure Gaussian noise and gradually reconstruct a realistic image?**

This is called the **Reverse Diffusion Process**.

---

# 36. The Reverse Process is the Decoder

Recall what happens in a Variational Autoencoder.

The encoder maps:

```text
Data

↓

Latent Space
```

The decoder performs the opposite operation.

```text
Latent Space

↓

Data
```

Exactly the same idea appears in DDPMs.

The forward process acts like the encoder.

The reverse process acts like the decoder.

Conceptually,

```text
Forward Process

Image

↓

Noise


Reverse Process

Noise

↓

Image
```

Unlike the forward process,

the reverse process is **unknown**.

This is the part we must learn.

---

# 37. Why is the Reverse Process Difficult?

Suppose someone gives you this image.

```text
██████████
██▓▒▓▒████
████▒▓████
██████████
```

Can you tell whether it originally came from:

* a cat?
* a dog?
* a car?
* a face?

Of course not.

Information has already been destroyed.

The challenge is to learn how to reverse this destruction.

This is exactly what a Diffusion Model learns.

---

# 38. The Reverse Transition Distribution

Earlier we defined the forward distribution:

**Q(Xt | Xt−1)**

which answered:

> Given the previous latent variable,
> what is the distribution of the next latent variable?

Now we define the opposite distribution.

**Pθ(Xt−1 | Xt)**

Read this as:

> Given the current noisy latent variable,
> what is the probability distribution of the previous (less noisy) latent variable?

Examples:

```text
Pθ(X₂ | X₃)

Pθ(X₁ | X₂)

Pθ(X₀ | X₁)
```

Each one removes a little bit of noise.

---

# 39. The Decoder Learns One Small Step at a Time

Notice something beautiful.

The decoder is **not** asked to recover the original image immediately.

Instead,

it learns a much easier problem.

For example,

instead of learning:

```text
Noise

↓

Original Image
```

it only learns:

```text
Very Noisy Image

↓

Slightly Less Noisy Image
```

Then,

```text
Slightly Less Noisy Image

↓

Even Cleaner Image
```

Eventually,

after many small corrections,

the original image appears.

---

# 40. Why Small Steps Make Learning Easier

Imagine restoring an old painting.

One approach is:

Paint the entire picture from scratch.

Very difficult.

Another approach is:

Remove one tiny layer of dust at a time.

Much easier.

DDPMs follow the second strategy.

Each reverse step removes only a tiny amount of noise.

This is one of the reasons why diffusion models produce remarkably realistic images.

---

# 41. The DDPM Model

Prathosh now defines the actual probabilistic model.

Remember,

the encoder is already fixed.

Therefore,

the only learnable part of the model is the reverse process.

The model consists of a sequence of reverse transition distributions:

```text
Pθ(XT−1 | XT)

Pθ(XT−2 | XT−1)

⋮

Pθ(X₂ | X₃)

Pθ(X₁ | X₂)

Pθ(X₀ | X₁)
```

Together,

these distributions define the complete generative model.

---

# 42. The Complete Reverse Chain

The reverse diffusion process can be visualized as:

```text
Pure Gaussian Noise

XT

↓

Pθ(XT−1 | XT)

↓

XT−1

↓

Pθ(XT−2 | XT−1)

↓

XT−2

↓

⋮

↓

Pθ(X₁ | X₂)

↓

X₁

↓

Pθ(X₀ | X₁)

↓

Generated Image
```

Notice that the reverse chain mirrors the forward chain exactly.

---

# 43. What Does the Neural Network Learn?

The forward process contains **no learnable parameters**.

Therefore,

the neural network focuses entirely on learning the reverse transition distributions.

Specifically,

for every diffusion step,

the network learns:

* how much noise should be removed,
* how the image should be updated,
* how to move one step closer to the original data distribution.

This is the only learning that happens inside a DDPM.

---

# 44. Why is the Reverse Distribution Gaussian?

Prathosh makes an important observation.

Since the forward process is a Gaussian Markov Chain,

probability theory tells us that the reverse process is also a Gaussian Markov Chain.

Therefore,

every reverse transition can also be modeled as a Gaussian distribution.

However,

there is one important difference.

The **form** of the distribution is known,

but its parameters are not.

Specifically,

we do not know:

* the mean
* the variance

These are exactly what the neural network must learn.

---

# 45. What Exactly Needs to be Learned?

Each reverse distribution has the form:

```text
Gaussian( Mean, Variance )
```

The Gaussian assumption is fixed.

What remains unknown are:

* Mean (μ)
* Variance (σ²)

The neural network learns these parameters from data.

Therefore,

training a DDPM ultimately becomes a parameter estimation problem.

Instead of directly generating images,

the model learns the Gaussian distributions that reverse the diffusion process.

---

# 46. The Overall Learning Pipeline

Putting everything together,

a DDPM works like this.

## Step 1

Start with a clean image.

```text
Image
```

---

## Step 2

Apply the fixed forward process.

```text
Image

↓

Gradually Add Noise

↓

Pure Gaussian Noise
```

---

## Step 3

Train a neural network to reverse every diffusion step.

```text
Noise

↓

Less Noise

↓

Cleaner Image

↓

Original Image
```

---

## Step 4

During inference,

start with completely random Gaussian noise.

Apply the learned reverse process repeatedly.

Eventually,

a completely new image is generated.

---

# 47. Why This is a Generative Model

Remember our original objective from Chapter 1.

We wanted to learn the data distribution.

After training,

the DDPM can generate completely new samples.

The inference process is:

```text
Random Gaussian Noise

↓

Learned Reverse Process

↓

New Image
```

Since every generated image is sampled from the learned data distribution,

DDPMs are true **Generative Models**.

---

# 48. Connecting DDPMs to VAEs

Although DDPMs look very different from VAEs,

they are surprisingly similar.

| Variational Autoencoder | Diffusion Model         |
| ----------------------- | ----------------------- |
| Learned Encoder         | Fixed Encoder           |
| Learned Decoder         | Learned Reverse Process |
| One Latent Variable     | Many Latent Variables   |
| ELBO Optimization       | ELBO Optimization       |

This is why Prathosh repeatedly emphasizes that:

> **A DDPM is a special type of Hierarchical Variational Autoencoder.**

The mathematics is still based on latent-variable modeling and ELBO optimization.

Only the architecture and the inference process are different.

---

# 49. Preparing for ELBO Optimization

At this point,

we have everything needed to formulate the learning problem.

We now know:

* how the encoder works,
* how the decoder is defined,
* what distributions are involved,
* which quantities are known,
* and which quantities must be learned.

The next step is exactly the same strategy we followed for VAEs.

We will derive an **Evidence Lower Bound (ELBO)** for the DDPM and optimize it.

The only difference is that the DDPM contains **many latent variables instead of one**.

---

# Chapter Summary

In this chapter, we introduced the foundations of Diffusion Models.

We learned that:

* DDPMs are hierarchical latent-variable models.
* The forward diffusion process is completely fixed.
* Noise is added gradually through a Gaussian Markov Chain.
* The final latent variable approaches pure Gaussian noise.
* The reverse process acts as the decoder.
* Only the reverse process is learned.
* The neural network learns the Gaussian transition parameters that progressively remove noise.
* The complete model remains a latent-variable model and will therefore be trained using ELBO optimization.

---

# Key Takeaways

✅ DDPMs generate data by learning to reverse a gradual noising process.

✅ The encoder is fixed; only the decoder is learned.

✅ Both forward and reverse processes are Gaussian Markov Chains.

✅ The neural network learns the reverse Gaussian distributions.

✅ DDPMs remain latent-variable generative models and are trained using ELBO optimization.

---

# Looking Ahead

In the next chapter,

we derive the **DDPM ELBO** step by step.

We will see how the familiar ELBO from Variational Autoencoders naturally extends to multiple latent variables and eventually leads to the practical loss function used to train modern diffusion models.

