# Mathematical Foundations of Machine Learning
# Distribution Functions & Random Variables

---

## 1. From Probability Measure to Distribution Function

Earlier we defined the probability triplet:

```text
(Ω, F, P)
```

where:

- `Ω` = Sample Space
- `F` = Event Space
- `P` = Probability Measure

But engineers cannot directly work with abstract sample spaces.

We need:
- numbers,
- vectors,
- measurable quantities.

This is why random variables are introduced.

---

## 2. Random Variable

A random variable is a function:

```text
X : Ω → R
```

or more generally:

```text
X : Ω → R^d
```

It maps:
- abstract outcomes
- to real-valued vectors.

### Example: X-ray Problem

Underlying random experiment:
- patient gets disease,
- X-ray is taken.

Random variable:
- converts this abstract process into image vectors.

Output:

```text
x ∈ R^d
```

where:
- `d` = number of image pixels/features.

---

## 3. Borel Sigma Algebra

When sample space is mapped to real space, the subsets considered in `R` are called:

# Borel Sigma Algebra

### Example in R

Typical subsets:

```text
(-∞, x]
```

These measurable intervals are used to define distributions.

---

## 4. Distribution Function

Once probability measure is transferred from sample space to real space through a random variable, we obtain:

# Distribution Function

Also called:
- Probability Distribution Function
- Cumulative Distribution Function (CDF)
- Push Forward Measure

---

## 5. Definition of Distribution Function

For a random variable `X`:

```text
F_X(x) = P(X ≤ x)
```

Meaning:
- probability that random variable takes value less than or equal to `x`.

---

## 6. Important Interpretation

Distribution function evaluates:
- probabilities of subsets in real space.

It is derived from:
- probability measure on sample space.

---

## 7. Inverse Image Concept

To compute:

```text
F_X(x)
```

we:

1. take interval:

```text
(-∞, x]
```

2. find its inverse image in sample space under random variable `X`

3. apply probability measure `P`

---

## 8. Key Idea

Random variables transform:

```text
Probability Measures on Sample Spaces
            ↓
Probability Distributions on Real Spaces
```

---

## 9. Vector-Valued Random Variables

In ML, random variables usually map into:

```text
R^d
```

instead of just `R`.

These are called:

# Vector-Valued Random Variables

### Examples

Image:

```text
x ∈ R^10000
```

Text embedding:

```text
x ∈ R^768
```

Speech signal:

```text
x ∈ R^d
```

---

## 10. Distribution in Higher Dimensions

For vector-valued random variables:

```text
X : Ω → R^d
```

distribution functions are defined over multidimensional regions.

### Example in 2D

Consider:

```text
(-∞, x1] × (-∞, x2]
```

This represents a rectangular region in 2D space.

---

## 11. Multidimensional Distribution Function

For vector random variable:

```text
X = (X1, X2, ..., Xd)
```

distribution becomes:

```text
F_X(x1, x2, ..., xd)
```

representing joint probabilities.

---

## 12. Joint Distributions

Multiple random variables can interact.

### Example

- disease occurrence,
- X-ray image generation.

These may come from:
- separate random experiments,
- or one combined random experiment.

Their interaction is modeled using:

# Joint Distributions

---

## 13. Conditional Distributions

Conditional distributions describe probabilities under conditions.

### Example

```text
P(Disease | X-ray)
```

Meaning:
- probability of disease given X-ray image.

Applications:
- classification,
- prediction,
- Bayesian learning.

---

## 14. Important ML Insight

In modern ML:
- labels,
- images,
- text,
- signals

can all be viewed as components of one vector-valued random variable.

---

## 15. Discriminative vs Generative View

### Discriminative Models

Learn:

```text
P(Y | X)
```

### Generative Models

Learn:

```text
P(X, Y)
```

Both are distribution learning problems.

---

## 16. Very Important Concept

Machine Learning is fundamentally:

# Distribution Estimation

not just function approximation.

---

## 17. Important Warning

A distribution function is a valid probability measure.

But:

# Density function values are NOT probabilities.

Probability density functions must be integrated over regions to obtain probabilities.

---

## 18. Main Flow of Ideas

```text
Random Experiment
        ↓
Sample Space
        ↓
Random Variable
        ↓
R^d
        ↓
Distribution Function
        ↓
Probability Estimation
```

---

## Final Takeaway

Modern machine learning treats data as:
- vectors generated from random variables,
- governed by probability distributions.

Learning means:
- estimating these underlying distributions from data.
