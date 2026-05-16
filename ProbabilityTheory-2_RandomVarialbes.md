# Mathematical Foundations of Machine Learning
# Random Variables & Probability Framework

---

## 1. Probability Triplet

Probability theory starts with three components:

```text
(Ω, F, P)
```

where:

- `Ω` = Sample Space
- `F` = Set of events (subsets of sample space)
- `P` = Probability Measure

This is called the:

# Probability Triplet

---

## 2. Problem with Sample Space

Sample space is:
- abstract,
- non-numeric.

### Examples

- "getting diseased"
- "sending spam email"
- "coin toss outcome"

These are difficult to compute mathematically.

But engineers need:
- numbers,
- vectors,
- algebraic operations.

Hence we need a bridge between:
- abstract world,
- measurable numeric world.

---

## 3. Random Variable

A random variable creates this bridge.

It is defined as:

```text
X : Ω → R
```

or more generally:

```text
X : Ω → R^d
```

### Meaning

A random variable maps:
- outcomes of random experiments
to
- real numbers/vectors.

---

## 4. Important Insight

A random variable is:

# NOT random

It is actually:
- a deterministic function.

The randomness comes from:
- the underlying experiment,
- not from the function itself.

---

## 5. Why Random Variables are Needed

Without random variables:
- we cannot work numerically,
- cannot perform algebra,
- cannot process data computationally.

Random variables convert abstract events into measurable vectors.

---

## 6. Connection to Machine Learning

In ML:
- every data point is viewed as an element of the range space of a random variable.

### Example: Image

An image is represented as:

```text
x ∈ R^d
```

where:
- `d` = number of pixels/features.

### Probabilistic Interpretation

An image is:
- an element in `R^d`,
- generated from a random variable,
- defined on an underlying sample space.

---

## 7. Data Representation

Examples of vectors:
- images,
- text embeddings,
- speech signals,
- brain scans,
- time series.

All ML data becomes:
# vectors in high-dimensional space.

---

## 8. Sample Space → Random Variable → Data

The lecture gives the probabilistic pipeline:

### Step 1
Random experiment occurs.

### Step 2
Outcomes form sample space:

```text
Ω
```

### Step 3
Random variable maps outcomes to vectors:

```text
X : Ω → R^d
```

### Step 4
Observed data points are obtained.

---

## 9. Deterministic vs Probabilistic View

### Deterministic View

Image:

```text
x ∈ R^d
```

is just a vector.

### Probabilistic View

Image:
- element of range space of a random variable,
- generated from an underlying random experiment.

---

## 10. Sigma Algebra

When random variable maps sample space to real space:

```text
X : Ω → R^d
```

the event space `F` transforms into:

# Sigma Algebra

on:

```text
R^d
```

It represents measurable subsets/events in real space.

---

## 11. Push Forward Measure

The probability measure `P` on sample space also gets transformed.

This transformed measure becomes:

# Probability Distribution

on:

```text
R^d
```

This is the mathematical origin of:
- distributions,
- PDFs,
- probabilistic models in ML.

---

## 12. Key Machine Learning Insight

Modern ML does not only learn functions.

It learns:
# probability distributions over data.

---

## 13. Very Important Concept

The lecturer emphasizes:

All data should be viewed probabilistically.

An image is not "just an image."

It is:
- generated from a random experiment,
- mapped through a random variable,
- represented as a vector.

---

## 14. Main Flow of Concepts

```text
Random Experiment
        ↓
Sample Space
        ↓
Random Variable
        ↓
R^d
        ↓
Probability Distribution
```

---

## 15. Final Takeaway

Machine Learning works with:
- vectors numerically,

but theoretically those vectors come from:
- random variables,
- probability measures,
- underlying random experiments.

This probabilistic viewpoint forms the foundation of:
- statistical learning,
- generative models,
- deep learning,
- modern AI.
