# Mathematical Foundations of Machine Learning
# Probability Foundations for Machine Learning

# 1. Why Do We Need the Probabilistic Viewpoint?

Real-world systems are uncertain.

Many AI problems cannot be solved using exact physical equations because:
- the systems are too complex,
- observations are noisy,
- outcomes are uncertain.

So instead of modeling exact physics, machine learning uses:

# Statistical Modeling

Statistical modeling means:
- observing the system repeatedly,
- collecting data,
- learning patterns from observations.

## Key Idea

Probability helps us:
- model uncertainty,
- make predictions,
- learn patterns from repeated observations.

The more observations we have, the better statistical learning works.

---

# 2. Real-World Example: Automatic Speech Recognition (ASR)

Suppose a person speaks:

> "Hello, how are you?"

What actually happens physically?

## Step 1 — Physical Sound Waves

Speech creates:
- air pressure variations,
- vibrations in air.

These are physical phenomena.

## Step 2 — Microphone Conversion

A microphone converts:
- physical sound waves into
- electrical signals.

This process is done using:
- transducers,
- sensors,
- signal conversion systems.

## Step 3 — Machine Learning Processing

The machine receives:
- numerical signals,
- not actual "meaning."

But the goal is to predict:
- words,
- phonemes,
- meaning,
- intent.

## Important Insight

Machine learning maps:

```text
Physical Measurements
        ↓
Abstract Semantic Meaning
```

Examples of abstract concepts:
- language,
- emotion,
- intent,
- meaning.

These cannot be measured directly.

So probability and statistics are used to estimate them.

---

# 3. Observation Loop in Machine Learning

Machine learning works through repeated observations:

```text
Physical World
      ↓
Measurement
      ↓
Observation
      ↓
Learning Patterns
      ↓
Repeat Again
```

More observations usually improve:
- prediction quality,
- statistical estimation,
- model accuracy.

---

# 4. Generative Models

Generative models work in the reverse direction.

They take:
- abstract concepts,
- ideas,
- semantic meanings

and generate:
- physical outputs.

## Examples

### ChatGPT

Input:
- abstract prompt.

Output:
- human-like text.

### Image Generation Models

Input:
- "Generate a cat sitting on moon"

Output:
- actual image.

## Important Idea

Generative AI converts:

```text
Abstract Concept
        ↓
Physical Reality
```

---

# 5. Probability Space Triplet

Probability theory starts with three components:

$$
(\Omega, \mathcal{F}, P)
$$

This is called the:

# Probability Space / Probability Triplet

---

# 6. Sample Space (\Omega)

$$
\Omega
$$

represents:
- the set of all possible outcomes of a random experiment.

## Example 1: Coin Toss

$$
\Omega = \{H, T\}
$$

where:
- $H$ = Head
- $T$ = Tail

## Example 2: Dice Toss

$$
\Omega = \{1,2,3,4,5,6\}
$$

---

# 7. Outcomes

Elements of sample space are called:

# Outcomes

Usually represented by:

$$
\omega
$$

(small omega)

---

# 8. Event Space (\mathcal{F})

Subsets of sample space are called:

# Events

Represented by:

$$
\mathcal{F}
$$

## Example

If we toss a coin twice:

$$
\Omega = \{HH, HT, TH, TT\}
$$

Possible events:

$$
A = \{HH\}
$$

$$
B = \{HT, TH\}
$$

An event is simply:
- a subset of outcomes.

---

# 9. Probability Measure (P)

Probability measure assigns:
- a real number,
- to every event.

## Definition

$$
P : \mathcal{F} \rightarrow [0,1]
$$

Meaning:
- every event gets a probability value between 0 and 1.

## Important Properties

### Entire Sample Space

$$
P(\Omega) = 1
$$

Meaning:
- some outcome must happen.

### Empty Set

$$
P(\emptyset) = 0
$$

Meaning:
- impossible event has probability 0.

### Additivity Property

If two events are disjoint:

$$
A_1 \cap A_2 = \emptyset
$$

then:

$$
P(A_1 \cup A_2) = P(A_1) + P(A_2)
$$

---

# 10. Random Variable

A random variable is a function:

$$
X : \Omega \rightarrow \mathbb{R}
$$

It converts:
- outcomes of experiments into
- real numbers.

## Important Insight

A random variable is:
- NOT random.

It is a deterministic mathematical function.

The randomness comes from:
- the experiment itself.

---

# 11. Examples of Random Variables

## Coin Toss

$$
X(H) = 1
$$

$$
X(T) = 0
$$

The random variable converts:
- Head → 1
- Tail → 0

## Image Representation

An image becomes:

$$
x \in \mathbb{R}^d
$$

where:
- $d$ = number of pixels/features.

---

# 12. Higher Dimensional Spaces

$$
\mathbb{R}^2
$$

means:
- 2-dimensional vector space.

Example:

$$
(3,5)
$$

$$
\mathbb{R}^3
$$

means:
- 3-dimensional vector space.

Example:

$$
(1,2,7)
$$

---

# 13. Final Big Picture

Machine learning pipeline:

```text
Random Experiment
        ↓
Sample Space (Ω)
        ↓
Events (F)
        ↓
Probability Measure (P)
        ↓
Random Variable (X)
        ↓
Real Numbers / Vectors
        ↓
Probability Distribution
        ↓
Machine Learning
```
