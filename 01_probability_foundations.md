# Probability Foundations for Machine Learning
# Part 1 — Probability, Sample Spaces, and Random Variables

---

# 1. Why Probability is Important in Machine Learning

Machine Learning works with uncertain real-world data.

Real-world observations are never perfectly predictable:

- Speech can have noise
- Images can be blurry
- Medical scans may vary
- Human language is ambiguous

Probability helps us model uncertainty mathematically.

---

# 2. Probability Space

Probability theory starts with:

$$
(\Omega, \mathcal{F}, P)
$$

where:

- $\Omega$ = Sample Space
- $\mathcal{F}$ = Event Space
- $P$ = Probability Measure

---

# 3. Sample Space

The sample space is the set of all possible outcomes.

Example:

$$
\Omega = \{H, T\}
$$

---

# 4. Events

Events are subsets of the sample space.

Example:

$$
A = \{H\}
$$

---

# 5. Probability Measure

Probability assigns numbers to events.

$$
P : \mathcal{F} \rightarrow [0,1]
$$

Properties:

$$
P(\Omega)=1
$$

$$
P(\emptyset)=0
$$

If:

$$
A_1 \cap A_2 = \emptyset
$$

then:

$$
P(A_1 \cup A_2)=P(A_1)+P(A_2)
$$

---

# 6. Random Variables

A random variable converts outcomes into numbers.

$$
X : \Omega \rightarrow \mathbb{R}
$$

More generally in ML:

$$
X : \Omega \rightarrow \mathbb{R}^d
$$

---

# 7. Why Random Variables Are Needed

Machine learning algorithms work with vectors and numbers.

Examples:
- images,
- speech,
- text,
- signals

must all become vectors.

---

# 8. Vector-Valued Random Variables

Images are represented as high-dimensional vectors.

Example:

A grayscale image of size:

$$
P \times Q
$$

becomes:

$$
x \in \mathbb{R}^{PQ}
$$

---

# 9. Important Insight

A random variable is NOT random.

It is a deterministic mathematical mapping.

The randomness comes from the experiment itself.

---

# 10. Main ML Pipeline

$$
\text{Random Experiment}
\rightarrow
\Omega
\rightarrow
X
\rightarrow
\mathbb{R}^d
$$

Modern ML treats data as vectors generated from random variables.
