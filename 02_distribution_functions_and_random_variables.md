# Probability Foundations for Machine Learning
# Part 2 — Distribution Functions and Probability Distributions

---

# 1. Random Variable vs Probability Distribution

This is one of the most important concepts in probability.

---

# 2. Random Variable

A random variable maps outcomes into numbers.

$$
X : \Omega \rightarrow \mathbb{R}
$$

Example:

$$
X(H)=1
$$

$$
X(T)=0
$$

---

# 3. Probability Distribution

A probability distribution tells how probabilities are assigned over values of the random variable.

Example:

$$
P(X=1)=0.5
$$

$$
P(X=0)=0.5
$$

---

# 4. Core Difference

| Concept | Meaning |
|---|---|
| Random Variable | Converts outcomes into numbers |
| Probability Distribution | Assigns probabilities to those numbers |

---

# 5. Distribution Function

For a random variable:

$$
X : \Omega \rightarrow \mathbb{R}
$$

the induced distribution is:

$$
P_X
$$

This is called:
- Distribution Function
- CDF
- Pushforward Measure

---

# 6. Cumulative Distribution Function (CDF)

The CDF is defined as:

$$
F_X(x)=P(X \le x)
$$

Meaning:
Probability that the random variable takes value less than or equal to $x$.

---

# 7. Probability Density Function (PDF)

For continuous variables:

$$
p(x)\ge0
$$

and:

$$
\int_{-\infty}^{\infty} p(x)\,dx = 1
$$

Relationship between PDF and CDF:

$$
F_X(x)=\int_{-\infty}^{x} p_X(t)\,dt
$$

---

# 8. Important Clarification

For continuous random variables:

$$
P(X=x)=0
$$

A density value itself is NOT a probability.

Probabilities are obtained by integration over intervals.

---

# 9. Probability Mass Function (PMF)

For discrete variables:

$$
P(X=x_i)
$$

Example for dice:

$$
P(X=1)=\frac{1}{6}
$$

---

# 10. Higher Dimensional Distributions

In ML:

$$
X : \Omega \rightarrow \mathbb{R}^d
$$

Examples:
- images,
- embeddings,
- speech vectors.

The distribution now exists over multidimensional space.
