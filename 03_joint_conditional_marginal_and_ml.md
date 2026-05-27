# Mathematical Foundations of Machine Learning
# Distribution Functions, Joint Probability and ML

---

# 1. Distribution Function

For a random variable:

$$
X : \Omega \rightarrow \mathbb{R}
$$

the induced distribution is denoted by:

$$
P_X
$$

This is called:
- Distribution Function
- Cumulative Distribution Function (CDF)
- Push Forward Measure

---

# 2. Cumulative Distribution Function

The CDF is defined as:

$$
F_X(x)=P(X \le x)
$$

Meaning:
- probability that random variable takes value less than or equal to $x$.

---

# 3. Vector-Valued Random Variables

In ML:

$$
X : \Omega \rightarrow \mathbb{R}^d
$$

Examples:
- images,
- embeddings,
- speech vectors.

---

# 4. Joint Distribution

Suppose:

$$
X_1 : \Omega \rightarrow \mathbb{R}
$$

$$
X_2 : \Omega \rightarrow \mathbb{R}
$$

Then the joint distribution is:

$$
P_{X_1X_2}
$$

---

# 5. Conditional Probability

Conditional probability means probability of $A$ given $B$.

$$
P(A \mid B)=\frac{P(A \cap B)}{P(B)}
$$

---

# 6. Marginal Distribution

Suppose we have:

$$
P(X,Y)
$$

Marginal distributions are:

$$
P(X)
$$

and

$$
P(Y)
$$

obtained by ignoring the other variable.

---

# 7. IID Assumption

Machine learning usually assumes data is:

- Independent
- Identically Distributed

---

# 8. Independence

Two events are independent if:

$$
P(A \cap B)=P(A)P(B)
$$

---

# 9. Dataset Representation

Most ML datasets look like:

$$
D=\{(x_1,y_1),(x_2,y_2),\dots,(x_n,y_n)\}
$$

where:
- $x_i$ = input/features
- $y_i$ = labels/outputs

---

# 10. Classification

Classification estimates:

$$
P(Y \mid X)
$$

where:

$$
Y \in \{1,2,\dots,K\}
$$

Examples:
- spam detection,
- disease prediction,
- sentiment analysis.

---

# 11. Regression

Regression predicts continuous outputs:

$$
Y \in \mathbb{R}^k
$$

Examples:
- house prices,
- temperatures,
- stock values.

---

# 12. Generative vs Discriminative Models

## Discriminative Models

Learn:

$$
P(Y \mid X)
$$

## Generative Models

Learn:

$$
P(X)
$$

or

$$
P(X,Y)
$$

and can generate new samples.

Examples:
- GANs
- Diffusion Models
- VAEs
- LLMs

---

# 13. Probability Density Function (PDF)

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

# 14. Important Clarification

For continuous random variables:

$$
P(X=x)=0
$$

A density value itself is NOT a probability.

Probabilities are obtained by integration over intervals.

---

# 15. Final Big Picture

Modern machine learning views data as:
- vectors generated from random variables,
- governed by probability distributions.

Learning means estimating these hidden probability distributions from data.

---

# 16. Main Conceptual Flow

$$
\text{Random Experiment}
\rightarrow
\Omega
\rightarrow
X
\rightarrow
\mathbb{R}^d
\rightarrow
P_X
\rightarrow
D
\rightarrow
\text{Machine Learning}
$$
