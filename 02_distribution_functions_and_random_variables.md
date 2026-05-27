# Distribution Functions and Random Variables

## 1. Random variable vs probability distribution
A very important question in probability and machine learning is:

**What is the difference between a random variable and a probability distribution?**

Beginners often confuse them.

The clean distinction is:
- **Random variable** converts abstract outcomes into numbers.
- **Probability distribution** tells how probabilities are assigned to those numbers.

---

## 2. Simple coin-toss example
Suppose we toss a coin.

Sample space:
\[
\Omega = \{H, T\}
\]

Define a random variable:
- X(H) = 1
- X(T) = 0

Now the abstract outcomes H and T are converted into numbers 1 and 0.

The probability distribution then tells us how likely those numbers are:
- P(X = 1) = 0.5
- P(X = 0) = 0.5

---

## 3. Core difference
| Concept | Purpose |
|---|---|
| Random Variable | Converts outcomes into numbers |
| Probability Distribution | Assigns probabilities to those numbers |

Another way to remember it:
- Random variable = coordinate system
- Probability distribution = how data is spread inside that coordinate system

---

## 4. Mathematical difference
A random variable is a function:

\[
X : \Omega \to \mathbb{R}
\]

A probability distribution is a measure on the range space of X. It tells how probability is distributed over values of X.

---

## 5. Dice example
Sample space:
\[
\Omega = \{1,2,3,4,5,6\}
\]

If we define the identity mapping \(X(\omega)=\omega\), then the distribution becomes:

| X value | Probability |
|---|---|
| 1 | 1/6 |
| 2 | 1/6 |
| 3 | 1/6 |
| 4 | 1/6 |
| 5 | 1/6 |
| 6 | 1/6 |

---

## 6. Distribution function / CDF / pushforward measure
Once probability is transferred from the sample space through a random variable, we obtain the **distribution function**.

This is also called:
- cumulative distribution function (CDF)
- probability distribution function in lecture notes
- pushforward measure

The notation often used is:
\[
P_X
\]

It means the distribution induced by the random variable X.

---

## 7. Meaning of distribution function
For a set A in the range space,
\[
P_X(A)
\]
means the probability that the random variable X falls inside A.

So the distribution is the probability structure on the values produced by the random variable.

---

## 8. CDF definition
For a scalar random variable X, the cumulative distribution function is:

\[
F_X(x)=P(X\le x)
\]

This gives the probability that X takes a value less than or equal to x.

---

## 9. Inverse-image idea
To compute \(F_X(x)\):
1. take the interval \((-\infty, x]\),
2. find its inverse image under X,
3. apply the probability measure P on the sample space.

This is why random variables connect abstract outcomes to measurable probabilities.

---

## 10. Borel sigma-algebra
In real-valued probability theory, the measurable subsets of \(\mathbb{R}\) are typically taken from the **Borel sigma-algebra**.

Simple intervals such as:
\[
(-\infty, x]
\]
are examples of measurable sets used when defining distributions.

---

## 11. Vector-valued random variables
In machine learning, random variables usually map into vectors:

\[
X : \Omega \to \mathbb{R}^d
\]

These are called **vector-valued random variables**.

### Examples
- image embedding in \(\mathbb{R}^{10000}\)
- text embedding in \(\mathbb{R}^{768}\)
- speech feature vector in \(\mathbb{R}^d\)

---

## 12. Multidimensional distributions
For a vector random variable:

\[
X=(X_1, X_2, \dots, X_d)
\]

the distribution is defined over regions in \(\mathbb{R}^d\).

For example, in 2D a rectangular region can be written as:

\[
(-\infty, x_1] \times (-\infty, x_2]
\]

---

## 13. Distribution function in machine learning
This idea is central in ML because all data is ultimately represented numerically.

A data point such as an image, a speech signal, or a text embedding can be treated as an element of the range space of a random variable. The distribution tells us how likely different data points are.

---

## 14. Density function and mass function
For continuous random variables, we often work with a **probability density function (PDF)**.

For discrete random variables, we use a **probability mass function (PMF)**.

Important warning:
- PDF values are not probabilities by themselves.
- Probabilities for continuous variables come from integrating the density over regions.

---

## 15. Final takeaway
A random variable converts outcomes into numbers or vectors. A probability distribution describes how probability is assigned to those numbers or vectors. In machine learning, this is the bridge between abstract uncertainty and numerical computation.
