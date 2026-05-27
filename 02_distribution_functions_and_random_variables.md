# Mathematical Foundations of Machine Learning
# Random Variable vs Probability Distribution

A very important question in probability and machine learning is:

> What is the difference between a Random Variable and a Probability Distribution?

Many beginners confuse these two concepts.

The key difference is:

- Random Variable converts abstract outcomes into numbers.
- Probability Distribution tells how probabilities are assigned to those numbers.

---

# 1. Big Picture

Suppose we toss a coin.

The random experiment is:

> Tossing a coin

Sample space:

$$
\Omega = \{H, T\}
$$

where:
- $H$ = Head
- $T$ = Tail

---

# 2. Random Variable

Now define a random variable:

$$
X(H) = 1
$$

$$
X(T) = 0
$$

What happened?

The random variable converted:

- $H \rightarrow 1$
- $T \rightarrow 0$

So instead of working with abstract symbols:
- H
- T

we can now work with numbers:
- 0
- 1

This is the job of a:

# Random Variable

---

# 3. What Does Probability Distribution Do?

Now we ask:

> How likely is each number?

We know:

$$
P(H) = 0.5
$$

$$
P(T) = 0.5
$$

Since:

$$
X(H) = 1
$$

$$
X(T) = 0
$$

the probabilities get transferred onto numbers.

So we obtain:

$$
P(X = 1) = 0.5
$$

$$
P(X = 0) = 0.5
$$

This is called the:

# Probability Distribution

---

# 4. Core Difference

| Concept | Purpose |
|---|---|
| Random Variable | Converts outcomes into numbers |
| Probability Distribution | Assigns probabilities to those numbers |

---

# 5. Mathematical Difference

## Random Variable

A function:

$$
X : \Omega \rightarrow \mathbb{R}
$$

Input:
- sample space outcomes.

Output:
- real numbers.

## Probability Distribution

A measure on real numbers.

It tells:

> How probability is distributed over values of $X$

---

# 6. Another Example — Dice

## Step 1: Sample Space

$$
\Omega = \{1,2,3,4,5,6\}
$$

## Step 2: Random Variable

Suppose:

$$
X(\omega) = \omega
$$

(identity mapping)

## Step 3: Probability Distribution

Distribution becomes:

| X value | Probability |
|---|---|
| 1 | 1/6 |
| 2 | 1/6 |
| 3 | 1/6 |
| 4 | 1/6 |
| 5 | 1/6 |
| 6 | 1/6 |

---

# 7. In Machine Learning

Random variables convert:
- images,
- speech,
- text,
- physical measurements

into vectors.

Example:

$$
X : \Omega \rightarrow \mathbb{R}^d
$$

Image becomes:

$$
x \in \mathbb{R}^d
$$

Probability Distribution then asks:

- What is the probability of seeing this image?
- What is the probability of this label given this image?

Examples:

$$
P(X)
$$

$$
P(Y|X)
$$

$$
P(X,Y)
$$

---

# 8. Most Important Intuition

Think like this:

## Random Variable

Creates the coordinate system.

## Probability Distribution

Describes how data is spread inside that coordinate system.

---

# 9. Final One-Line Difference

## Random Variable

Maps:

$$
\text{outcomes} \rightarrow \text{numbers}
$$

## Probability Distribution

Maps:

$$
\text{numbers} \rightarrow \text{probabilities}
$$
