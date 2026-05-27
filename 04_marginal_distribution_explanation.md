# Marginal Distribution — Intuitive Explanation for Machine Learning

# 1. What is Marginal Distribution?

Marginal distribution means:

> studying one variable while ignoring the other variables.

Suppose we have two random variables:

$$
X
$$

and

$$
Y
$$

Together they form a:

# Joint Distribution

$$
P(X,Y)
$$

This distribution studies both variables together.

---

# 2. Real-Life Example — Students in a Classroom

Suppose we collect data about students:

- Height
- Weight

| Student | Height (X) | Weight (Y) |
|---|---|---|
| A | 170 | 65 |
| B | 160 | 55 |
| C | 170 | 70 |
| D | 180 | 80 |

This dataset represents samples from:

$$
P(X,Y)
$$

because each student has:
- a height,
- and a weight.

---

# 3. What Does Joint Distribution Mean?

Joint distribution studies relationships between variables.

Example questions:

- What is the probability that:
  - Height = 170 cm
  - AND Weight = 65 kg?

This is a question about:

$$
P(X,Y)
$$

---

# 4. Marginal Distribution of Height

Now suppose we only care about:

> "How are heights distributed?"

We completely ignore weight.

Then we study:

$$
P(X)
$$

This is called:

# Marginal Distribution of X

---

# 5. What Does “Ignoring Y” Mean?

Originally data points looked like:

$$
(height, weight)
$$

Example:

$$
(170,65)
$$

Now we remove weight:

$$
170
$$

The dataset becomes:

| Height |
|---|
| 170 |
| 160 |
| 170 |
| 180 |

Now we can ask:

- how common is 170 cm?
- how common is 180 cm?

This distribution is:

$$
P(X)
$$

---

# 6. Marginal Distribution of Weight

Similarly, if we ignore height:

| Weight |
|---|
| 65 |
| 55 |
| 70 |
| 80 |

then we study:

$$
P(Y)
$$

This is called:

# Marginal Distribution of Y

---

# 7. Why is it Called “Marginal”?

Historically, statisticians wrote probability tables.

Example:

| Height | Weight | Probability |
|---|---|---|
| 170 | 65 | 0.2 |
| 170 | 70 | 0.3 |
| 180 | 80 | 0.5 |

To compute probabilities for height alone:

they added probabilities row-wise and wrote results on the margins (sides) of the table.

Hence the name:

# Marginal Distribution

---

# 8. Visual Intuition

## Joint Distribution

```text
(Height, Weight)
```

## Marginal Distribution of Height

```text
Height only
```

## Marginal Distribution of Weight

```text
Weight only
```

---

# 9. Machine Learning Example

Suppose:

- $$X$$ = image
- $$Y$$ = label

Example dataset:

| Image | Label |
|---|---|
| Cat image | Cat |
| Dog image | Dog |
| Cat image | Cat |

Together they form:

$$
P(X,Y)
$$

---

# 10. Marginal Distribution of Images

Suppose we ignore labels.

Now we only study images.

This becomes:

$$
P(X)
$$

Questions answered:
- what kinds of images occur frequently?
- how are images distributed?

This is important in:
- generative AI,
- image generation,
- unsupervised learning.

---

# 11. Marginal Distribution of Labels

If we ignore images and only study labels:

$$
P(Y)
$$

Questions answered:
- how many cat images exist?
- how many dog images exist?

Example:

| Label | Probability |
|---|---|
| Cat | 0.7 |
| Dog | 0.3 |

---

# 12. Mathematical Definition

Suppose joint distribution is:

$$
P(X,Y)
$$

Then marginal distribution of $$X$$ is obtained by summing or integrating over all possible values of $$Y$$.

## Discrete Case

$$
P(X)=\sum_Y P(X,Y)
$$

## Continuous Case

$$
P(X)=\int P(X,Y)\,dY
$$

Meaning:

> collapse/remove the other variable.

---

# 13. Simple Intuition

## Joint Distribution

Studies variables together.

$$
P(X,Y)
$$

## Marginal Distribution

Studies one variable alone.

$$
P(X)
$$

or

$$
P(Y)
$$

---

# 14. Why Marginal Distributions Matter in ML

Marginal distributions are extremely important in Machine Learning.

## Example 1 — Generative AI

Image generators learn:

$$
P(X)
$$

distribution of images.

---

## Example 2 — Classification

Classifiers learn:

$$
P(Y|X)
$$

labels given images.

---

## Example 3 — Dataset Bias

Suppose:

$$
P(Y)
$$

is very unbalanced.

Example:
- 95% healthy patients
- 5% diseased patients

Then ML models may become biased.

---

# 15. Final One-Line Intuition

## Joint Distribution

"Study everything together"

$$
P(X,Y)
$$

## Marginal Distribution

"Study one variable after ignoring the others"

$$
P(X)
$$

or

$$
P(Y)
$$
