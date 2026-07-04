# 11. Regularization, Bias-Variance Tradeoff and Advanced VAEs

> **Core Theme:** Understand why machine learning models overfit, how regularization controls model complexity, and why a Variational Autoencoder (VAE) is fundamentally a **regularized autoencoder**.

---

# 1. Where We Are in the Story

So far, we have studied:

* Why generative models learn probability distributions.
* Latent Variable Models.
* Variational Inference.
* Evidence Lower Bound (ELBO).
* Expectation Maximization (EM).
* Variational Autoencoders (VAEs).

We learned that a VAE optimizes the ELBO:

**ELBO = Reconstruction Term − KL Regularization Term**

But one important question still remains:

> **Why does the KL divergence term exist?**

To answer this question, we first need to understand one of the most fundamental concepts in Machine Learning:

**Regularization.**

This chapter builds that intuition from the ground up.

---

# 2. Machine Learning as a Parameter Estimation Problem

Almost every machine learning problem can be viewed as a **parameter estimation problem**.

Suppose we have:

* Unknown data distribution: **PX**
* Model distribution: **Pθ**

Our objective is to learn the model parameters **θ** such that the model distribution becomes as close as possible to the true data distribution.

Conceptually,

**Pθ ≈ PX**

One way to formulate this objective is:

Find θ that minimizes the divergence between the two distributions.

In previous chapters we showed that this is equivalent to **Maximum Likelihood Estimation (MLE).**

In practice, we optimize:

Average Log Likelihood over the training samples.

Since the true distribution PX is unknown, we approximate the expectation using the available dataset.

Thus, every machine learning algorithm ultimately learns **model parameters** from finite samples.

---

# 3. Why Learning from Finite Data is Difficult

Imagine we are trying to understand the height distribution of all humans.

Unfortunately, we cannot measure every human on Earth.

Instead, we collect a dataset of perhaps 10,000 people.

That dataset is only a **sample** from the true distribution.

Every time we collect another dataset, we obtain slightly different samples.

Therefore:

Different datasets produce different learned models.

This introduces one of the biggest challenges in Machine Learning:

> **How sensitive is our model to changes in the training dataset?**

This question naturally leads to the concepts of **Bias** and **Variance**.

---

# 4. Understanding Bias

Bias measures:

> **How well does the model explain the underlying data?**

A model with **low bias** closely matches the training data.

A model with **high bias** is unable to capture important patterns in the data.

### Intuition

Imagine drawing a smooth curve through several data points.

A model with low bias follows the overall trend closely.

A model with high bias is too simple and misses important relationships.

---

### Real-World Example

Suppose we want to predict house prices.

A model that always predicts:

**Price = ₹50 Lakhs**

for every house is clearly too simple.

Although it is easy to compute, it ignores:

* location
* size
* age
* amenities

This model has **very high bias** because it fails to represent the true relationship between the inputs and the target.

---

# 5. Understanding Variance

Variance measures:

> **How sensitive is the learned model to changes in the training data?**

Imagine training the same neural network on two slightly different datasets sampled from the same population.

If the learned models become dramatically different,

the model has **high variance**.

If both learned models are nearly identical,

the model has **low variance**.

---

### Intuition

Bias answers:

> "Does my model explain the data well?"

Variance answers:

> "Will my model change drastically if I collect a different dataset?"

A good model should perform well on **both** the current training data and future unseen data.

---

# 6. The Bias–Variance Tradeoff

Ideally, we would like:

* Low Bias
* Low Variance

Unfortunately, these two objectives often compete with each other.

Reducing one usually increases the other.

This relationship is called the **Bias–Variance Tradeoff**.

---

## Extreme Case 1 — Memorization

Suppose someone gives us 10 training examples.

Instead of learning patterns, we simply memorize every sample.

For every training example, we store the exact answer.

This model behaves like a lookup table.

Training Accuracy:

100%

But what happens when a completely new sample arrives?

The model has never seen it before.

It has no idea how to respond.

Therefore:

* Bias → Very Low
* Variance → Extremely High

The model has memorized the training data but has learned almost nothing about the underlying distribution.

---

## Extreme Case 2 — Constant Prediction

Now consider the opposite extreme.

Suppose the model always predicts the same output regardless of the input.

Example:

Every house costs ₹50 Lakhs.

This model behaves exactly the same for every dataset.

Changing the training samples hardly affects it.

Therefore:

* Variance → Very Low

However,

the model fails to explain almost every training sample.

Thus:

* Bias → Extremely High

---

# 7. Visualizing the Two Extremes

Imagine the following two models.

## Model A — Lookup Table

```
Training Data

A → 10
B → 15
C → 20
D → 25
```

Prediction:

```
A → 10 ✓
B → 15 ✓
C → 20 ✓
D → 25 ✓
```

New sample:

```
E → ?
```

The model has no answer.

This model has:

* Zero Bias
* Very High Variance

---

## Model B — Constant Model

Prediction:

```
Everything → 20
```

Training samples:

```
A → 20
B → 20
C → 20
D → 20
```

This model hardly changes even if the dataset changes.

Therefore:

* Very Low Variance
* Very High Bias

---

# 8. The Goal of Machine Learning

Neither extreme is desirable.

We do **not** want:

* A model that memorizes everything.
* A model that ignores everything.

Instead, we seek a balance.

The ideal model should:

* Capture the important structure in the data.
* Ignore random noise.
* Generalize well to unseen examples.

Finding this balance is one of the central goals of Machine Learning.

---

# Key Insights

* Every machine learning algorithm learns model parameters from finite data.
* Different datasets produce different learned models.
* **Bias** measures how well the model explains the data.
* **Variance** measures how sensitive the model is to changes in the training data.
* Memorization leads to **low bias** but **high variance**.
* Oversimplification leads to **high bias** but **low variance**.
* Successful machine learning requires balancing both.

---

# What's Next?

Now that we understand **Bias** and **Variance**, the next question naturally arises:

> **How do we intentionally control this tradeoff?**

The answer is **Regularization**, which we will study in the next section.

# 9. Underfitting and Overfitting

Understanding **Bias** and **Variance** naturally leads us to two important concepts in Machine Learning:

* Underfitting
* Overfitting

Every machine learning model falls somewhere between these two extremes.

---

## Underfitting

A model is said to **underfit** when it is **too simple** to capture the underlying patterns present in the data.

Characteristics of an underfitted model:

* High Bias
* Low Variance
* Poor training accuracy
* Poor test accuracy

### Intuition

Imagine trying to fit a straight line through data that actually follows a curved relationship.

The model is incapable of learning the true pattern because it is too simple.

### Example

Suppose house prices depend on:

* Location
* Area
* Number of bedrooms
* Age of the house

Now imagine using the model:

**Price = ₹50 Lakhs**

for every house.

The model ignores all useful information.

It underfits the data.

---

## Overfitting

A model is said to **overfit** when it memorizes the training data instead of learning the underlying pattern.

Characteristics of an overfitted model:

* Very Low Bias
* Very High Variance
* Extremely high training accuracy
* Poor performance on unseen data

### Intuition

Imagine a student preparing for an exam by memorizing every question from last year's paper.

If the same questions appear again, the student scores 100%.

If the questions change slightly, the student struggles.

The student has memorized rather than understood.

Machine learning models behave in exactly the same way.

---

# 10. Training Error vs Validation Error

One of the most important graphs in Machine Learning is the relationship between:

* Training Error
* Validation Error

As model complexity increases, both errors behave differently.

```text
Error
 ^
 |
 |\
 | \
 |  \____________________   Training Error
 |                    \
 |                     \
 |                      \
 |       Validation Error
 |      \
 |       \
 |        \
 |         \______
 |                \
 |                 \
 +----------------------------------------> Model Complexity
        Underfitting     Good Fit     Overfitting
```

### What happens initially?

When the model is very simple:

* Training error is high.
* Validation error is also high.

The model cannot even explain the training data.

This is the **underfitting region**.

---

### As the model becomes more expressive

Increasing the model complexity allows it to learn more useful patterns.

Both:

* Training Error
* Validation Error

begin to decrease.

Eventually, the validation error reaches its minimum.

This is the ideal operating point.

---

### What happens after that?

If we continue increasing the model complexity:

Training error keeps decreasing.

Eventually it may even become zero.

However,

Validation error starts increasing.

Why?

Because the model begins memorizing the training dataset rather than learning general patterns.

This is the **overfitting region**.

---

# 11. The Sweet Spot

The goal of Machine Learning is **not** to obtain the lowest training error.

The goal is to obtain the **lowest validation error**.

That point represents the model that generalizes best to unseen data.

Prathosh emphasizes an important observation:

> A model with 100% training accuracy is not necessarily a good model.

It may simply be overfitting the training data.

Always judge a model by how well it performs on **unseen data**, not by how perfectly it memorizes the training data.

---

# 12. What is Model Complexity?

Prathosh uses the term **Model Complexity** repeatedly.

But what exactly does it mean?

Model complexity refers to the model's capacity to represent complicated functions.

A more complex model can fit more intricate relationships.

Examples:

### Linear Regression

Very low complexity.

Can only fit straight lines.

---

### Polynomial Regression

Higher complexity.

Can fit curved relationships.

---

### Neural Networks

Complexity depends on:

* Number of layers
* Number of neurons
* Number of learnable parameters

Generally,

more parameters

↓

higher model complexity.

---

# 13. The Practical Strategy Used in Deep Learning

One of the most important practical insights from the lecture is this:

> Modern deep learning **does not begin with a small model.**

Instead,

we deliberately start with a model that is **large enough to overfit**.

Why?

Because it is easier to reduce overfitting than to increase model capacity later.

Prathosh summarizes the strategy as:

1. Build a sufficiently expressive model.
2. Verify that it can achieve very high training accuracy.
3. Apply regularization techniques.
4. Reduce overfitting.
5. Stop when validation performance is best.

This is the workflow followed in most modern deep learning projects.

---

# 14. What is Regularization?

Now we arrive at the central idea of this chapter.

Regularization is one of the most misunderstood concepts in Machine Learning.

Prathosh gives a very elegant interpretation.

> **Regularization is the deliberate process of increasing the bias of a model in order to reduce its variance.**

At first glance this sounds strange.

Why would we intentionally make our model "worse"?

The answer is simple.

A small increase in bias often leads to a **large reduction in variance**, allowing the model to generalize much better to unseen data.

In other words,

we intentionally sacrifice a little training accuracy to obtain much better real-world performance.

---

# 15. An Intuitive Analogy

Imagine learning to play cricket.

### Student A

Memorizes every ball bowled during practice.

Excellent during practice.

Fails in a real match.

High Variance.

---

### Student B

Learns general batting principles.

Footwork.

Timing.

Shot selection.

May not reproduce every practice ball perfectly.

But performs consistently in new situations.

Low Variance.

Regularization encourages the model to become more like **Student B**.

---

# 16. Early Stopping — The Simplest Regularization Technique

One of the simplest and most widely used regularization methods is **Early Stopping**.

Instead of training forever,

we monitor the validation error.

The moment validation performance starts degrading,

we stop training.

The model obtained at that point is used for deployment.

The workflow is:

1. Split the dataset into:

   * Training Set
   * Validation Set
   * Test Set

2. Train using only the Training Set.

3. After every epoch, evaluate the model on the Validation Set.

4. Continue training while validation performance improves.

5. Stop when validation performance starts becoming worse.

This prevents the model from entering the overfitting region.

---

# 17. Why Validation Data Exists

A common beginner question is:

> Why do we need a validation dataset?

The answer is that validation data is used to make **training decisions**, such as:

* Choosing the learning rate.
* Selecting the number of layers.
* Selecting the number of epochs.
* Deciding when to stop training.
* Choosing hyperparameters.

Validation data actively guides model development.

---

# 18. Why Test Data Must Never Be Used During Training

The **test dataset** has a completely different purpose.

It should remain untouched until the very end.

Using test accuracy to modify your model introduces **Test Data Leakage**.

For example:

1. Train a model.
2. Evaluate on the test set.
3. Notice poor performance.
4. Change the architecture.
5. Evaluate again on the same test set.

Although the test data was never directly used for gradient descent,

it still influenced the model design.

This is called **Test Data Leakage**.

It produces overly optimistic performance estimates.

Therefore,

**Validation Set → Model Development**

**Test Set → Final Evaluation Only**

---

# Key Insights

* Underfitting corresponds to **High Bias** and **Low Variance**.
* Overfitting corresponds to **Low Bias** and **High Variance**.
* The objective is to minimize **Validation Error**, not Training Error.
* Modern deep learning intentionally starts with models capable of overfitting.
* Regularization increases model bias to reduce variance.
* Early Stopping is one of the simplest and most effective regularization techniques.
* Validation data guides training.
* Test data should never influence model development.

# 19. Parameter Regularization

So far, we have understood that regularization is the process of **increasing model bias** to reduce variance.

The next question is:

> **How do we actually increase the bias of a model?**

One of the most common approaches is **Parameter Regularization**.

The basic idea is simple.

Instead of allowing the model parameters to take any arbitrary values,

we restrict the range of values they are allowed to take.

---

## Why Does This Increase Bias?

Suppose we are fitting a polynomial to ten data points.

If we allow the polynomial coefficients to become arbitrarily large,

the polynomial can twist and bend until it passes through every training point.

This produces:

* Very low training error
* Very low bias
* Very high variance

Now imagine restricting those coefficients.

For example,

instead of allowing weights like

100

250

−180

we encourage them to remain closer to zero.

The model loses some flexibility.

It can no longer perfectly memorize the training data.

As a result:

* Bias increases slightly.
* Variance decreases significantly.

This is exactly what regularization tries to achieve.

---

# 20. Parameter Penalty

Instead of minimizing only the learning objective,

we now optimize:

**Total Loss = Original Loss + Regularization Penalty**

The second term penalizes large parameter values.

The optimization objective becomes:

**Loss = Data Loss + α × Regularization Penalty**

where

* Data Loss measures how well the model fits the training data.
* α (alpha) controls the strength of regularization.

A larger value of α means stronger regularization.

---

# 21. L2 Regularization (Weight Decay)

The most common regularization technique is **L2 Regularization**.

Instead of allowing weights to grow without bound,

we penalize the squared magnitude of every parameter.

Conceptually:

Penalty = Sum of (weight²)

This encourages every weight to remain small.

### Why is it called Weight Decay?

During optimization,

large weights gradually shrink toward zero.

Hence the name:

**Weight Decay**

---

### Intuition

Imagine stretching a rubber band.

The farther you pull,

the stronger the restoring force.

L2 regularization behaves similarly.

The larger a weight becomes,

the stronger the optimization process tries to pull it back.

---

# 22. L1 Regularization

Another popular regularization method is **L1 Regularization**.

Instead of squaring the weights,

we penalize their absolute values.

Conceptually:

Penalty = Sum of |weight|

Unlike L2,

L1 encourages many parameters to become **exactly zero**.

Therefore,

L1 naturally performs **feature selection**.

The model automatically learns which parameters are useful and which can be ignored.

---

# 23. Comparing L1 and L2

| L1 Regularization                | L2 Regularization                    |
| -------------------------------- | ------------------------------------ |
| Uses absolute values             | Uses squared values                  |
| Produces sparse models           | Produces smooth models               |
| Many weights become exactly zero | Weights become small but rarely zero |
| Useful for feature selection     | Useful for improving generalization  |

Both methods increase the bias of the model,

thereby reducing variance.

---

# 24. A Different Way to Regularize

Until now,

we have been regularizing **model parameters**.

Prathosh now introduces a much deeper idea.

Instead of constraining the parameters,

why not constrain the **latent variables themselves?**

This is one of the key ideas behind Variational Autoencoders.

---

# 25. Neural Networks as Latent Variable Models

Consider a simple neural network.

```text
Input (X)
      │
      ▼
Layer 1 Output (Z₁)
      │
      ▼
Layer 2 Output (Z₂)
      │
      ▼
Layer 3 Output (Z₃)
      │
      ▼
Output (Y)
```

Notice something interesting.

The output of every hidden layer is **not directly observed**.

These hidden representations are exactly what we call **latent variables**.

Therefore,

a neural network itself can be viewed as a **latent variable model**.

Each hidden layer learns a new representation of the input.

---

# 26. Latent Variable Regularization

If every hidden layer is a latent variable,

then we can regularize:

* the model parameters θ

or

* the latent variables Z

Both approaches ultimately influence the learned model.

Instead of saying:

"Keep the weights small"

we can instead say:

"Restrict how the latent representations are allowed to behave."

This introduces another powerful class of regularization methods.

---

# 27. Why Does Regularizing Latent Variables Work?

Every latent variable is produced by the model parameters.

Therefore,

changing the latent variables indirectly changes the parameters.

Think of the relationship like this:

```text
Parameters (θ)
        │
        ▼
Latent Variables (Z)
        │
        ▼
Predictions
```

If we constrain the latent variables,

the parameters must automatically adjust themselves to satisfy those constraints.

Thus,

**Regularizing latent variables indirectly regularizes the model parameters.**

This is one of the most important conceptual ideas in this lecture.

---

# 28. Normalization as Regularization

Prathosh makes another beautiful observation.

Many techniques that practitioners use every day are actually forms of **latent-variable regularization**.

Examples include:

* Batch Normalization
* Layer Normalization

These methods do not directly modify the model parameters.

Instead,

they constrain the outputs of hidden layers.

Since hidden-layer outputs are latent variables,

normalization becomes a form of latent-space regularization.

---

# 29. Batch Normalization

Batch Normalization computes:

* Mean of a mini-batch
* Variance of a mini-batch

Every hidden activation is then transformed so that the batch approximately has:

* Mean = 0
* Variance = 1

Conceptually:

```text
Original Activations

2.1
3.7
5.8
1.5

↓

Normalize

Mean = 0
Variance = 1
```

By restricting the distribution of latent variables,

the model becomes more stable during training.

This also acts as a regularizer.

---

# 30. Layer Normalization

Layer Normalization follows the same idea,

but computes statistics across the **features within a single example** rather than across the batch.

Therefore:

### BatchNorm

Normalize across:

**Training Examples**

---

### LayerNorm

Normalize across:

**Feature Dimensions**

Layer Normalization is widely used in:

* Transformers
* Large Language Models
* GPT
* BERT

---

# 31. Why Does Normalization Increase Bias?

At first,

normalization looks like a simple mathematical transformation.

However,

Prathosh encourages us to think more deeply.

By forcing latent variables to follow a specific distribution,

we are restricting the space of possible representations.

The model now has **less freedom**.

Less freedom means:

* Higher Bias
* Lower Variance

Therefore,

normalization is fundamentally another form of regularization.

---

# 32. Architectural Regularization

Prathosh introduces one of the most insightful ideas of the lecture.

He argues that even **neural network architectures themselves** can be viewed as forms of regularization.

Think about a fully connected neural network.

Every neuron connects to every neuron in the next layer.

```text
Every Node
     │
     ▼
Every Node
```

Now consider a Convolutional Neural Network (CNN).

Each neuron connects only to a small local region.

Many possible connections are permanently removed.

Those missing connections can be viewed as parameters that are permanently fixed to zero.

In other words,

the architecture itself imposes constraints on the parameter space.

Those constraints increase model bias.

Hence,

CNNs are simply **strongly regularized fully connected neural networks.**

---

# 33. A Powerful Perspective

This viewpoint extends beyond CNNs.

Many modern architectures can be interpreted as different ways of introducing useful inductive biases.

Examples:

* CNN → Local connectivity bias
* RNN → Sequential dependency bias
* Transformer → Attention-based bias

Each architecture introduces a different kind of regularization that is well suited for a particular type of data.

This explains why:

* CNNs work well for images.
* RNNs work well for sequential data.
* Transformers work well for language.

The architecture itself becomes a way of controlling the Bias–Variance tradeoff.

---

# Key Insights

* Parameter regularization limits how large model weights can become.
* L2 regularization is also known as Weight Decay.
* L1 regularization encourages sparse models.
* Hidden layers of neural networks can be viewed as latent variables.
* Regularizing latent variables indirectly regularizes model parameters.
* Batch Normalization and Layer Normalization are both forms of latent-space regularization.
* Neural network architectures themselves can be interpreted as different forms of regularization.
* CNNs, RNNs, and Transformers introduce different inductive biases that improve generalization for different kinds of data.
