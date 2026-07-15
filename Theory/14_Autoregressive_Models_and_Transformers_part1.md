# 14_Autoregressive_Models_and_Transformers.md

> **Core Theme:** Learn how modern Large Language Models (LLMs) generate text one token at a time using autoregressive modeling, and understand why the Transformer architecture replaced Recurrent Neural Networks (RNNs).

---

# 1. Where We Are in the Story

So far in this course, we have studied three major families of generative models.

```text
Generative Models

│

├── GANs
│      │
│      └── Adversarial Learning

├── Variational Autoencoders
│      │
│      └── Latent Variable Models

└── Diffusion Models
       │
       └── Denoising Process
```

These models are extremely successful for generating:

- Images
- Audio
- Videos

However,

there is one type of data that behaves very differently.

That data is **text**.

---

# 2. Why is Text Different?

Suppose I ask you to complete the following sentence.

```text
I went to the restaurant and ordered ______
```

What words naturally come to your mind?

Perhaps

- pizza
- dosa
- biryani
- coffee

Now consider another sentence.

```text
I went to the bank to withdraw ______
```

The answer now changes completely.

Possible words become

- cash
- money
- ₹5000

Notice something important.

The next word depends on **all the previous words**.

This is very different from image generation.

Images do not have a natural ordering.

Text does.

This is why language models require a completely different approach.

---

# 3. Sequential Data

Data where the order of observations matters is called **Sequential Data**.

Examples include:

- Natural language
- Speech
- Music
- Stock prices
- Sensor measurements
- DNA sequences
- Videos

The key characteristic is:

> **Changing the order changes the meaning.**

---

## Example

Consider the sentence

```text
Dog bites man.
```

Now reverse the order.

```text
Man bites dog.
```

The words are exactly the same.

Only the order changed.

Yet the meaning is completely different.

This is why sequence modeling must preserve order.

---

# 4. What is a Sequence?

We now introduces a slight change in notation.

Until now,

a single data point was simply written as

```text
X
```

For language models,

one data point is **not one vector**.

Instead,

one data point is an entire sequence.

Conceptually,

```text
X

=

(X₁ , X₂ , X₃ , ... , XT)
```

where

- T is the sequence length.
- Each Xi is called a **token**.

Notice something important.

The entire sequence is still **one data point**.

During training,

we will have many such sequences.

However,

the mathematical derivation focuses on **one sequence at a time**.

This is exactly the same philosophy we used in VAEs and DDPMs.

---

# 5. What is a Token?

One of the most important concepts in Large Language Models is the **Token**.

A token is the smallest unit processed by the language model.

Many beginners think:

> One token = One word

This is **not always true**.

A token could be:

- a word
- part of a word
- punctuation
- a number
- a special symbol

For now,

you can safely imagine one token as one word.

Later,

we will study Byte Pair Encoding (BPE), which explains how modern LLMs create tokens.

---

# 6. Representing a Sequence

Suppose our sentence is

```text
I

love

Machine

Learning
```

The sequence becomes

```text
X₁

↓

"I"

X₂

↓

"love"

X₃

↓

"Machine"

X₄

↓

"Learning"
```

Each token will eventually be converted into a numerical vector.

Therefore,

a sequence can be viewed as

```text
X₁

X₂

X₃

...

XT
```

where every Xi is a vector.

---

# 7. Every Token is a Vector

Just like images become tensors,

words also become vectors.

Suppose every token is represented using

128 numbers.

Then

```text
"I"

↓

128-dimensional vector

"love"

↓

128-dimensional vector

"Machine"

↓

128-dimensional vector

...
```

Therefore,

a sequence is nothing more than

a collection of vectors.

This is exactly the notation we uses throughout the lecture.

---

# 8. Why Does Every Token Become a Vector?

Neural networks understand numbers.

They do **not** understand English.

Consider the word

```text
Apple
```

To us,

it represents

- a fruit,
- sometimes a company,
- sometimes a logo.

To a neural network,

it is initially just text.

Therefore,

we first convert it into numbers.

This numerical representation is called an **Embedding**.

We will study embeddings in detail later.

For now,

simply remember:

```text
Word

↓

Vector
```

---

# 9. The Goal of Language Modeling

Now the question becomes:

> Given a sequence of tokens,

how do we learn its probability distribution?

Exactly the same objective appears again.

```text
Unknown Distribution

↓

Estimate

↓

Model Distribution
```

Our goal remains

```text
Pθ ≈ PX
```

Nothing has changed.

Only the nature of the data has changed.

---

# 10. Autoregressive Modeling

We introduce a completely new family of generative models.

These are called

**Autoregressive Models**.

Instead of generating an entire sequence at once,

they generate one token at a time.

Conceptually,

```text
Token 1

↓

Predict Token 2

↓

Predict Token 3

↓

Predict Token 4

↓

...
```

The model repeatedly asks one question.

> **Given everything I have already seen, what is the most likely next token?**

This simple idea powers almost every modern LLM.

---

# 11. Real-Life Intuition

Imagine someone starts singing a familiar song.

```text
Twinkle

Twinkle

Little
```

Most people can immediately predict

```text
Star
```

Why?

Because your brain has learned the probability of word sequences.

Language models attempt to do exactly the same thing.

They predict the next token based on the previous tokens.

---

# 12. The Chain Rule of Probability

How can we represent the probability of an entire sequence?

Probability theory already provides the answer.

It is called the **Chain Rule of Probability**.

Conceptually,

the probability of the entire sequence becomes

```text
P(X₁, X₂, ..., XT)

=

P(X₁)

×

P(X₂ | X₁)

×

P(X₃ | X₁, X₂)

×

...

×

P(XT | X₁, X₂, ..., XT−1)
```

This equation is the mathematical foundation of every autoregressive language model.

---

# 13. Understanding the Chain Rule Intuitively

Suppose we want to generate

```text
I love Machine Learning
```

Instead of generating the whole sentence at once,

we generate one word at a time.

```text
Step 1

Generate

"I"

↓

Step 2

Predict

"love"

given

"I"

↓

Step 3

Predict

"Machine"

given

"I love"

↓

Step 4

Predict

"Learning"

given

"I love Machine"
```

Every prediction uses the previous history.

This is exactly what the Chain Rule expresses mathematically.

---

# 14. Why is it Called Autoregressive?

The word

**Auto**

means

"self".

The word

**Regression**

means

"predicting using previous observations."

Therefore,

an Autoregressive Model predicts

future values

using

its own previously generated values.

In other words,

the model continuously feeds its own output back into itself.

---

# Key Insights

- Text is sequential data.
- The order of tokens is essential.
- One training example is an entire sequence.
- Every token is converted into a numerical vector.
- Language modeling still estimates a probability distribution.
- Autoregressive models generate one token at a time.
- The Chain Rule of Probability is the mathematical foundation of autoregressive language models.

---

# What's Next?

Now that we understand what an autoregressive model is,

the next question naturally arises:

> **How have researchers historically modeled these conditional probabilities?**

We will begin with:

- Linear Predictive Coding (LPC)
- N-gram Models
- Neural Language Models
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory Networks (LSTMs)

before finally understanding why the Transformer architecture replaced all of them.
