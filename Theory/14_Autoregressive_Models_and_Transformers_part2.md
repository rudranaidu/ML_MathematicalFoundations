# 15. The Evolution of Language Models

Now that we understand what an autoregressive model is,

the next question naturally arises:

> **How have researchers actually modeled these conditional probabilities over time?**

Recall the autoregressive equation.

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

Every language model ultimately tries to estimate these conditional probabilities.

The only thing that changes over the years is **how** these probabilities are modeled.

---

# 16. First Generation — Linear Predictive Coding (LPC)

The earliest autoregressive models came from **signal processing** in the 1950s and 1960s.

One of the most popular techniques was

**Linear Predictive Coding (LPC).**

Originally,

it was developed for compressing speech signals.

---

## The Main Idea

Suppose we have a speech signal.

Instead of transmitting every sample,

we predict the next sample using the previous samples.

Conceptually,

```text
Previous Samples

↓

Linear Combination

↓

Predict Next Sample
```

For example,

suppose we have

```text
2

4

6
```

A simple linear model might predict

```text
Next Value

=

0.2 × 2

+

0.3 × 4

+

0.5 × 6
```

The coefficients are learned from data.

---

## Why Was LPC Useful?

Instead of transmitting every speech sample,

only the coefficients needed to be transmitted.

This significantly reduced the amount of data sent over telephone networks.

---

## Limitation

LPC assumes that the relationship between past and future is **linear**.

Unfortunately,

natural language is highly nonlinear.

For example,

the next word depends on grammar,

context,

meaning,

and sometimes even sarcasm.

Linear models cannot capture these complexities.

---

# 17. Second Generation — N-gram Models

Researchers then introduced a statistical approach called the **N-gram Model**.

Instead of assuming a linear relationship,

they estimated probabilities by counting how frequently words appeared together.

---

## Example

Suppose our training corpus contains the sentence

```text
I love machine learning
```

many times.

Whenever the model sees

```text
I love
```

it learns that

```text
machine
```

is likely to follow.

The probability is estimated simply by counting occurrences.

---

## Bigram

A Bigram considers one previous word.

```text
Predict

Learning

given

Machine
```

---

## Trigram

A Trigram considers two previous words.

```text
Predict

Learning

given

Love Machine
```

---

## General N-gram

An N-gram considers the previous

N − 1

tokens.

Conceptually,

```text
Previous N−1 Words

↓

Count Occurrences

↓

Estimate Probability
```

---

## Why Did N-grams Work?

They were simple.

They required no neural networks.

Everything was based on counting frequencies.

For small vocabularies,

they worked surprisingly well.

---

## The Problem

Suppose we encounter

```text
I visited Mars yesterday.
```

If this sentence never appeared during training,

the N-gram model cannot estimate its probability.

This is called the

**Data Sparsity Problem.**

As the sequence length increases,

the number of possible combinations grows explosively.

Eventually,

counting becomes impractical.

---

# 18. Third Generation — Neural Language Models

Researchers then asked a very natural question.

> Instead of counting word occurrences,

why don't we use a neural network to learn these probabilities?

This idea was proposed by

**Yoshua Bengio** and colleagues,

and it became one of the earliest **Neural Language Models (NLMs).**

---

## The Basic Idea

Instead of counting,

feed the previous words into a neural network.

Conceptually,

```text
Previous Words

↓

Neural Network

↓

Predict Next Word
```

This was a revolutionary idea.

Instead of memorizing frequencies,

the network could learn complex nonlinear relationships.

---

# 19. A Simple Example

Suppose the input sentence is

```text
I

love

machine
```

The neural network receives

```text
"I"

"love"

"machine"
```

as input.

The output is a probability distribution over every possible next word.

Example:

| Word | Probability |
|------|------------:|
| learning | 0.82 |
| food | 0.03 |
| banana | 0.01 |
| train | 0.08 |
| ... | ... |

The network predicts

```text
learning
```

because it has the highest probability.

---

# 20. Why Neural Networks Were Better

Unlike N-gram models,

neural networks can generalize.

Even if they have never seen

```text
I enjoy deep learning
```

they may still predict

```text
algorithms
```

because they have learned meaningful representations of words.

This was a huge improvement over counting-based methods.

---

# 21. The Hidden Problem

Although Neural Language Models worked better,

they had one serious weakness.

Suppose the input sentence becomes very long.

```text
The student who attended the machine learning course last year finally completed the project ...
```

Should we feed the **entire sentence** into one neural network?

That quickly becomes impractical.

The number of input neurons keeps increasing.

---

# 22. Another Problem — Word Order

There is an even bigger issue.

Consider two sentences.

```text
Dog bites man.
```

and

```text
Man bites dog.
```

If we simply feed all words into a standard Multi-Layer Perceptron,

the network does not naturally understand sequence order.

A fully connected network treats every input independently.

It has no built-in notion of time.

---

# 23. Why Sequence Order Matters

Imagine reading this sentence.

```text
The exam was surprisingly easy.
```

Now rearrange the words.

```text
Easy surprisingly was exam the.
```

The same words are present.

Yet the sentence becomes meaningless.

Language is fundamentally sequential.

A model that ignores order cannot truly understand language.

---

# 24. Researchers Needed a Better Architecture

At this point,

researchers realized that simply replacing counting with neural networks was not enough.

They needed an architecture that naturally understood sequences.

The desired architecture should:

- Process one word at a time.
- Remember previous words.
- Handle sequences of different lengths.
- Preserve temporal order.

This motivation led to the invention of

**Recurrent Neural Networks (RNNs).**

---

# 25. The Key Idea Behind RNNs

Instead of feeding the entire sentence into the network,

process one token at a time.

Conceptually,

```text
Word 1

↓

Word 2

↓

Word 3

↓

...

↓

Word T
```

At every step,

the network remembers what it has already seen.

This memory is stored in something called the

**Hidden State.**

---

# Key Insights

- Every language model estimates the same conditional probabilities.
- LPC models assume linear relationships.
- N-gram models estimate probabilities by counting word occurrences.
- Neural Language Models replace counting with neural networks.
- Fully connected neural networks still struggle with long sequences.
- Language requires models that preserve sequence order.
- These limitations motivated the invention of Recurrent Neural Networks.

---

# What's Next?

Now that we understand why Neural Language Models were insufficient,

we are ready to study

**Recurrent Neural Networks (RNNs)**

and see:

- how they remember previous tokens,
- why they became the dominant architecture for sequence modeling,
- and why they were eventually replaced by Transformers.
