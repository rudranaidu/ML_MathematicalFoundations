# 26. Recurrent Neural Networks (RNNs)

Researchers realized that language is fundamentally sequential.

Instead of giving the entire sentence to a neural network at once,

why not process one word at a time?

This simple idea led to the invention of the

**Recurrent Neural Network (RNN).**

---

# 27. The Main Idea Behind an RNN

Imagine reading a novel.

You do not read the entire book in one glance.

Instead,

you read one word,

then the next,

then the next.

While reading,

your brain remembers what you have already read.

RNNs work in exactly the same way.

They process one token at a time while carrying forward a memory of the past.

---

# 28. The Hidden State

This memory is called the **Hidden State**.

Conceptually,

```text
Word 1

↓

Hidden State H₁

↓

Word 2

↓

Hidden State H₂

↓

Word 3

↓

Hidden State H₃

↓

...

↓

Word T

↓

Hidden State HT
```

The hidden state acts like the network's memory.

It summarizes everything the network has seen so far.

---

# 29. Real-Life Analogy

Imagine someone tells you a story.

Sentence 1

```text
John bought a dog.
```

Sentence 2

```text
He loves playing with it.
```

When you hear

```text
He
```

you immediately know it refers to

```text
John
```

How?

Because your brain remembers the previous sentence.

The hidden state plays exactly the same role.

It stores useful information from earlier words.

---

# 30. How Does an RNN Work?

At every time step,

the network receives two inputs.

```text
Current Word

+

Previous Hidden State
```

These are combined to produce:

```text
New Hidden State
```

Conceptually,

```text
Current Word

+

Previous Memory

↓

Neural Network

↓

Updated Memory
```

This process repeats for every token in the sequence.

---

# 31. Why is it Called "Recurrent"?

Notice something interesting.

The same neural network is reused again and again.

```text
Word 1

↓

Same Network

↓

Word 2

↓

Same Network

↓

Word 3

↓

Same Network

↓

...
```

The network is repeatedly applied across time.

This repeated application is called

**Recurrence**.

Hence the name:

**Recurrent Neural Network.**

---

# 32. Parameter Sharing

One of the biggest advantages of an RNN is that

the same parameters are reused at every time step.

Conceptually,

```text
θ

↓

Word 1

↓

θ

↓

Word 2

↓

θ

↓

Word 3

↓

θ

↓

...
```

There is only **one neural network**.

It is reused for every token.

---

# 33. Why Share Parameters?

Suppose every word had its own neural network.

A sentence of

100 words

would require

100 different networks.

That would be impossible to train.

Instead,

the same parameters are shared across time.

This allows the RNN to process sequences of any length.

Whether the sentence has:

- 10 words
- 100 words
- 1000 words

the same network is reused.

---

# 34. The RNN Computation

At every time step,

the hidden state is updated using:

- the current input token,
- the previous hidden state.

Conceptually,

```text
HT

=

Neural Network

(

XT ,

HT−1

)
```

where

- XT is the current token.
- HT−1 is the previous memory.
- HT is the updated memory.

Notice something important.

The hidden state depends on **everything that came before it**.

---

# 35. Why RNNs Were Revolutionary

For the first time,

neural networks could naturally process sequences.

Instead of treating every word independently,

the model now had memory.

This made RNNs successful for tasks such as:

- Language Modeling
- Speech Recognition
- Machine Translation
- Handwriting Recognition

For many years,

RNNs became the standard architecture for sequential data.

---

# 36. The Problem with Long Sequences

Unfortunately,

RNNs introduced a new problem.

Imagine reading this sentence.

```text
The student who attended the machine learning course in 2022 and later worked on several research projects finally published a paper...
```

To understand the word

```text
published
```

the model may need information from the beginning of the sentence.

However,

that information must travel through dozens of hidden states.

```text
H₁

↓

H₂

↓

H₃

↓

...

↓

H₅₀
```

Every step slightly weakens the information.

Eventually,

the model begins to forget.

---

# 37. Real-Life Analogy

Imagine playing the children's game:

**Chinese Whispers (Telephone Game).**

One person whispers a sentence to the next.

The second person whispers it to the third.

The third whispers it to the fourth.

Eventually,

the final sentence becomes completely different.

Information gradually degrades.

RNNs suffer from exactly the same problem.

As information passes through many hidden states,

important details are gradually lost.

---

# 38. The Vanishing Gradient Problem

Prathosh explains that this issue can also be understood mathematically.

During training,

the network learns using **Backpropagation**.

The gradients must travel backward through every time step.

```text
H₅₀

↑

H₄₉

↑

H₄₈

↑

...

↑

H₁
```

At every step,

the gradients are multiplied by matrices.

After many multiplications,

the gradients become extremely small.

Eventually,

the learning signal almost disappears.

This phenomenon is called the

**Vanishing Gradient Problem.**

---

# 39. Why Does This Matter?

Suppose the model needs to learn that

```text
John

...

he
```

refer to the same person.

If these words are separated by many tokens,

the gradient becomes too weak.

The network fails to learn this long-range dependency.

As a result,

RNNs struggle with long documents,

long conversations,

and long paragraphs.

---

# 40. Long Short-Term Memory (LSTM)

Researchers proposed a solution called the

**Long Short-Term Memory Network (LSTM).**

Instead of using only one hidden state,

LSTMs introduce additional memory mechanisms.

Conceptually,

```text
Current Word

↓

Memory Gates

↓

Updated Memory
```

The network learns:

- what to remember,
- what to forget,
- what to update.

This greatly improves long-term memory.

---

# 41. Why LSTMs Help

Imagine carrying a notebook while reading a book.

Whenever something important happens,

you write it down.

Whenever information becomes irrelevant,

you erase it.

LSTMs behave similarly.

They maintain an internal memory and learn when to keep or discard information.

---

# 42. Did LSTMs Solve Everything?

LSTMs significantly improved sequence modeling.

However,

they still had one major limitation.

They processed words **sequentially**.

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

Notice that

Word 100

cannot be processed until

Words 1–99

have already been processed.

This makes computation inherently sequential.

As sequence lengths grow,

training becomes slower.

---

# 43. Another Limitation

Suppose the first word and the last word are closely related.

The information still has to travel through every intermediate hidden state.

Although LSTMs improve memory,

they still struggle with extremely long contexts.

This became a major obstacle as researchers began training models on:

- books,
- websites,
- Wikipedia,
- entire internet corpora.

---

# 44. The Encoder–Decoder Architecture

To solve problems such as machine translation,

researchers introduced the

**Encoder–Decoder Architecture.**

Conceptually,

```text
English Sentence

↓

Encoder

↓

Hidden Representation

↓

Decoder

↓

French Sentence
```

The encoder converts the input sentence into a hidden representation.

The decoder generates the output sentence one token at a time.

---

# 45. The Hidden Bottleneck

This architecture worked well for short sentences.

However,

there was a serious limitation.

The entire meaning of the input sentence had to be compressed into **one hidden vector**.

```text
Long Sentence

↓

Encoder

↓

One Hidden Vector

↓

Decoder
```

Imagine trying to summarize an entire novel into one sentence.

Important details would inevitably be lost.

This bottleneck became a major weakness.

---

# 46. Researchers Asked a New Question

Instead of forcing the decoder to depend on only one hidden vector,

why not allow it to look back at **every hidden state** generated by the encoder?

This simple question led to one of the most influential ideas in modern AI.

That idea is called

**Attention.**

---

# Key Insights

- RNNs process one token at a time.
- The hidden state stores information about previous tokens.
- The same network parameters are reused across all time steps.
- Parameter sharing allows RNNs to process sequences of arbitrary length.
- Long sequences lead to the Vanishing Gradient Problem.
- LSTMs improve memory using learnable gates.
- LSTMs still process sequences sequentially.
- Encoder–Decoder architectures compress the entire input into one hidden vector.
- This bottleneck motivated the invention of the Attention Mechanism.

---

# What's Next?

Now that we understand **why RNNs and LSTMs eventually became insufficient**,

we are ready to study one of the most important ideas in modern Artificial Intelligence:

**The Attention Mechanism.**

This single idea eventually led to:

- Transformers
- BERT
- GPT
- ChatGPT
- Gemini
- Claude
- Llama
- DeepSeek

Nearly every modern Large Language Model is built upon this concept.
