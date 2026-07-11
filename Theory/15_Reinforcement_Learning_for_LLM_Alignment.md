# 15. Reinforcement Learning for LLM Alignment

> **Core Theme:** Understand why next-token prediction alone is not enough to create a useful language assistant, and learn how supervised fine-tuning and reinforcement learning align a pretrained language model with human preferences.

---

# 1. Where We Are in the Story

In the previous chapter, we studied autoregressive language models and Transformers.

A language model represents the conditional probability:

`Pθ(Xₜ | X₁, X₂, ..., Xₜ₋₁)`

The model receives the previous tokens and predicts a probability distribution over the next token.

Its training objective is usually:

`Loss = −Σₜ log Pθ(Xₜ | X₁, ..., Xₜ₋₁)`

This is simply maximum-likelihood training through next-token prediction.

After training on a very large text collection, the model learns:

- grammar,
- factual patterns,
- writing styles,
- programming patterns,
- question-and-answer formats,
- relationships between concepts.

However, next-token prediction creates a model that imitates the training data.

It does not automatically create a model that behaves like a helpful assistant.

---

# 2. The Main Problem with Pretraining Alone

Imagine training a language model on a large portion of the internet.

The internet contains:

- high-quality educational content,
- useful discussions,
- misinformation,
- offensive language,
- contradictory opinions,
- unsafe instructions,
- incomplete answers,
- spam,
- low-quality writing.

The pretrained model learns patterns from all of these sources.

Conceptually:

```text
Large Internet Dataset

↓

Next-Token Prediction

↓

Model Learns Everything Present in the Data
```

The model does not automatically know:

- which responses are helpful,
- which responses are harmful,
- which writing style humans prefer,
- when it should refuse,
- when it should ask for clarification,
- how detailed an answer should be.

Prathosh describes the raw pretrained model as containing everything:

```text
Good

Bad

Useful

Unhelpful

Safe

Unsafe
```

Pretraining creates capability.

It does not necessarily create desirable behavior.

---

# 3. Pretraining Versus Alignment

It is helpful to separate two goals.

## Pretraining

The goal of pretraining is:

> Learn the statistical structure of language.

The model learns by predicting the next token.

```text
Large Text Corpus

↓

Next-Token Prediction

↓

General Language Model
```

---

## Alignment

The goal of alignment is:

> Make the model behave according to desired human rules and preferences.

Examples of desired behavior include:

- answering questions clearly,
- following instructions,
- avoiding harmful responses,
- admitting uncertainty,
- maintaining a useful conversation,
- preferring truthful answers over persuasive falsehoods.

Alignment is therefore a form of post-training.

---

# 4. What Is Post-Training?

**Post-training** refers to training performed after the large-scale next-token pretraining stage.

A simplified modern language-model pipeline is:

```text
Stage 1

Pretraining

↓

Stage 2

Supervised Fine-Tuning

↓

Stage 3

Preference or Reinforcement Learning

↓

Aligned Assistant
```

Each stage has a different purpose.

---

# 5. Stage 1 — Pretraining

During pretraining, the model receives ordinary text sequences.

Example:

```text
Input:

Machine learning is

Target:

a
```

The model predicts the next token.

This process is repeated over enormous datasets.

No human has to manually label every sequence.

The text itself provides the target token.

This is why pretraining is usually called:

**Self-Supervised Learning**

---

# 6. What Does Pretraining Teach?

Pretraining teaches general capabilities such as:

- completing sentences,
- generating paragraphs,
- translating patterns,
- recognizing programming syntax,
- recalling frequently represented information,
- imitating many writing styles.

However, the model is still fundamentally a next-token predictor.

It may continue text without correctly understanding the user's intention.

---

# 7. Example: Completion Versus Assistance

Suppose the prompt is:

```text
How can I improve my machine-learning model?
```

A raw next-token model may continue the text in a style resembling its training data.

It may produce:

- a useful tutorial,
- an irrelevant forum reply,
- an incomplete answer,
- contradictory recommendations.

A helpful assistant must do more than continue probable text.

It should infer:

- what the user is asking,
- the user's likely experience level,
- what information is missing,
- which answer would be most useful.

This requires post-training.

---

# 8. Stage 2 — Supervised Fine-Tuning

The first major post-training stage is:

**Supervised Fine-Tuning**, usually abbreviated as **SFT**.

In SFT, the model is trained on curated prompt-response pairs.

Example:

```text
Prompt:

Explain gradient descent to a beginner.

Desired Response:

Gradient descent is an optimization method that repeatedly adjusts...
```

The desired answer is provided by:

- human annotators,
- domain experts,
- carefully generated synthetic data,
- or a combination of these sources.

---

# 9. SFT Training Data

An SFT dataset may contain examples such as:

```text
User Question

↓

High-Quality Answer
```

```text
Instruction

↓

Correctly Formatted Response
```

```text
Programming Problem

↓

Working Code and Explanation
```

```text
Unsafe Request

↓

Appropriate Refusal
```

The model learns to imitate the desired responses.

---

# 10. How SFT Differs from Pretraining

During pretraining, the target is usually the naturally occurring next text.

During SFT, the target is a deliberately selected response.

## Pretraining

```text
Input Text

↓

Predict the Next Token Found in the Dataset
```

## Supervised Fine-Tuning

```text
User Instruction

↓

Predict a Curated Desired Response
```

Both stages may still use cross-entropy loss.

The major difference is the quality and purpose of the training data.

---

# 11. Why SFT Is Important

SFT teaches the model the expected interaction format.

It helps the model learn to:

- recognize instructions,
- answer instead of merely continue,
- produce structured responses,
- follow specific styles,
- use examples,
- refuse certain requests.

Without SFT, a pretrained model may possess knowledge but fail to behave like an assistant.

---

# 12. Is SFT Enough?

SFT improves the model considerably.

However, it has important limitations.

Suppose two answers are both reasonable:

```text
Answer A:

Very accurate but extremely technical.
```

```text
Answer B:

Slightly less detailed but much easier to understand.
```

Which one is better?

There may not be one objectively correct answer.

The preference depends on:

- the user,
- the context,
- the purpose,
- the desired level of detail.

SFT assumes that one target response is the correct response.

Human preference is often more subtle.

---

# 13. Human Preference Is Not a Simple Label

Traditional supervised learning often assumes a clear label.

Example:

```text
Image → Cat
```

But evaluating language-model responses is more subjective.

Consider two responses to:

```text
Explain neural networks.
```

Response A may be:

- mathematically rigorous,
- but difficult for a beginner.

Response B may be:

- intuitive,
- but less mathematically complete.

Different users may prefer different responses.

Therefore, language-model quality is difficult to represent using one fixed target.

---

# 14. Why Reinforcement Learning Is Introduced

Reinforcement Learning provides a framework for learning from feedback rather than only from fixed labels.

Instead of saying:

```text
This is the one correct response.
```

we may say:

```text
Response A is preferred over Response B.
```

or:

```text
This response received a high reward.
```

The model can then adjust its behavior to produce responses that humans are more likely to prefer.

---

# 15. Stage 3 — Reinforcement Learning from Human Feedback

The classical approach is called:

**Reinforcement Learning from Human Feedback**, abbreviated as **RLHF**.

A simplified RLHF pipeline is:

```text
Prompt

↓

Language Model Produces Responses

↓

Humans Evaluate or Rank Responses

↓

Learn from the Preferences

↓

Update the Language Model
```

The objective is not merely to predict likely text.

The objective becomes:

> Generate responses that receive higher human preference.

---

# 16. A Simple RLHF Example

Suppose the prompt is:

```text
Explain overfitting with an easy analogy.
```

The model generates two responses.

## Response A

```text
Overfitting occurs when empirical risk minimization produces a hypothesis
with low training error but high expected generalization error.
```

## Response B

```text
Overfitting is like memorizing the answers to a practice test instead of
understanding the subject. You perform well on familiar questions but poorly
on new ones.
```

For a beginner, a human annotator may prefer Response B.

The preference data becomes:

```text
Response B > Response A
```

The model is trained to make responses like B more likely.

---

# 17. What Does Alignment Mean?

In this context, **alignment** means modifying the model so that its behavior better matches a desired set of human goals or rules.

Examples include:

- helpfulness,
- harmlessness,
- honesty,
- instruction following,
- clarity,
- relevance,
- politeness.

Conceptually:

```text
General Language Capability

+

Human Preference Information

↓

Aligned Language Model
```

---

# 18. Alignment Is Not the Same as Knowledge

A model may know the answer but still respond poorly.

For example, it may:

- provide unnecessary detail,
- ignore the user's instruction,
- use an inappropriate tone,
- confidently invent information,
- produce unsafe content.

Alignment attempts to influence how the model uses its capabilities.

Therefore:

```text
Capability ≠ Alignment
```

Capability asks:

> Can the model perform the task?

Alignment asks:

> Does the model perform the task in the desired way?

---

# 19. Why Data Quality Matters

Prathosh emphasizes that the quality of post-training depends heavily on the quality of curated data.

Two organizations may begin with similar pretrained models.

Their final assistants may behave differently because they use different:

- instruction datasets,
- human annotators,
- evaluation rules,
- preference data,
- reward models,
- post-training procedures.

This means that data collection is a major part of alignment.

---

# 20. Preference Data

Preference data usually contains:

- a prompt,
- multiple candidate responses,
- a human ranking or selection.

Example:

```text
Prompt:

What is the difference between validation and test data?

Response A:

...

Response B:

...

Human Preference:

A is better than B
```

This is different from ordinary supervised data.

The annotator does not necessarily write the correct answer.

They compare available answers.

---

# 21. Why Pairwise Comparison Is Easier

Suppose a human is asked:

> Give this response a quality score between 0 and 10.

This can be difficult.

What exactly separates a score of `7` from a score of `8`?

Different annotators may use the scale differently.

It is often easier to ask:

> Which of these two responses is better?

Pairwise comparison reduces the cognitive burden.

This is why preference-learning systems frequently use ranked or paired responses.

---

# 22. Introduction to Reinforcement Learning

Before understanding RLHF, PPO, DPO, or GRPO, we need the basic Reinforcement Learning framework.

Reinforcement Learning is a machine-learning paradigm in which an entity learns through interaction.

The basic loop is:

```text
Agent

↓

Performs an Action

↓

Environment

↓

Produces a Reward and New Situation

↓

Agent Learns
```

---

# 23. Real-Life Intuition

Imagine teaching a dog to sit.

The dog performs an action.

If it sits correctly:

```text
Positive Reward

↓

Treat
```

If it ignores the instruction:

```text
No Reward
```

Over time, the dog learns which action produces the desirable outcome.

Reinforcement Learning follows the same basic idea.

---

# 24. Supervised Learning Versus Reinforcement Learning

## Supervised Learning

The model receives the correct answer directly.

```text
Input

↓

Correct Label

↓

Learn the Mapping
```

Example:

```text
Image → Cat
```

---

## Reinforcement Learning

The agent performs an action and receives feedback.

```text
State

↓

Action

↓

Reward

↓

Update Behavior
```

The correct action may not be provided explicitly.

The agent must discover which actions lead to larger rewards.

---

# 25. Unsupervised Learning Versus Reinforcement Learning

In unsupervised learning:

- no labels are provided,
- the model discovers structure in the data.

In reinforcement learning:

- the system interacts with an environment,
- actions influence future situations,
- reward provides feedback.

Therefore, Reinforcement Learning is not simply supervised learning without labels.

It has a different problem formulation.

---

# 26. The Agent

The **Agent** is the decision-making entity.

Examples:

| Problem | Agent |
|---|---|
| Chess | Chess-playing program |
| Self-driving car | Driving controller |
| Robot navigation | Robot |
| Video game | Game-playing algorithm |
| LLM alignment | Language model |

The agent observes a situation and chooses an action.

---

# 27. The Environment

The **Environment** is everything with which the agent interacts.

Examples:

| Problem | Environment |
|---|---|
| Chess | Chessboard and game rules |
| Self-driving car | Road, traffic, pedestrians |
| Robot navigation | Physical surroundings |
| Video game | Game world |
| LLM alignment | Prompt, conversation and evaluation process |

The environment determines the consequences of the agent's actions.

---

# 28. The Interaction Loop

A simplified Reinforcement Learning interaction is:

```text
Agent Observes Current Situation

↓

Agent Chooses an Action

↓

Environment Responds

↓

Agent Receives Reward

↓

A New Situation Begins
```

This process repeats over time.

---

# 29. The State

The **State** represents the information describing the current situation.

Examples:

## Chess

```text
Current Board Configuration
```

## Self-Driving Car

```text
Vehicle Position

Traffic Lights

Nearby Vehicles

Road Conditions
```

## Language Model

```text
Prompt

+

Tokens Generated So Far
```

The state should contain enough information for the agent to choose its next action.

---

# 30. The Action

The **Action** is the decision taken by the agent.

Examples:

## Chess

```text
Move a Piece
```

## Robot

```text
Move Left

Move Right

Stop
```

## Language Model

```text
Generate the Next Token
```

In an autoregressive language model, selecting the next token can be treated as an action.

---

# 31. The Reward

The **Reward** is feedback returned by the environment.

It is usually represented as a number.

Examples:

```text
+1  → desirable action

0   → neutral action

−1  → undesirable action
```

The reward tells the agent whether its behavior was useful.

It does not necessarily tell the agent exactly what it should have done instead.

---

# 32. Example: Learning to Play a Game

Suppose an agent is learning a video game.

Possible rewards might be:

```text
Collect Coin   → +1

Lose Life      → −5

Finish Level   → +20
```

The agent gradually learns a behavior that increases its total reward.

It is not explicitly told the perfect sequence of button presses.

It learns from consequences.

---

# 33. Reward in Language-Model Alignment

For language models, reward represents response quality.

A high reward may correspond to a response that is:

- helpful,
- accurate,
- safe,
- relevant.

A low reward may correspond to a response that is:

- incorrect,
- irrelevant,
- unsafe,
- poorly written.

The challenge is that response quality is difficult to measure directly.

This motivates:

- human preference data,
- reward models,
- preference-optimization algorithms.

---

# 34. The Policy

The most important object in Reinforcement Learning is the **Policy**.

A policy determines which action the agent is likely to take in a particular state.

It is written as:

`πθ(a | s)`

Read this as:

> The probability of taking action `a` when the agent is in state `s`.

Here:

- `s` is the state,
- `a` is the action,
- `θ` represents the policy parameters.

---

# 35. Policy Intuition

Imagine deciding what to do when it starts raining.

Your possible actions are:

- open an umbrella,
- run indoors,
- continue walking.

Your decision rule acts like a policy.

Conceptually:

```text
State:

It Is Raining

↓

Policy

↓

Action Probabilities:

Open Umbrella → 0.70

Run Indoors   → 0.25

Keep Walking  → 0.05
```

The policy does not always have to be deterministic.

It may define a probability distribution over actions.

---

# 36. The Language Model as a Policy

An autoregressive language model already produces a probability distribution over next tokens.

`Pθ(Xₜ | X₁, ..., Xₜ₋₁)`

This has the same mathematical form as a policy.

`πθ(a | s)`

The correspondence is:

```text
State s

=

Prompt + Tokens Generated So Far
```

```text
Action a

=

Next Token
```

```text
Policy πθ(a | s)

=

Language-Model Next-Token Distribution
```

This is the key connection between language models and Reinforcement Learning.

---

# 37. One-Step Example

Suppose the current state is:

```text
Machine learning is
```

The model's policy may be:

| Next-token action | Probability |
|---|---:|
| useful | 0.40 |
| difficult | 0.25 |
| changing | 0.15 |
| banana | 0.001 |

Selecting one token is the action.

After the token is appended, the state changes.

```text
Old State:

Machine learning is
```

```text
Action:

useful
```

```text
New State:

Machine learning is useful
```

The process repeats.

---

# 38. From One Token to a Complete Response

A complete response is generated through many state-action transitions.

```text
Prompt

↓

Choose Token 1

↓

Updated State

↓

Choose Token 2

↓

Updated State

↓

Choose Token 3

↓

...

↓

Complete Response
```

This sequence will later be called a **trajectory**.

---

# Key Insights

- Pretraining teaches language capability through next-token prediction.
- Pretraining alone does not guarantee helpful or safe behavior.
- Supervised Fine-Tuning teaches instruction-following using curated prompt-response pairs.
- Human preference is often easier to express through comparisons than numerical scores.
- RLHF uses human feedback to influence the model's behavior.
- Alignment concerns how model capability is used.
- Reinforcement Learning involves an agent interacting with an environment.
- The agent observes a state, chooses an action and receives a reward.
- A policy is a conditional distribution over actions given the state.
- An autoregressive language model can naturally be interpreted as an RL policy.
- In an LLM, the current prompt and generated tokens form the state, while the next token is the action.

---

# What's Next?

In Part 2, we will formalize the Reinforcement Learning framework using a:

**Markov Decision Process**

We will study:

- states,
- actions,
- transition probabilities,
- rewards,
- policies,
- trajectories,
- the Markov property,
- trajectory probability.

These concepts provide the mathematical foundation needed to understand policy optimization and RLHF.
