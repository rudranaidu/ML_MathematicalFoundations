# 132. From Human Preferences to LLM Alignment

In the previous parts, we learned that a language model can be viewed as a Reinforcement Learning policy.

`πθ(a | s) = Pθ(next token | prompt and tokens generated so far)`

A complete generated response is a trajectory.

The remaining question is:

> How do we obtain a useful reward for a language-model response?

For a game, defining reward may be easy.

```text
Win Game  → +1

Lose Game → −1
```

For language, response quality is much harder to measure.

A response may need to be:

- correct,
- relevant,
- clear,
- safe,
- concise,
- complete,
- aligned with the user's instruction.

There is usually no simple mathematical formula that perfectly measures all these properties.

This is why human preferences become important.

---

# 133. Absolute Scores Are Difficult

Suppose a human evaluator reads an answer and is asked:

```text
Rate this response from 0 to 10.
```

This may be difficult.

What exactly is the difference between:

```text
Score 7
```

and:

```text
Score 8
```

Different evaluators may interpret the scale differently.

One evaluator may consider `7` a strong score.

Another may consider `7` average.

Therefore, absolute scores can be inconsistent.

---

# 134. Pairwise Preference Is Easier

Instead of asking for a numerical score, we can show two answers.

```text
Prompt

↓

Response A

Response B

↓

Which Response Is Better?
```

The human evaluator chooses one.

For example:

```text
Response B is preferred over Response A.
```

This produces a pairwise preference.

It is usually easier to compare two answers than to invent an exact numerical reward.

---

# 135. Chosen and Rejected Responses

Preference datasets commonly contain:

- a prompt,
- a chosen response,
- a rejected response.

We may represent them as:

```text
x

=

Prompt
```

```text
y_w

=

Preferred or Winning Response
```

```text
y_l

=

Less Preferred or Losing Response
```

The preference is written as:

`y_w ≻ y_l`

Read this as:

> Response `y_w` is preferred over response `y_l`.

---

# 136. Simple Preference Example

Consider the prompt:

```text
Explain overfitting to a beginner.
```

## Response A

```text
Overfitting occurs when the empirical hypothesis minimizes training loss
without minimizing expected population risk.
```

## Response B

```text
Overfitting is like memorizing the answers to a practice exam instead of
understanding the subject. You perform well on familiar questions but poorly
on new questions.
```

For a beginner, an evaluator may choose Response B.

The preference dataset records:

`Response B ≻ Response A`

The training system then tries to make responses like B more likely.

---

# 137. The Reward Model

In traditional RLHF, pairwise preferences are used to train a separate model called a:

**Reward Model**

The reward model receives:

- the prompt,
- a candidate response.

It outputs one scalar score.

`rφ(x,y) = reward score for response y given prompt x`

Here:

- `φ` represents reward-model parameters,
- `x` is the prompt,
- `y` is the response.

A larger score means the reward model considers the response more desirable.

---

# 138. Reward Model Intuition

Think of the reward model as a learned judge.

```text
Prompt + Response

↓

Reward Model

↓

Quality Score
```

For example:

```text
Helpful and correct answer

↓

Reward = 8.7
```

```text
Incorrect or irrelevant answer

↓

Reward = 2.1
```

The reward model attempts to reproduce human preferences.

---

# 139. How Is the Reward Model Trained?

Suppose a human prefers `y_w` over `y_l`.

The reward model should satisfy:

`rφ(x,y_w) > rφ(x,y_l)`

It should assign a larger score to the preferred response.

The important information is often the difference:

`rφ(x,y_w) − rφ(x,y_l)`

A large positive difference means the model strongly prefers the winning response.

---

# 140. Converting Reward Differences into Preference Probability

A sigmoid function can convert the score difference into a probability.

`Pφ(y_w ≻ y_l | x) = σ(rφ(x,y_w) − rφ(x,y_l))`

where:

`σ(z) = 1 / (1 + exp(−z))`

If the winning response receives a much larger reward, the probability approaches `1`.

If both responses receive similar scores, the probability approaches `0.5`.

---

# 141. Reward-Model Loss

The reward model is trained to assign high probability to the human preference.

A common loss is:

`L_RM(φ) = −log σ(rφ(x,y_w) − rφ(x,y_l))`

Minimizing this loss encourages:

`rφ(x,y_w) > rφ(x,y_l)`

The reward model therefore learns a numerical approximation of human preference.

---

# 142. Reward Model Is Not Perfect Human Judgment

The reward model does not truly become a human.

It only learns patterns from the available preference data.

Its behavior depends on:

- who supplied the preferences,
- how instructions were written,
- which responses were compared,
- how much data was collected,
- what evaluation rules were used.

Therefore, the reward model is only an approximation of the desired behavior.

---

# 143. The Classical RLHF Pipeline

A simplified RLHF pipeline contains several stages.

```text
Stage 1

Large-Scale Pretraining

↓

Stage 2

Supervised Fine-Tuning

↓

Stage 3

Generate Multiple Responses

↓

Stage 4

Collect Human Preferences

↓

Stage 5

Train Reward Model

↓

Stage 6

Optimize Language Model Using RL

↓

Aligned Language Model
```

Each stage solves a different problem.

---

# 144. Stage 1 — Pretrained Model

The pretrained model learns general language patterns through next-token prediction.

```text
Internet-Scale Text

↓

Autoregressive Training

↓

Pretrained Language Model
```

It gains broad capabilities but is not yet necessarily a useful assistant.

---

# 145. Stage 2 — Supervised Fine-Tuned Model

The model is then trained on curated instruction-response examples.

```text
Prompt

↓

Desired Answer
```

This teaches the model:

- instruction following,
- response formatting,
- conversational behavior,
- domain-specific tasks.

The resulting model is commonly used as the starting policy for preference optimization.

---

# 146. Stage 3 — Collecting Preference Data

The model generates multiple answers for the same prompt.

```text
Prompt

↓

Response A

Response B
```

Human evaluators compare the responses.

The resulting dataset contains examples such as:

`(x, y_w, y_l)`

This becomes the training data for preference learning.

---

# 147. Stage 4 — Training the Reward Model

The reward model learns to score preferred responses higher.

```text
Prompt + Response

↓

Reward Model

↓

Scalar Reward
```

Once trained, it can evaluate new model-generated responses without asking a human to score every response manually.

---

# 148. Stage 5 — Reinforcement Learning

The language model generates a response.

The reward model evaluates it.

```text
Prompt

↓

Language Model Generates Response

↓

Reward Model Assigns Reward

↓

Update Language-Model Policy
```

The policy is adjusted so that responses receiving larger rewards become more likely.

---

# 149. Why a Reference Model Is Needed

Suppose we optimize only the reward-model score.

The language model may change too aggressively.

It may:

- forget useful language abilities,
- exploit weaknesses in the reward model,
- produce unnatural text,
- lose response diversity,
- discover strange patterns that receive high reward.

To reduce this problem, alignment commonly keeps a frozen copy of the original fine-tuned model.

This is called the:

**Reference Model**

Its policy is written as:

`π_ref(y | x)`

---

# 150. Staying Close to the Reference Model

The optimized model should receive high reward while remaining reasonably close to the reference model.

A simplified objective is:

`J(θ) = E[rφ(x,y)] − β KL(πθ(·|x) || π_ref(·|x))`

The first term encourages high-reward responses.

The second term penalizes the model for moving too far from the reference policy.

The coefficient `β` controls the strength of this penalty.

---

# 151. Intuition for the KL Penalty

Imagine improving a recipe.

You want to make it better, but you do not want to replace every ingredient at once.

```text
Reward

↓

Improve the Recipe
```

```text
KL Penalty

↓

Do Not Change It Too Aggressively
```

The reward pushes the model toward preferred behavior.

The KL penalty acts like an anchor.

---

# 152. What Happens When Beta Is Large?

If `β` is large:

- the model stays close to the reference model,
- updates are conservative,
- behavior changes slowly.

Conceptually:

```text
Large β

↓

Strong Anchor
```

---

# 153. What Happens When Beta Is Small?

If `β` is small:

- the reward has more influence,
- the policy may change more aggressively,
- reward exploitation becomes more likely.

Conceptually:

```text
Small β

↓

Weak Anchor
```

Choosing the balance is an important engineering decision.

---

# 154. Reward Hacking

A policy may discover a way to receive high reward without actually producing the behavior humans intended.

This is called:

**Reward Hacking**

For example, suppose a reward model incorrectly associates longer answers with higher quality.

The language model may begin producing unnecessarily long responses.

It has optimized the measured reward but not the true human goal.

---

# 155. Easy Reward-Hacking Analogy

Suppose a school rewards students only by the number of pages written.

Students may produce very long answers containing repeated or irrelevant material.

They maximize the score.

But they do not necessarily improve the quality of their answers.

Similarly:

```text
Measured Reward

≠

True Desired Behavior
```

This is one of the central difficulties of alignment.

---

# 156. Proximal Policy Optimization

One popular method used in classical RLHF is:

**Proximal Policy Optimization**

abbreviated as:

**PPO**

The word **proximal** means:

> Keep the updated policy close to the previous policy.

PPO attempts to improve reward without allowing one optimization step to change the policy too dramatically.

---

# 157. Why Large Policy Updates Are Dangerous

Suppose the model currently gives a useful response with high probability.

A very large update may accidentally destroy that behavior.

```text
Useful Existing Policy

↓

Extremely Large Update

↓

Unstable or Degraded Policy
```

PPO introduces a constraint-like mechanism that limits excessive change.

---

# 158. Old Policy and New Policy

During one PPO update, we compare:

- the new policy `πθ`,
- the policy used to generate the data `π_old`.

The probability ratio is:

`rₜ(θ) = πθ(Aₜ | Sₜ) / π_old(Aₜ | Sₜ)`

This ratio measures how much the probability of an action has changed.

---

# 159. Understanding the Probability Ratio

If:

`rₜ(θ) = 1`

then the action probability has not changed.

If:

`rₜ(θ) = 1.5`

then the new policy makes that action `1.5` times as likely.

If:

`rₜ(θ) = 0.5`

then the new policy makes that action half as likely.

PPO tries to prevent this ratio from becoming too extreme.

---

# 160. Advantage

PPO commonly uses a quantity called the:

**Advantage**

written as:

`Aₜ`

The advantage indicates whether an action was better or worse than expected in a particular state.

Conceptually:

```text
Positive Advantage

↓

Action Was Better Than Expected
```

```text
Negative Advantage

↓

Action Was Worse Than Expected
```

The policy should increase the probability of positive-advantage actions and reduce the probability of negative-advantage actions.

---

# 161. PPO Clipping

PPO clips the probability ratio to a limited interval.

`clip(rₜ(θ), 1−ε, 1+ε)`

For example, if:

`ε = 0.2`

the preferred range is approximately:

```text
0.8 to 1.2
```

This prevents excessively large policy changes.

---

# 162. PPO Objective

A common clipped PPO objective is:

`L_CLIP(θ) = E[min(rₜ(θ)Aₜ, clip(rₜ(θ),1−ε,1+ε)Aₜ)]`

The important intuition is:

- improve actions with positive advantage,
- reduce actions with negative advantage,
- do not move too far in one update.

The clipping operation stabilizes policy optimization.

---

# 163. PPO Intuition Using a Steering Wheel

Imagine learning to drive.

If the car moves slightly to the left, you make a small steering correction.

You do not immediately rotate the steering wheel completely to the right.

```text
Small Error

↓

Small Controlled Correction
```

PPO follows the same philosophy.

It improves the policy using controlled updates rather than extreme changes.

---

# 164. Why PPO Is Complicated for LLMs

Using PPO for language models may require several components:

- the trainable policy model,
- a reference model,
- a reward model,
- often a value or critic model,
- generated response rollouts,
- advantage estimation,
- KL control.

This makes the training pipeline computationally and operationally complex.

This complexity motivated simpler preference-optimization methods.

---

# 165. Direct Preference Optimization

A method called:

**Direct Preference Optimization**

or:

**DPO**

uses preference pairs directly.

It avoids the classical pipeline of:

```text
Train Reward Model

↓

Run Reinforcement Learning Against Reward Model
```

Instead, it directly updates the language model using:

- the preferred response,
- the rejected response,
- a reference model.

---

# 166. DPO Intuition

Suppose the human says:

`Response A ≻ Response B`

DPO directly teaches the model:

```text
Increase the relative probability of Response A

and

Decrease the relative probability of Response B
```

The model learns from the comparison itself.

A separate explicit reward-model optimization loop is not required.

---

# 167. Why a Reference Model Still Appears in DPO

Without a reference model, the trainable policy could increase probabilities in uncontrolled ways.

DPO compares how much the policy prefers each response relative to the reference model.

The reference model acts as an anchor, similar to the KL regularization used in RLHF.

---

# 168. DPO Preference Signal

For a preferred response `y_w` and rejected response `y_l`, DPO compares:

`log πθ(y_w|x) − log πθ(y_l|x)`

It also compares the same difference under the reference model:

`log π_ref(y_w|x) − log π_ref(y_l|x)`

The trainable model should prefer the winning response more strongly than the reference model does.

---

# 169. DPO Loss

A commonly written DPO loss is:

`L_DPO(θ) = −E[log σ(β[(log πθ(y_w|x)−log π_ref(y_w|x))−(log πθ(y_l|x)−log π_ref(y_l|x))])]`

Although the equation looks long, its intention is simple:

> Make the chosen response relatively more likely than the rejected response while remaining anchored to the reference model.

---

# 170. DPO Step-by-Step Intuition

For every preference example:

```text
1. Read the prompt.

2. Calculate the probability of the chosen response.

3. Calculate the probability of the rejected response.

4. Compare both probabilities with the reference model.

5. Increase the chosen response's relative preference.

6. Decrease the rejected response's relative preference.
```

No online interaction with a reward model is required during this optimization step.

---

# 171. PPO Versus DPO

## PPO-Style RLHF

```text
Human Preferences

↓

Train Reward Model

↓

Generate New Responses

↓

Reward the Responses

↓

Run Policy Optimization
```

## DPO

```text
Human Preferences

↓

Directly Optimize Chosen Versus Rejected Responses
```

DPO simplifies the preference-training pipeline.

---

# 172. Is DPO Ordinary Supervised Fine-Tuning?

Not exactly.

SFT says:

```text
Increase the probability of this desired answer.
```

DPO says:

```text
Prefer this answer over another answer, relative to a reference model.
```

DPO learns from comparisons.

SFT learns from target demonstrations.

---

# 173. Group Relative Policy Optimization

The lecture also previews:

**Group Relative Policy Optimization**

abbreviated as:

**GRPO**

The main idea is to generate a group of responses for the same prompt and compare their rewards within that group.

```text
One Prompt

↓

Response 1

Response 2

Response 3

...

Response G

↓

Score All Responses

↓

Compare Them Relative to the Group
```

---

# 174. Why Use a Group?

Suppose all responses to one prompt receive the following rewards:

```text
Response 1 → 3

Response 2 → 8

Response 3 → 6

Response 4 → 2
```

Instead of asking whether a reward of `6` is universally good, we can compare it with the other responses in the same group.

Response 2 is best within the group.

Response 4 is worst within the group.

This provides a relative learning signal.

---

# 175. Group-Relative Advantage

A simplified group-relative advantage can be computed as:

`Aᵢ = (rᵢ − mean(r₁,...,rG)) / (std(r₁,...,rG) + ε)`

where:

- `rᵢ` is the reward of response `i`,
- `mean(...)` is the group-average reward,
- `std(...)` is the group standard deviation,
- `ε` prevents division by zero.

Responses scoring above the group average receive positive advantages.

Responses below the group average receive negative advantages.

---

# 176. GRPO Intuition

Imagine interviewing five candidates.

Instead of assigning an absolute universal intelligence score, you compare their performances within the interview group.

```text
Better Than Group Average

↓

Positive Signal
```

```text
Worse Than Group Average

↓

Negative Signal
```

GRPO uses a similar relative comparison.

---

# 177. Why Avoid a Separate Value Model?

Traditional policy-gradient systems may train a value model or critic to estimate expected future return.

GRPO-style approaches can use group-relative rewards to estimate the learning signal without relying on a separate value model in the same way.

This can reduce memory and computational requirements.

The precise algorithmic details depend on the implementation.

---

# 178. High-Level GRPO Workflow

```text
1. Sample a prompt.

2. Generate a group of responses.

3. Score each response.

4. Compute group-relative advantages.

5. Increase probabilities of above-average responses.

6. Reduce probabilities of below-average responses.

7. Limit policy movement using clipping or reference regularization.
```

The overall goal remains:

> Improve high-quality behavior without allowing unstable policy updates.

---

# 179. Comparing SFT, PPO, DPO and GRPO

| Method | Main training signal | Separate reward model | Online response generation during optimization | Main intuition |
|---|---|---:|---:|---|
| SFT | Desired target response | No | No | Imitate the demonstration |
| PPO-style RLHF | Scalar reward and advantage | Usually yes | Yes | Maximize reward with controlled updates |
| DPO | Preferred versus rejected responses | No explicit reward model | No | Directly prefer the chosen answer |
| GRPO-style training | Relative reward within response groups | Depends on setup | Yes | Compare sampled responses within a group |

---

# 180. When Is SFT Useful?

SFT is useful when high-quality demonstrations are available.

It teaches:

- desired answer structure,
- domain-specific knowledge usage,
- instruction-following patterns,
- style and formatting.

SFT gives the model a strong starting point before preference optimization.

---

# 181. When Is Preference Optimization Useful?

Preference optimization is useful when:

- several answers may be acceptable,
- ranking is easier than writing a perfect target,
- response quality is subjective,
- we want to improve behavior beyond imitation.

It helps the model learn which outputs are preferred, not merely which outputs are possible.

---

# 182. Alignment Does Not Guarantee Truth

A response can be preferred by humans and still be factually wrong.

Human evaluators may:

- miss an error,
- reward confidence,
- prefer fluent writing,
- disagree with one another.

Therefore:

```text
Human Preference

≠

Guaranteed Truth
```

Alignment and factual correctness are related but different goals.

---

# 183. Alignment Does Not Mean One Universal Preference

Different users may prefer different:

- levels of detail,
- tones,
- formats,
- cultural conventions,
- risk tolerances.

Therefore, alignment is always relative to some chosen set of objectives, policies or annotator preferences.

There is no single preference function that perfectly represents every human being.

---

# 184. Post-Training Depends Strongly on Data

Prathosh emphasizes that high-quality post-training data can strongly affect final model behavior.

The final system depends on:

- prompt diversity,
- annotator instructions,
- chosen and rejected responses,
- domain coverage,
- reward design,
- evaluation procedures.

Two models with similar architectures can behave differently because their post-training data and objectives differ.

---

# 185. Common Failure Modes

## Reward Hacking

The model exploits imperfections in the reward function.

---

## Over-Optimization

The policy moves too far from the original model and loses useful behavior.

---

## Preference Bias

The model learns biases present in the annotator population or dataset.

---

## Reduced Diversity

The model repeatedly produces safe but generic responses.

---

## Length Bias

The model may learn that longer or shorter answers receive better scores even when length is unrelated to quality.

---

## Evaluation Mismatch

The training reward may not accurately represent real user satisfaction.

---

# 186. Common Misconceptions

## Misconception 1

**RLHF gives the model human intelligence.**

It does not.

It modifies the model's behavior using human feedback.

---

## Misconception 2

**The reward model knows the objective truth.**

It does not.

It approximates preferences found in its training data.

---

## Misconception 3

**PPO simply increases the probability of every high-reward response without restriction.**

It does not.

PPO limits large policy changes using probability ratios and clipping.

---

## Misconception 4

**DPO requires a separately trained reward model.**

It does not require an explicit reward-model training stage in the classical DPO pipeline.

It learns directly from preference pairs.

---

## Misconception 5

**DPO and SFT are identical.**

They are not.

SFT imitates target responses.

DPO learns relative preference between chosen and rejected responses.

---

## Misconception 6

**GRPO gives every response an independent absolute judgment.**

Its key idea is to use relative comparisons among a group of sampled responses.

---

## Misconception 7

**Alignment completely solves hallucination and safety.**

It does not.

Alignment can improve behavior, but model errors and failures may still occur.

---

# 187. Connecting the Entire Post-Training Story

The complete progression is:

```text
Pretraining

↓

Learn General Language Patterns

↓

Supervised Fine-Tuning

↓

Learn Instruction Following

↓

Preference Data

↓

Learn Which Responses Humans Prefer

↓

PPO, DPO or Related Optimization

↓

Improve the Model's Behavior
```

Each stage builds upon the previous one.

---

# 188. Full LLM Alignment Mapping

| Concept | LLM interpretation |
|---|---|
| Agent | Language model |
| State | Prompt and generated prefix |
| Action | Next token |
| Policy | Next-token probability distribution |
| Trajectory | Complete generated response |
| Reward | Estimated response quality |
| Human feedback | Preference between responses |
| Reward model | Learned approximation of human preferences |
| Reference policy | Frozen model used as an anchor |
| PPO | Reward maximization with controlled policy updates |
| DPO | Direct learning from chosen and rejected responses |
| GRPO | Relative optimization over groups of sampled responses |

---

# Chapter Summary

In this chapter, we studied how language models are aligned using supervised learning and Reinforcement Learning ideas.

We learned that:

- pretraining creates general language capability,
- next-token prediction alone does not guarantee useful assistant behavior,
- SFT teaches the model using curated prompt-response pairs,
- human preferences can be collected through pairwise comparisons,
- a reward model converts preferences into scalar scores,
- an LLM can be interpreted as a policy over next-token actions,
- complete responses can be viewed as trajectories,
- expected return provides the Reinforcement Learning objective,
- policy gradients improve actions associated with high return,
- a reference model prevents the policy from changing too aggressively,
- PPO uses clipped policy updates,
- DPO learns directly from preference pairs,
- GRPO compares rewards among groups of sampled responses,
- alignment quality depends strongly on data and evaluation design.

---

# Key Takeaways

✅ Pretraining teaches a model how language behaves.

✅ SFT teaches the model how to respond to instructions.

✅ Preference data teaches the model which responses are more desirable.

✅ A reward model approximates human judgment using a scalar score.

✅ PPO improves reward while limiting excessive policy changes.

✅ DPO directly increases the relative probability of preferred responses.

✅ GRPO uses relative performance within groups of generated responses.

✅ Human preference is not identical to factual truth.

✅ Alignment is an ongoing optimization problem, not a complete guarantee of perfect behavior.

---

# Important Scope Note

This lecture introduces the foundations needed for:

- Policy Gradient methods,
- PPO,
- DPO,
- GRPO,
- RLHF.

However, the detailed mathematical derivations of these algorithms were left for the following class.

Therefore, this part provides the conceptual and mathematical bridge needed to understand those future derivations without pretending that the full algorithms were completed in the current lecture.

---

# Looking Ahead

The next theory lecture can continue with:

```text
Policy Gradient Theorem

↓

Value Functions and Advantages

↓

PPO Derivation

↓

Reward Modeling

↓

DPO Derivation

↓

GRPO Derivation
```

These topics explain how modern preference-optimization objectives are obtained from the general Reinforcement Learning framework.
