# 17. Value, Q and Advantage Functions for LLM Alignment

This chapter develops the variance-reduction ideas that turn the basic policy-gradient estimator into a more useful foundation for LLM post-training.

The main path is:

```text
REINFORCE Gradient

↓

High Variance

↓

Subtract a Baseline

↓

Value Function Vπ

↓

Action-Value Function Qπ

↓

Advantage Function Aπ

↓

Map the Framework to an Autoregressive LLM
```

---

## Learning Objectives

After completing this chapter, you should be able to:

- explain why REINFORCE has high variance,
- show why a state-dependent baseline does not bias the policy gradient,
- define `Vπ`, `Qπ` and `Aπ`,
- interpret advantage as relative action quality,
- use a sampled return to estimate advantage,
- map prompts, tokens and responses to RL states, actions and trajectories,
- distinguish pretraining, SFT and post-training,
- explain what remains missing before PPO-style optimization.

---

# 223. Starting Point: The REINFORCE Gradient

The policy-gradient estimator is:

`∇θJ(θ) = Eτ[Σₜ ∇θ log πθ(Aₜ|Sₜ) Gₜ]`

It increases the probability of actions associated with large returns and decreases the probability of actions associated with poor returns.

The mathematics is correct, but the estimator may be noisy.

---

# 224. What Does High Variance Mean?

Suppose the same policy is evaluated several times.

The sampled gradients may differ greatly:

```text
Trajectory 1 → Strong Positive Update

Trajectory 2 → Small Negative Update

Trajectory 3 → Very Large Positive Update

Trajectory 4 → Large Negative Update
```

Even when the average direction is useful, individual updates may fluctuate strongly.

High variance can make training:

- unstable,
- slow,
- sensitive to rare outcomes,
- dependent on large batches of trajectories.

---

# 225. Why Returns Have High Variance

A return contains many sources of randomness:

- the sampled action at the current step,
- future sampled actions,
- stochastic environment transitions,
- long-horizon rewards,
- randomness unrelated to the current action.

The estimator may credit or blame one action for outcomes caused by many later events.

This is a credit-assignment problem.

---

# 226. Baseline Subtraction

We can subtract a baseline from the return:

`∇θJ(θ) = Eτ[Σₜ ∇θ log πθ(Aₜ|Sₜ)(Gₜ − b(Sₜ))]`

The baseline should not depend on the sampled action `Aₜ`.

A good baseline removes predictable variation while preserving information about whether the chosen action performed better or worse than expected.

---

# 227. Why Does the Baseline Not Bias the Gradient?

Consider the expected baseline term for a fixed state:

`Eₐ∼πθ[∇θ log πθ(a|s) b(s)]`

Because `b(s)` does not depend on the action:

`Eₐ[∇θ log πθ(a|s) b(s)] = b(s) Σₐ πθ(a|s) ∇θ log πθ(a|s)`

Using `πθ(a|s)∇θlogπθ(a|s)=∇θπθ(a|s)`:

`b(s) Σₐ ∇θπθ(a|s) = b(s) ∇θ Σₐπθ(a|s)`

Since probabilities sum to one:

`b(s) ∇θ1 = 0`

Therefore, subtracting an action-independent baseline changes variance but not the expected gradient.

---

# 228. Baseline Intuition

Suppose two students score `80`.

For an easy exam with an average of `78`, a score of `80` is only slightly above expectation.

For a difficult exam with an average of `45`, a score of `80` is exceptional.

The raw score alone is incomplete.

What matters is:

`Performance − Expected Performance`

The baseline provides the expected-performance reference.

---

# 229. Constant Baseline Versus State-Dependent Baseline

A constant baseline might be the average return across a batch.

`b = average return`

A state-dependent baseline is more informative:

`b = b(Sₜ)`

Different states may naturally have very different expected outcomes.

For example, a chess position one move from checkmate has a different expected return from an opening position.

---

# 230. Value Function

The state-value function is:

`Vπ(s) = Eπ[Gₜ | Sₜ=s]`

It means:

> If the agent starts from state `s` and follows policy `π`, what return should it expect?

The value function evaluates the quality of a state under the policy.

---

# 231. Value-Function Intuition

Imagine a race.

Starting near the front may produce a larger expected chance of winning than starting far behind.

The value function measures the expected outcome associated with the starting situation.

It does not fix the first action.

It averages over actions selected according to the policy.

---

# 232. Action-Value Function

The action-value function, or Q function, is:

`Qπ(s,a) = Eπ[Gₜ | Sₜ=s, Aₜ=a]`

It means:

> Starting in state `s`, take action `a` first, then follow policy `π`. What return should be expected?

Unlike `Vπ(s)`, the Q function evaluates a specific first action.

---

# 233. Value and Q Are Related

The value function averages Q values over the policy's action distribution:

`Vπ(s) = Eₐ∼π(·|s)[Qπ(s,a)]`

For a discrete action space:

`Vπ(s) = Σₐ π(a|s) Qπ(s,a)`

This relation is intuitive:

- `Qπ(s,a)` evaluates each possible action,
- `Vπ(s)` averages those evaluations according to how likely the policy is to choose each action.

---

# 234. Advantage Function

The advantage function is:

`Aπ(s,a) = Qπ(s,a) − Vπ(s)`

It measures how much better or worse action `a` is than the policy's average action in state `s`.

---

# 235. Advantage Interpretation

If:

`Aπ(s,a) > 0`

then action `a` is better than expected in state `s`.

If:

`Aπ(s,a) < 0`

then action `a` is worse than expected.

If:

`Aπ(s,a) = 0`

then the action is approximately average under the current policy.

---

# 236. Real-Life Advantage Example

Suppose a person's normal commute from home takes `40` minutes.

That expected time is the baseline.

A particular route takes `30` minutes.

The route gives a positive advantage of `10` minutes relative to expectation.

Another route takes `55` minutes.

That route has negative advantage.

The advantage compares an action against what is normal for the current state.

---

# 237. Policy Gradient with Advantage

Using the advantage function, the policy gradient can be written as:

`∇θJ(θ) = Eτ[Σₜ ∇θ log πθ(Aₜ|Sₜ) Aπ(Sₜ,Aₜ)]`

This form gives a clearer learning signal.

```text
Positive Advantage

↓

Increase Action Probability
```

```text
Negative Advantage

↓

Decrease Action Probability
```

---

# 238. Sampled Return as a Q Estimate

The true Q function is an expectation over possible futures.

A sampled return from one trajectory provides a Monte Carlo estimate:

`Gₜ ≈ Qπ(Sₜ,Aₜ)`

Therefore, an advantage estimate is:

`Âₜ = Gₜ − Vπ(Sₜ)`

This is the correct interpretation of return-minus-value.

The immediate reward alone is generally not the same as the return.

---

# 239. Why `Gₜ − Vπ(Sₜ)` Reduces Variance

The value function predicts the part of the return that is expected from the state.

Subtracting it leaves the surprising part:

`Observed Return − Expected Return`

This reduces variation caused by states that are naturally easy or difficult.

The remaining signal focuses more directly on the chosen action.

---

# 240. Actor and Critic Intuition

A common architecture uses two learned components.

## Actor

The actor is the policy:

`πθ(a|s)`

It chooses actions.

## Critic

The critic estimates value:

`Vφ(s)`

It judges how promising the state is.

The actor changes behavior.

The critic provides a lower-variance learning signal.

---

# 241. Actor-Critic Update

A common estimated advantage is:

`Âₜ = Gₜ − Vφ(Sₜ)`

The actor loss is:

`L_actor(θ) = −E[Σₜ log πθ(Aₜ|Sₜ) Âₜ]`

The critic may be trained by regression:

`L_value(φ) = E[(Vφ(Sₜ) − Gₜ)²]`

The actor and critic solve related but different learning problems.

---

# 242. What Must Be Computable?

For policy-gradient training, we need:

1. `log πθ(Aₜ|Sₜ)` for sampled actions,
2. an estimate of `Aπ(Sₜ,Aₜ)`.

For a neural policy, log probabilities are directly available from its output distribution.

The harder engineering problem is constructing reliable rewards and advantage estimates.

---

# 243. Mapping an Autoregressive LLM to Reinforcement Learning

An autoregressive language model predicts:

`Pθ(xₜ | x₍<t₎, prompt)`

This is already a probability distribution over possible next tokens.

Therefore, it can be interpreted as a policy:

`πθ(Aₜ | Sₜ) = Pθ(next token | prompt and generated prefix)`

The transformer itself is the policy network.

---

# 244. LLM State

The initial state is the prompt:

`S₀ = prompt`

After the model generates tokens, the state becomes the prompt plus the generated prefix:

`Sₜ = prompt ⊕ (A₀,A₁,...,Aₜ₋₁)`

Here, `⊕` denotes concatenation.

The state grows as generation proceeds.

---

# 245. LLM Action

The action is one generated token:

`Aₜ = next generated token`

The action space is the vocabulary.

For each state, the transformer outputs logits over all vocabulary tokens.

Softmax converts those logits into:

`πθ(a|s)`

---

# 246. Recursive State Construction

The sequence evolves as:

`S₁ = S₀ ⊕ A₀`

`S₂ = S₁ ⊕ A₁`

`S₃ = S₂ ⊕ A₂`

In general:

`Sₜ₊₁ = Sₜ ⊕ Aₜ`

The state is not equal to the action.

The action is one token; the next state contains the entire context after appending that token.

---

# 247. LLM Trajectory

A complete generated response is a trajectory:

`τ = (S₀,A₀,S₁,A₁,...,S_T)`

Equivalently, once the prompt is fixed, the generated token sequence identifies the trajectory:

`y = (A₀,A₁,...,A_T)`

The response probability factorizes autoregressively:

`πθ(y|x) = ∏ₜ πθ(Aₜ|Sₜ)`

Taking logarithms gives:

`log πθ(y|x) = Σₜ log πθ(Aₜ|Sₜ)`

---

# 248. What Is the Environment for an LLM?

The environment includes everything outside the policy that determines feedback and episode structure.

Depending on the setup, it may include:

- the prompt dataset,
- a user or simulator,
- a reward model,
- a correctness checker,
- safety constraints,
- a tool or external system,
- termination rules.

In a simple post-training setup, the policy generates a response and a reward model assigns a scalar score to that response.

---

# 249. Reward Timing in Language Generation

A reward can be assigned:

- after every token,
- at selected steps,
- only after the complete response.

In many LLM-alignment setups, the most important reward is sequence-level:

`R(τ) = score of the complete response`

That terminal reward must then influence all token decisions that produced the response.

This makes credit assignment challenging.

---

# 250. Token Log Probabilities Are Directly Available

The transformer outputs a probability distribution over the vocabulary.

For the token actually sampled at time `t`, we can obtain:

`log πθ(Aₜ|Sₜ)`

Automatic differentiation can compute:

`∇θ log πθ(Aₜ|Sₜ)`

This is why an autoregressive language model fits naturally into the policy-gradient framework.

---

# 251. The Three Training Stages

Modern LLM development is commonly organized into three conceptual stages.

```text
Pretraining

↓

Supervised Fine-Tuning

↓

Preference-Based Post-Training
```

All three stages may update transformer parameters, but they use different data and objectives.

---

# 252. Stage 1 — Pretraining

Pretraining learns next-token prediction from large text corpora.

For a token sequence, the loss is:

`L_pretrain(θ) = −Σₜ log Pθ(xₜ | x₍<t₎)`

The target token comes directly from the observed text.

The model learns broad language structure and knowledge patterns.

---

# 253. Stage 2 — Supervised Fine-Tuning

SFT uses curated prompt-response demonstrations.

The mathematical loss still resembles next-token cross-entropy:

`L_SFT(θ) = −Σₜ log Pθ(yₜ | x,y₍<t₎)`

The difference is the data.

The desired responses are selected or written to teach instruction-following behavior.

---

# 254. Stage 3 — Post-Training or Alignment

Post-training uses feedback about generated behavior.

The transformer is treated as a policy, and the objective prefers responses associated with larger rewards or stronger human preferences.

A simplified objective is:

`J_align(θ) = Eₓ∼D, y∼πθ(·|x)[R(x,y)]`

The gradient is estimated through response log probabilities rather than by differentiating through the reward source.

---

# 255. Why Not Train All Three Stages Together?

The stages use different resources and solve different problems.

## Pretraining

- enormous general-purpose text,
- broad next-token learning,
- expensive but reusable base capability.

## SFT

- smaller curated demonstrations,
- task and instruction behavior.

## Post-Training

- preference or reward feedback,
- behavioral refinement.

A single pretrained model can be adapted into multiple specialized assistants using different SFT and alignment datasets.

---

# 256. Does Post-Training Replace Pretraining?

No.

Post-training starts from an already capable language model.

It does not teach language from scratch.

It changes the probabilities of behaviors that the base model can already express.

Conceptually:

```text
Pretraining

Builds Capability
```

```text
SFT

Teaches Instruction Patterns
```

```text
Alignment

Shapes Behavioral Preferences
```

---

# 257. What Remains Before PPO?

We now have:

- a policy-gradient objective,
- token log probabilities,
- value and advantage concepts,
- an LLM-as-policy mapping.

However, plain policy-gradient updates can still move the model too far in one step.

A practical method must address:

- unstable policy updates,
- reuse of sampled data,
- probability ratios between old and new policies,
- clipping or trust-region control,
- value estimation,
- reward-model quality.

These ideas motivate PPO and related methods.

---

# 258. Scope Boundary of This Lesson

The transcript introduced PPO as the next major method but did not complete its detailed derivation.

Therefore, this chapter stops at the mathematical foundation needed for PPO:

`Policy Gradient + Advantage Estimation + LLM Policy Mapping`

The next theory chapter should derive controlled policy updates rather than prematurely treating PPO as completed.

---

# Common Misconceptions

## Misconception 1

**Subtracting a baseline removes reward from learning.**

It does not. The advantage still depends on return. The baseline removes predictable variation.

---

## Misconception 2

**The Q function and value function are the same.**

`Vπ(s)` averages over actions. `Qπ(s,a)` conditions on one specific action.

---

## Misconception 3

**Advantage is the immediate reward minus value.**

In the Monte Carlo form, advantage is estimated using return: `Âₜ = Gₜ − Vπ(Sₜ)`.

---

## Misconception 4

**An LLM action is the complete response.**

At the token-level MDP, one action is one token. The complete response is a trajectory.

---

## Misconception 5

**The next LLM state is only the latest token.**

The next state is the prompt and the entire generated prefix after appending the latest token.

---

## Misconception 6

**Alignment trains the transformer from random initialization.**

Alignment modifies an already pretrained and usually supervised-fine-tuned transformer.

---

# Chapter Summary

REINFORCE has a correct but high-variance gradient estimator. Subtracting an action-independent baseline preserves the expected gradient while reducing variance.

The state-value function is:

`Vπ(s) = Eπ[Gₜ|Sₜ=s]`

The action-value function is:

`Qπ(s,a) = Eπ[Gₜ|Sₜ=s,Aₜ=a]`

The advantage function is:

`Aπ(s,a) = Qπ(s,a) − Vπ(s)`

A sampled return gives the practical estimate:

`Âₜ = Gₜ − Vπ(Sₜ)`

For an LLM, the transformer is the policy, the prompt and generated prefix form the state, one token is an action, and the complete response is a trajectory.

Pretraining, SFT and post-training all operate on transformer parameters, but they use different data and objectives.

---

# Key Takeaways

✅ A baseline reduces gradient variance without changing the expected policy gradient.

✅ `Vπ(s)` measures expected return from a state.

✅ `Qπ(s,a)` measures expected return after taking a particular action.

✅ Advantage measures whether an action is better or worse than the policy's expectation.

✅ In an LLM, the transformer output distribution is the policy distribution.

✅ One generated token is an action; the prompt plus prefix is the state.

✅ A complete response is an RL trajectory.

✅ Pretraining builds broad capability, SFT teaches demonstrations, and alignment shapes preferences.

✅ PPO belongs to the next derivation after these foundations.
