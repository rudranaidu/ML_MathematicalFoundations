# 16. Reinforcement Learning Foundations and the Policy Gradient Theorem

This chapter continues the LLM-alignment discussion by developing the minimum Reinforcement Learning mathematics needed for policy-based post-training.

The main path is:

```text
Agent–Environment Interaction

↓

Markov Decision Process

↓

Policy and Trajectory Distribution

↓

Expected Return Objective

↓

Log-Derivative Trick

↓

Policy Gradient Theorem

↓

REINFORCE
```

---

## Learning Objectives

After completing this chapter, you should be able to:

- distinguish supervised, unsupervised and Reinforcement Learning feedback,
- identify the components of a Markov Decision Process,
- explain why the policy is not part of the environment definition,
- write the probability of a trajectory,
- define discounted return and expected return,
- derive the policy gradient using the log-derivative trick,
- explain why the environment transition model disappears from the final gradient,
- connect the mathematical result to the REINFORCE algorithm.

---

## Prerequisites

You should already be comfortable with:

- conditional probability,
- the chain rule of probability,
- expectations,
- gradients,
- logarithms,
- neural-network optimization,
- autoregressive language models.

---

# 189. Why Reinforcement Learning Appears in LLM Alignment

During pretraining, a language model receives exact next-token targets from text.

During supervised fine-tuning, it receives curated prompt-response demonstrations.

During alignment, the desired output is often not a single exact answer.

For one prompt, many responses may be acceptable, but some may be:

- more helpful,
- more accurate,
- safer,
- clearer,
- more relevant,
- better aligned with a preferred behavior.

This is closer to **feedback about behavior** than to an exact label.

Reinforcement Learning provides a framework for learning from this type of feedback.

---

# 190. Supervised Learning Versus Reinforcement Learning

## Supervised Learning

A supervised dataset contains pairs:

`(x, y)`

The target `y` tells the model what the correct output should be.

For example:

```text
Image → Cat

Email → Spam

Sentence → Translation
```

## Reinforcement Learning

In Reinforcement Learning, the learner performs actions and receives rewards.

The reward tells the learner whether its behavior was useful, but it may not specify the exact action that should have been taken.

```text
State

↓

Agent Chooses Action

↓

Environment Produces Reward and Next State
```

This feedback is weaker than a direct label, but it can express preferences over complete behaviors.

---

# 191. The Agent–Environment Loop

Reinforcement Learning contains two central participants.

## Agent

The agent chooses actions.

Examples:

- a robot,
- a chess player,
- a self-driving system,
- a language model generating tokens.

## Environment

The environment receives the action and determines:

- the next state,
- the reward.

The interaction repeats:

```text
Current State Sₜ

↓

Agent Chooses Aₜ

↓

Environment Returns Rₜ and Sₜ₊₁

↓

Repeat
```

---

# 192. State

A **state** summarizes the situation in which the agent currently operates.

It is written as:

`Sₜ`

Examples:

- the current board configuration in chess,
- the robot's current position,
- the prompt and generated prefix in an LLM.

The state should contain enough information for choosing the next action.

---

# 193. Action

An **action** is a choice made by the agent.

It is written as:

`Aₜ`

Examples:

- move a chess piece,
- turn left,
- apply a control signal,
- generate the next token.

The set of all possible actions is the action space:

`𝒜`

---

# 194. Transition Probability

After the agent chooses action `Aₜ` in state `Sₜ`, the environment moves to the next state `Sₜ₊₁`.

This is modeled by:

`P(Sₜ₊₁ | Sₜ, Aₜ)`

It means:

> Given the current state and action, what is the probability of reaching a particular next state?

The transition may be deterministic or probabilistic.

A deterministic transition is a special case in which one next state has probability `1`.

---

# 195. Why Model Transitions Probabilistically?

Real environments may contain:

- incomplete observations,
- unknown physical effects,
- randomness,
- other agents,
- complex dynamics that are difficult to model exactly.

A probabilistic transition model captures uncertainty by describing repeated behavior rather than demanding a perfect deterministic equation.

This is the same reason probability is useful throughout machine learning.

---

# 196. Reward

The reward is a real-valued signal returned by the environment.

`R(Sₜ, Aₜ) ∈ ℝ`

A positive reward encourages behavior.

A negative reward discourages behavior.

Examples:

```text
Win a Game       → +1

Lose a Game      → −1

Reach a Goal     → +10

Unsafe Response  → Negative Reward
```

The quality of Reinforcement Learning is strongly limited by the quality of the reward signal.

---

# 197. Initial-State Distribution

An episode must begin somewhere.

The initial state is sampled from:

`S₀ ∼ ρ₀`

Here, `ρ₀(S₀)` is the probability that the process starts in state `S₀`.

For an LLM, the initial-state distribution may correspond to a distribution over prompts.

---

# 198. Discount Factor

The discount factor is:

`γ ∈ [0,1]`

It controls how future rewards are weighted.

A reward received `k` steps later is multiplied by:

`γᵏ`

When `γ` is close to `0`, immediate rewards dominate.

When `γ` is close to `1`, long-term rewards remain important.

---

# 199. The Markov Decision Process

A common convention describes an MDP using:

`MDP = (𝒮, 𝒜, P, R, ρ₀)`

where:

- `𝒮` is the state space,
- `𝒜` is the action space,
- `P` is the transition distribution,
- `R` is the reward function,
- `ρ₀` is the initial-state distribution.

The discount factor `γ` is sometimes included in the tuple and sometimes specified separately.

---

# 200. Why Is the Policy Not Part of the MDP?

The MDP describes the **rules of the world**.

The policy describes **how one agent behaves inside that world**.

Chess provides a useful analogy.

```text
Chessboard + Pieces + Legal Moves

=

Environment or MDP
```

```text
How a Particular Player Chooses Moves

=

Policy
```

Many different players can operate in the same chess environment.

Therefore, the MDP is agent-agnostic, while the policy belongs to the agent.

---

# 201. Policy

A policy is a probability distribution over actions conditioned on the state.

`πθ(a | s) = P(Aₜ=a | Sₜ=s)`

The parameters `θ` may be the weights of a neural network.

A stochastic policy can assign nonzero probability to multiple actions.

Example:

```text
Current State

Action Left  → 0.20
Action Right → 0.65
Action Wait  → 0.15
```

The policy defines the agent's behavior.

---

# 202. Why Use a Stochastic Policy?

A stochastic policy supports:

- exploration,
- uncertainty,
- multiple acceptable actions,
- sampling diverse behaviors,
- differentiable probability-based optimization.

For an LLM, the policy is naturally stochastic because the model outputs a probability distribution over the vocabulary.

---

# 203. Trajectory

A trajectory is a sequence of states and actions produced during interaction.

`τ = (S₀, A₀, S₁, A₁, ..., S_T)`

A trajectory may also include rewards:

`τ = (S₀, A₀, R₀, S₁, A₁, R₁, ..., S_T)`

Examples:

- one complete game,
- one robot-navigation episode,
- one complete LLM response.

---

# 204. Probability of a Trajectory

The trajectory probability is obtained using the chain rule of probability.

`pθ(τ) = ρ₀(S₀) · ∏ₜ πθ(Aₜ | Sₜ) P(Sₜ₊₁ | Sₜ, Aₜ)`

This equation contains three sources of probability:

1. how the episode starts,
2. how the agent chooses actions,
3. how the environment transitions.

---

# 205. Reading the Trajectory Probability Step by Step

The process begins with:

`S₀ ∼ ρ₀`

The policy chooses:

`A₀ ∼ πθ(· | S₀)`

The environment transitions:

`S₁ ∼ P(· | S₀,A₀)`

The policy then chooses:

`A₁ ∼ πθ(· | S₁)`

This repeats until the episode terminates.

Multiplying these probabilities gives the probability of the complete trajectory.

---

# 206. Immediate Reward Versus Return

An immediate reward evaluates one interaction.

`Rₜ = R(Sₜ,Aₜ)`

A return accumulates rewards over time.

`Gₜ = Σₖ₌₀∞ γᵏ Rₜ₊ₖ`

The return answers:

> Starting from time `t`, how much discounted reward will the agent receive?

---

# 207. Why Optimize Return Instead of Immediate Reward?

An action may have delayed consequences.

For example, in chess, sacrificing a piece may have a negative immediate effect but produce a later win.

Optimizing only `Rₜ` may encourage short-sighted behavior.

Optimizing `Gₜ` allows current actions to receive credit for future outcomes.

---

# 208. Reinforcement Learning Objective

The policy should maximize expected return.

`J(θ) = Eτ∼pθ(τ)[G₀]`

The optimization problem is:

`θ* = arg maxθ J(θ)`

The expectation is over trajectories generated by the current policy and the environment.

---

# 209. Why Is This Objective Difficult?

The objective depends on a trajectory distribution:

`pθ(τ)`

We usually cannot enumerate every possible trajectory.

We may also not know the environment transition probabilities explicitly.

However, we can interact with the environment and collect sampled trajectories.

The central question becomes:

> How can we estimate `∇θJ(θ)` using samples?

---

# 210. Policy Gradient Approach

The policy is represented by a differentiable model.

We update its parameters using gradient ascent:

`θ ← θ + η ∇θJ(θ)`

where `η` is the learning rate.

The challenge is deriving a computable expression for `∇θJ(θ)`.

---

# 211. Write the Expectation as an Integral

The expected-return objective can be written as:

`J(θ) = ∫ pθ(τ) G₀(τ) dτ`

Therefore:

`∇θJ(θ) = ∇θ ∫ pθ(τ) G₀(τ) dτ`

Assuming regularity conditions allow differentiation inside the integral:

`∇θJ(θ) = ∫ ∇θpθ(τ) G₀(τ) dτ`

The return is treated as a measured outcome of the trajectory rather than a differentiable function of `θ`.

---

# 212. The Log-Derivative Trick

For any positive differentiable function `fθ(x)`:

`∇θ log fθ(x) = ∇θfθ(x) / fθ(x)`

Rearranging gives:

`∇θfθ(x) = fθ(x) ∇θ log fθ(x)`

Applying it to the trajectory distribution:

`∇θpθ(τ) = pθ(τ) ∇θ log pθ(τ)`

This is also called the score-function trick or likelihood-ratio trick.

---

# 213. Insert the Log-Derivative Trick

Substitute the identity into the gradient:

`∇θJ(θ) = ∫ pθ(τ) ∇θ log pθ(τ) G₀(τ) dτ`

Recognize the integral as an expectation:

`∇θJ(θ) = Eτ∼pθ[∇θ log pθ(τ) G₀(τ)]`

We have converted the gradient of a probability distribution into the probability distribution multiplied by a gradient of log probability.

---

# 214. Expand the Log Trajectory Probability

Recall:

`pθ(τ) = ρ₀(S₀) · ∏ₜ πθ(Aₜ | Sₜ) P(Sₜ₊₁ | Sₜ,Aₜ)`

Taking logarithms converts products into sums:

`log pθ(τ) = log ρ₀(S₀) + Σₜ log πθ(Aₜ | Sₜ) + Σₜ log P(Sₜ₊₁ | Sₜ,Aₜ)`

Now differentiate with respect to `θ`.

---

# 215. Why the Environment Terms Disappear

The initial-state distribution and environment transitions are not parameterized by the policy parameters `θ`.

Therefore:

`∇θ log ρ₀(S₀) = 0`

and:

`∇θ log P(Sₜ₊₁ | Sₜ,Aₜ) = 0`

Only the policy term remains:

`∇θ log pθ(τ) = Σₜ ∇θ log πθ(Aₜ | Sₜ)`

This is a major result.

We do not need to differentiate through the unknown environment dynamics.

---

# 216. Policy Gradient Theorem in Trajectory Form

Substituting the remaining term gives:

`∇θJ(θ) = Eτ∼pθ[(Σₜ ∇θ log πθ(Aₜ | Sₜ)) G₀(τ)]`

A more useful credit-assignment form uses the return from each time step:

`∇θJ(θ) = Eτ∼pθ[Σₜ ∇θ log πθ(Aₜ | Sₜ) Gₜ]`

This is the central policy-gradient estimator used by REINFORCE-style methods.

---

# 217. Intuition Behind the Policy Gradient

Consider one term:

`∇θ log πθ(Aₜ | Sₜ) Gₜ`

If `Gₜ` is large and positive, gradient ascent increases the probability of the sampled action.

If `Gₜ` is negative, gradient ascent decreases the probability of that action.

Conceptually:

```text
Action Followed by High Return

↓

Make That Action More Likely in Similar States
```

```text
Action Followed by Low Return

↓

Make That Action Less Likely in Similar States
```

---

# 218. Why the Log Probability Is Useful

A trajectory probability is a product of many action probabilities.

Differentiating that product directly is awkward.

Taking the logarithm converts it into a sum:

`log ∏ₜ πθ(Aₜ|Sₜ) = Σₜ log πθ(Aₜ|Sₜ)`

This makes each action's contribution explicit and easy to compute with automatic differentiation.

---

# 219. Monte Carlo Approximation

The expectation cannot usually be computed exactly.

Suppose we sample `N` trajectories.

A Monte Carlo estimator is:

`∇θJ(θ) ≈ (1/N) Σᵢ₌₁ᴺ Σₜ ∇θ log πθ(Aₜ⁽ⁱ⁾ | Sₜ⁽ⁱ⁾) Gₜ⁽ⁱ⁾`

This follows from the law of large numbers.

More sampled trajectories generally produce a more reliable estimate, but they also increase computation.

---

# 220. REINFORCE Algorithm

REINFORCE is the direct Monte Carlo implementation of the policy-gradient estimator.

```text
1. Sample trajectories using the current policy.

2. Compute returns for each time step.

3. Compute log probabilities of the sampled actions.

4. Multiply each log-probability gradient by its return.

5. Average over samples.

6. Perform gradient ascent.
```

A loss suitable for standard gradient descent is:

`L_REINFORCE(θ) = −E[Σₜ log πθ(Aₜ|Sₜ) Gₜ]`

Minimizing this loss is equivalent to maximizing expected return.

---

# 221. Minimal Pseudocode

```python
for batch in training_iterations:
    trajectories = collect_trajectories(policy)

    loss = 0.0

    for trajectory in trajectories:
        returns = discounted_returns(trajectory.rewards, gamma)

        for log_prob, return_t in zip(trajectory.log_probs, returns):
            loss += -log_prob * return_t

    loss /= len(trajectories)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The environment need not be differentiable.

Only `log_prob` must be differentiable with respect to the policy parameters.

---

# 222. Why Plain REINFORCE Is Not Enough

REINFORCE is conceptually elegant, but its gradient estimate can have high variance.

Two trajectories generated from similar states may produce very different returns.

This may cause:

- unstable gradients,
- slow learning,
- sensitivity to random samples,
- large differences between updates.

The next chapter introduces baselines, value functions, Q functions and advantage functions to reduce this variance and prepare for modern alignment algorithms.

---

# Common Misconceptions

## Misconception 1

**The policy is part of the MDP.**

The MDP describes the environment. The policy describes how an agent behaves within that environment.

---

## Misconception 2

**A policy must be deterministic.**

A policy is commonly modeled as a probability distribution over actions.

---

## Misconception 3

**Policy gradients require a differentiable environment.**

They do not. The environment terms disappear because they are independent of policy parameters.

---

## Misconception 4

**We differentiate the numerical reward directly.**

The reward may be observed without having a differentiable reward formula. The gradient flows through the action log probabilities.

---

## Misconception 5

**A high immediate reward always identifies the best action.**

An action may have delayed consequences. Return captures future rewards.

---

# Chapter Summary

We modeled Reinforcement Learning using an MDP containing states, actions, transition probabilities, rewards and an initial-state distribution.

A policy `πθ(a|s)` describes the agent's stochastic behavior. When the policy interacts with the environment, it generates trajectories with probability `pθ(τ)`.

The learning objective is to maximize expected discounted return:

`J(θ) = Eτ∼pθ(τ)[G₀]`

Using the log-derivative trick and the factorization of the trajectory distribution, we obtained:

`∇θJ(θ) = Eτ[Σₜ ∇θ log πθ(Aₜ|Sₜ) Gₜ]`

The initial-state and transition terms vanish because they do not depend on `θ`.

REINFORCE estimates this expectation using sampled trajectories, but its estimator often has high variance.

---

# Key Takeaways

✅ An MDP defines the environment, not the agent's policy.

✅ A policy is a distribution over actions conditioned on state.

✅ A trajectory is a sequence of state-action interactions.

✅ The RL objective maximizes expected discounted return.

✅ The log-derivative trick converts an intractable distribution gradient into a sample-based expectation.

✅ Environment dynamics disappear from the policy gradient when they are independent of policy parameters.

✅ REINFORCE is the direct Monte Carlo policy-gradient algorithm.

✅ Variance reduction is the next essential step toward practical policy optimization.
