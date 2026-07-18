 **Core Theme:** Learn how an agent improves its behavior from rewards, and understand how the Policy Gradient 
 Theorem converts an apparently intractable Reinforcement Learning objective into a gradient that can be estimated from sampled experience.

---

## Learning Objectives

After completing this chapter, you should be able to:

- explain why Reinforcement Learning is useful when an exact target answer is unavailable,
- distinguish an agent, environment, state, action, reward and policy,
- explain a Markov Decision Process using an everyday example,
- distinguish the rules of an environment from the strategy of an agent,
- describe a trajectory and calculate its probability,
- distinguish immediate reward, return and expected return,
- explain why the Reinforcement Learning objective is difficult to differentiate directly,
- derive the Policy Gradient Theorem using the log-derivative trick,
- interpret the policy-gradient update in plain language,
- explain how REINFORCE estimates the gradient using sampled trajectories.

---

## Prerequisites

You should already be comfortable with:

- conditional probability,
- the Chain Rule of Probability,
- expectations,
- gradients,
- logarithms,
- neural-network optimization,
- basic autoregressive language modeling.

---

## Chapter Roadmap

```text
Learning from Feedback

↓

Agent and Environment

↓

Markov Decision Process

↓

Policy and Trajectory

↓

Reward and Return

↓

Expected-Return Objective

↓

Log-Derivative Trick

↓

Policy Gradient Theorem

↓

REINFORCE
```

---

# 189. Why Reinforcement Learning Appears in LLM Alignment

During ordinary supervised learning, the model is shown the desired answer.

```text
Input

↓

Correct Target

↓

Learn to Match the Target
```

For example:

```text
Image → Cat

Email → Spam

Sentence → Translation
```

The target tells the model exactly what it should produce.

Language-model alignment is often different.

For one prompt, there may be several reasonable answers.

Example prompt:

```text
Explain overfitting to a beginner.
```

Possible answers may differ in:

- wording,
- length,
- examples,
- mathematical depth,
- tone.

There may not be one uniquely correct response.

However, a human can often say:

```text
Response B is more helpful than Response A.
```

This is not an exact label.

It is feedback about behavior.

Reinforcement Learning provides a framework for learning from this type of feedback.

---

## Answer Key Versus Coach

A simple analogy helps.

### Supervised Learning

A teacher gives you:

```text
Question

+

Exact Answer Key
```

You know exactly what should have been written.

### Reinforcement Learning

A coach watches you play and says:

```text
Good move.

Bad move.

You won.

You lost.
```

The coach may not tell you the exact move you should have made at every moment.

You must learn which decisions tend to produce good outcomes.

That is the central idea of Reinforcement Learning.

---

# 190. Supervised, Unsupervised and Reinforcement Learning

These three paradigms provide different kinds of information.

| Learning type | Information available | Simple analogy |
|---|---|---|
| Supervised | Exact target labels | Teacher gives the answer key |
| Unsupervised | Data without labels | Organize books without category labels |
| Reinforcement Learning | Rewards from actions | Learn a game from wins and losses |

In Reinforcement Learning:

- the output is not always specified exactly,
- actions affect what happens next,
- rewards may arrive later,
- the learner must discover useful behavior through interaction.

---

# 191. A Running Example: A Delivery Robot

We will use one simple example throughout the chapter.

Imagine a small delivery robot inside an office.

Its task is to carry a package from the reception desk to Room B.

The office contains:

```text
Reception

↓

Corridor

↓

Room A

↓

Room B
```

The robot can perform actions such as:

```text
Move Forward

Turn Left

Turn Right

Stop
```

It may receive rewards such as:

```text
Reach Room B       → +10

Hit a Wall         → −5

Move One Step      → −0.1

Deliver Package    → +20
```

This example will help us understand every Reinforcement Learning concept.

---

# 192. The Agent–Environment Interaction

Reinforcement Learning contains two central participants.

## Agent

The **agent** is the decision-making entity.

In our example:

```text
Agent = Delivery Robot
```

Other examples include:

| Problem | Agent |
|---|---|
| Chess | Chess-playing program |
| Self-driving | Driving controller |
| Video game | Game-playing algorithm |
| LLM alignment | Language model |
| Robot navigation | Robot |

## Environment

The **environment** is the world in which the agent operates.

For the delivery robot, it includes:

- rooms,
- corridors,
- walls,
- doors,
- obstacles,
- the package destination.

One interaction step is:

```text
Current Situation

↓

Agent Chooses an Action

↓

Environment Responds

↓

Reward + New Situation

↓

Repeat
```

For example:

```text
Robot Is in Corridor

↓

Action = Move Forward

↓

Robot Reaches Room A

↓

Reward = −0.1
```

---

# 193. State

A **state** is a description of the agent's current situation.

It is written as:

`Sₜ`

The subscript `t` means the state at time step `t`.

For the robot, a state may contain:

```text
Current Room

Current Direction

Package Status

Nearby Obstacles
```

A possible state could be:

`Sₜ = [Room A, facing east, carrying package, wall ahead]`

A state does not have to be one number.

It may be:

- a vector,
- an image,
- a board configuration,
- a text sequence,
- a collection of sensor readings.

---

## State as a Snapshot

Think of a state as one photograph.

```text
State

=

One Snapshot
```

A trajectory, which we will define later, is the complete video.

For an LLM, a state can be:

```text
User Prompt

+

Tokens Generated So Far
```

The state should contain enough information to make the next decision.

If the robot knows only its room but not its direction or nearby obstacles, the state may be incomplete.

---

# 194. Action

An **action** is a decision made by the agent.

It is written as:

`Aₜ`

For the robot:

`𝒜 = {forward, left, right, stop}`

The set `𝒜` is called the **action space**.

Examples:

## Chess

```text
All Legal Chess Moves
```

## Robot

```text
Forward

Left

Right

Stop
```

## Language Model

```text
Every Token in the Vocabulary
```

At one time step, the agent chooses one action.

A stochastic policy may assign probabilities to many actions, but only one action is selected at that step.

---

# 195. Reward

A **reward** is feedback returned by the environment.

It is written as:

`Rₜ`

or:

`R(Sₜ,Aₜ)`

A reward is usually a real number.

```text
Positive Reward

↓

Desired Outcome
```

```text
Negative Reward

↓

Undesired Outcome
```

For the robot:

```text
Correct Delivery → +20

Hit Wall         → −5

Every Step       → −0.1
```

The reward evaluates the consequence of behavior.

It usually does not provide the exact correct action.

---

## Reward Does Not Give the Answer Key

A reward may say:

```text
The delivery succeeded.

Reward = +20
```

It may not say:

```text
At time step 7, you should have turned left.
```

The agent must determine which earlier actions helped produce the success.

This is called part of the **credit-assignment problem**.

---

## Reward Design Matters

A Reinforcement Learning system is only as good as its reward.

Suppose we reward the robot only for moving quickly.

It may:

- hit walls,
- drop the package,
- enter the wrong room.

The agent optimizes what we measure, not necessarily what we intended.

```text
Poor Reward Design

↓

Poor Learned Behavior
```

This becomes especially important for LLMs, where qualities such as helpfulness, honesty and safety are difficult to measure perfectly.

---

# 196. Transition Probability

After an action is taken, the environment moves to a new state.

This is described by:

`P(Sₜ₊₁ | Sₜ,Aₜ)`

Read it as:

> Given the current state and action, what is the probability of reaching a particular next state?

---

## Deterministic Transition

A transition is deterministic when one action always produces one next state.

```text
Robot at Reception

+

Move Forward

↓

Robot Enters Corridor
```

If this always happens, that next state has probability `1`.

---

## Stochastic Transition

Imagine the floor is slippery.

The robot chooses:

```text
Move Forward
```

Possible outcomes may be:

| Next outcome | Probability |
|---|---:|
| Moves forward correctly | 0.80 |
| Slips left | 0.10 |
| Slips right | 0.10 |

The robot controls the action.

It does not completely control the outcome.

Real systems may contain:

- sensor noise,
- unknown physical effects,
- other agents,
- incomplete information,
- dynamics too complex to model exactly.

A probabilistic transition model captures this uncertainty.

---

# 197. Initial State and Discount Factor

Every episode must start somewhere.

The starting state is sampled from:

`S₀ ~ ρ₀`

For the robot:

```text
Reception → 0.70

Corridor  → 0.20

Room A    → 0.10
```

The symbol `ρ₀` describes the initial-state distribution.

---

## Discount Factor

The discount factor is:

`γ ∈ [0,1]`

It controls how strongly future rewards matter.

```text
γ close to 0

↓

Focus on Immediate Rewards
```

```text
γ close to 1

↓

Consider Long-Term Rewards
```

Think of `γ` as the agent's patience.

A reward received `k` steps later is weighted by `γᵏ`.

---

# 198. The Markov Decision Process

A **Markov Decision Process**, or MDP, describes the rules of the environment.

A common definition is:

`MDP = (𝒮, 𝒜, P, R, ρ₀, γ)`

where:

- `𝒮` is the state space,
- `𝒜` is the action space,
- `P` is the transition distribution,
- `R` is the reward function,
- `ρ₀` is the initial-state distribution,
- `γ` is the discount factor.

Some books list `γ` separately.

The core idea is unchanged.

---

## The Markov Property

The word **Markov** means that the next state depends only on:

- the current state,
- the current action.

The property is:

`P(Sₜ₊₁ | S₀,A₀,...,Sₜ,Aₜ) = P(Sₜ₊₁ | Sₜ,Aₜ)`

This does not mean that the past is irrelevant.

It means the current state should already summarize all relevant information from the past.

For an LLM, the state may contain the full prompt and generated prefix, so earlier tokens remain available.

---

# 199. Why the Policy Is Not Part of the MDP

The MDP describes the **rules of the world**.

The policy describes **how one agent behaves inside that world**.

Consider chess.

```text
Chessboard

Pieces

Legal Moves

Win/Loss Rules
```

These belong to the environment or MDP.

```text
How a Particular Player Chooses Moves
```

This is the policy.

Many players can use different strategies in the same chess environment.

Therefore:

```text
MDP

=

Rules of the World
```

```text
Policy

=

Agent's Strategy
```

The policy is not included in the MDP because the policy is the object we want to learn.

---

# 200. Policy

A **policy** is a probability distribution over actions given the state.

It is written as:

`πθ(a | s)`

Read it as:

> The probability that the agent chooses action `a` while in state `s`.

The parameters `θ` may be the weights of a neural network.

For the robot:

| Action | Probability |
|---|---:|
| Move forward | 0.60 |
| Turn left | 0.20 |
| Turn right | 0.15 |
| Stop | 0.05 |

The policy answers:

```text
Given My Current Situation

↓

What Should I Do?
```

---

## Deterministic and Stochastic Policies

A deterministic policy always selects one action.

A stochastic policy produces action probabilities.

Stochastic policies are useful because they support:

- exploration,
- uncertainty,
- multiple acceptable actions,
- diverse behavior,
- probability-based optimization.

For an LLM, the Transformer already outputs a probability distribution over vocabulary tokens, so it naturally behaves like a stochastic policy.

---

# 201. Trajectory

A **trajectory** is one complete sequence of states, actions and rewards produced during interaction.

It is written as:

`τ = (S₀,A₀,R₀,S₁,A₁,R₁,...,S_T)`

The Greek letter `τ`, pronounced **tau**, represents the trajectory.

For the robot:

```text
Reception

↓

Move Forward

↓

Corridor

↓

Turn Right

↓

Room B

↓

Deliver Package
```

That complete experience is one trajectory.

---

## State Versus Trajectory

```text
State

=

One Photograph
```

```text
Trajectory

=

Complete Video
```

In supervised learning, data may be input-label pairs.

In Reinforcement Learning, data is often a collection of trajectories.

---

# 202. Probability of a Trajectory

A trajectory is generated by three mechanisms:

1. how the episode starts,
2. how the policy chooses actions,
3. how the environment changes state.

Its probability is:

`pθ(τ) = ρ₀(S₀) × ∏ₜ πθ(Aₜ | Sₜ) P(Sₜ₊₁ | Sₜ,Aₜ)`

This is the Chain Rule of Probability applied across time.

---

## Reading the Equation Slowly

The process begins with:

`S₀ ~ ρ₀`

Then the policy chooses:

`A₀ ~ πθ(· | S₀)`

The environment produces:

`S₁ ~ P(· | S₀,A₀)`

Then the policy chooses:

`A₁ ~ πθ(· | S₁)`

The environment produces:

`S₂ ~ P(· | S₁,A₁)`

This repeats.

Multiplying the probability of every event gives the probability of the complete trajectory.

---

## Numerical Example

Suppose:

`ρ₀(S₀) = 1`

`πθ(A₀ | S₀) = 0.6`

`P(S₁ | S₀,A₀) = 0.9`

`πθ(A₁ | S₁) = 0.8`

`P(S₂ | S₁,A₁) = 0.95`

Then:

`pθ(τ) = 1 × 0.6 × 0.9 × 0.8 × 0.95 = 0.4104`

This is the probability of observing that exact sequence of states and actions.

---

# 203. Immediate Reward Versus Return

An immediate reward evaluates one step:

`Rₜ = R(Sₜ,Aₜ)`

A **return** summarizes future rewards.

The return from time `t` is:

`Gₜ = Rₜ + γRₜ₊₁ + γ²Rₜ₊₂ + ...`

Equivalently:

`Gₜ = Σₖ₌₀ γᵏRₜ₊ₖ`

The return asks:

> Starting now, how much future reward will this trajectory produce?

---

## Why Not Optimize Only Immediate Reward?

Suppose the robot is one step away from Room B.

Moving through the doorway costs:

`Rₜ = −0.1`

Delivering the package one step later gives:

`Rₜ₊₁ = +20`

If the robot optimizes only immediate reward, it may refuse to move.

Optimizing return allows a small short-term cost for a large later reward.

---

# 204. Discounted Return Example

Suppose:

`γ = 0.9`

and:

```text
R₀ = 2

R₁ = 3

R₂ = 10
```

Then:

`G₀ = 2 + 0.9 × 3 + 0.9² × 10`

Therefore:

`G₀ = 12.8`

The distant reward still matters, but it receives slightly less weight.

If `γ = 0`, only the immediate reward matters.

If `γ` is close to `1`, long-term rewards remain important.

---

# 205. Expected Return

One trajectory may be unusually lucky or unlucky.

Therefore, we evaluate the average performance of the policy over all trajectories it can generate.

The objective is:

`J(θ) = Eτ~pθ(τ)[G₀]`

Read this as:

> The average return obtained by trajectories generated by policy `πθ`.

---

## Expected-Return Intuition

Imagine evaluating a cricket player using one ball.

The player may:

- hit a six,
- get out,
- receive an easy delivery,
- face an unusually difficult delivery.

One ball is not enough.

Average performance over many attempts is more meaningful.

```text
One Trajectory

↓

May Be Lucky or Unlucky
```

```text
Expected Return

↓

Average Policy Quality
```

---

# 206. The Reinforcement Learning Objective

A good policy is one that produces a large expected return.

Therefore:

`θ* = argmaxθ J(θ)`

or:

`θ* = argmaxθ Eτ~pθ(τ)[G₀]`

Changing `θ` changes:

```text
Action Probabilities

↓

Generated Trajectories

↓

Collected Rewards

↓

Expected Return
```

The reward function may not explicitly contain `θ`, but the distribution of trajectories does.

---

# 207. The Repeating Statistical Pattern

This problem should look familiar.

We have:

```text
An Expectation

↓

Taken Under a Distribution

↓

The Distribution Is Difficult to Enumerate

↓

But We Can Draw Samples
```

The same pattern appeared in:

- maximum likelihood,
- variational inference,
- ELBO optimization,
- Monte Carlo estimation,
- diffusion-model training.

Here, the expectation is over trajectories.

We cannot list every possible trajectory, but we can generate sample trajectories by running the current policy.

---

# 208. Why the Gradient Is Difficult

To improve the policy, we need:

`∇θJ(θ)`

A direct approach is difficult because:

- actions may be discrete,
- the environment may be unknown,
- the environment may be non-differentiable,
- rewards may come from humans,
- the number of possible trajectories may be enormous.

Suppose a human rates an LLM answer as:

```text
8 out of 10
```

We cannot backpropagate through the human evaluator.

The key idea is:

> Do not differentiate through the environment. Differentiate the probability of choosing the actions that produced the reward.

This is the intuition behind policy gradients.

---

# 209. Policy Gradient Intuition

Suppose the robot chooses:

```text
Turn Right
```

and later receives a large reward.

The learning rule should make:

```text
Turn Right
```

more likely in similar states.

Suppose it chooses:

```text
Turn Left
```

and later receives a poor return.

The learning rule should make:

```text
Turn Left
```

less likely in similar states.

```text
High Return

↓

Reinforce the Chosen Actions
```

```text
Low Return

↓

Discourage the Chosen Actions
```

This is why the simplest policy-gradient algorithm is called **REINFORCE**.

---

# 210. The Gradient of Log Probability

Consider:

`∇θ log πθ(Aₜ | Sₜ)`

This gradient points in the direction that increases the log probability of the action that was actually selected.

If it is multiplied by a positive return:

```text
Positive Return

×

∇θ log πθ(Aₜ|Sₜ)

↓

Selected Action Becomes More Likely
```

If it is multiplied by a negative return, the update moves in the opposite direction.

The selected action becomes less likely.

---

## Two-Action Example

Suppose:

| Action | Probability |
|---|---:|
| Move toward Room B | 0.40 |
| Move away from Room B | 0.60 |

The robot samples:

```text
Move toward Room B
```

and receives a large positive return.

After learning, the probabilities may move toward:

| Action | New probability |
|---|---:|
| Move toward Room B | 0.55 |
| Move away from Room B | 0.45 |

The exact numbers depend on the model and learning rate.

The direction is what matters.

---

# 211. Write the Objective as a Sum

For a finite trajectory space:

`J(θ) = Στ pθ(τ) G(τ)`

Differentiate:

`∇θJ(θ) = Στ ∇θpθ(τ) G(τ)`

The return is the observed score of the trajectory.

The difficult term is:

`∇θpθ(τ)`

We need to rewrite it in a form that can be estimated from samples.

---

# 212. The Log-Derivative Trick

For any positive differentiable function `fθ(x)`:

`∇θ log fθ(x) = ∇θfθ(x) / fθ(x)`

Rearrange:

`∇θfθ(x) = fθ(x) ∇θ log fθ(x)`

Apply it to the trajectory probability:

`∇θpθ(τ) = pθ(τ) ∇θ log pθ(τ)`

This identity is called:

- the log-derivative trick,
- the likelihood-ratio trick,
- the score-function trick.

---

## Why the Trick Helps

Before the trick:

`∇θpθ(τ)`

After the trick:

`pθ(τ) ∇θ log pθ(τ)`

An expression of the form:

`Στ pθ(τ) × something`

is an expectation.

Expectations can be estimated using sample averages.

The trick converts a difficult derivative into a sample-friendly form.

---

# 213. Substitute the Log-Derivative Trick

Start with:

`∇θJ(θ) = Στ ∇θpθ(τ) G(τ)`

Substitute:

`∇θpθ(τ) = pθ(τ) ∇θ log pθ(τ)`

Then:

`∇θJ(θ) = Στ pθ(τ) ∇θ log pθ(τ) G(τ)`

Recognize the expectation:

`∇θJ(θ) = Eτ~pθ[∇θ log pθ(τ) G(τ)]`

We have converted the objective gradient into an expectation that can be estimated from sampled trajectories.

---

# 214. Expand the Log Trajectory Probability

Recall:

`pθ(τ) = ρ₀(S₀) × ∏ₜ πθ(Aₜ|Sₜ) P(Sₜ₊₁|Sₜ,Aₜ)`

Take the logarithm:

`log pθ(τ) = log ρ₀(S₀) + Σₜ log πθ(Aₜ|Sₜ) + Σₜ log P(Sₜ₊₁|Sₜ,Aₜ)`

The logarithm converts products into sums.

Now differentiate with respect to `θ`.

---

# 215. Why the Environment Terms Disappear

The initial-state distribution does not depend on the policy parameters:

`∇θ log ρ₀(S₀) = 0`

The transition distribution also does not depend on the policy parameters:

`∇θ log P(Sₜ₊₁|Sₜ,Aₜ) = 0`

Only the policy depends on `θ`.

Therefore:

`∇θ log pθ(τ) = Σₜ ∇θ log πθ(Aₜ|Sₜ)`

This is the crucial simplification.

---

## Why This Is Powerful

We do not need:

- a differentiable environment,
- the mathematical transition formula,
- gradients through a human evaluator,
- gradients through the reward function.

We need only:

- sampled states,
- sampled actions,
- observed rewards,
- the policy's log probability of each sampled action.

This is why policy gradients can work with black-box environments.

---

# 216. The Policy Gradient Theorem

Substitute the policy-only gradient:

`∇θJ(θ) = Eτ~pθ[(Σₜ ∇θ log πθ(Aₜ|Sₜ)) G(τ)]`

A more useful credit-assignment form uses the future return from each time step:

`∇θJ(θ) = Eτ~pθ[Σₜ ∇θ log πθ(Aₜ|Sₜ) Gₜ]`

This is the central policy-gradient expression used by REINFORCE-style methods.

---

# 217. Return-to-Go and Credit Assignment

An action taken at time `t` cannot influence rewards that happened before time `t`.

Therefore, when evaluating action `Aₜ`, we use rewards from time `t` onward.

```text
Past Rewards

↓

Cannot Be Caused by the Current Action
```

```text
Future Rewards

↓

May Be Influenced by the Current Action
```

This is why the term `Gₜ` is used.

It provides a more sensible learning signal for action `Aₜ`.

---

# 218. Reading One Policy-Gradient Term

Consider:

`∇θ log πθ(Aₜ|Sₜ) × Gₜ`

It contains two ideas.

## What action did the policy choose?

`∇θ log πθ(Aₜ|Sₜ)`

## How good was the future outcome?

`Gₜ`

The return scales the update.

If:

`Gₜ > 0`

the selected action becomes more likely.

If:

`Gₜ < 0`

the selected action becomes less likely.

If:

`Gₜ ≈ 0`

the update is small.

This is the mathematical form of rewarding good behavior and discouraging poor behavior.

---

# 219. Monte Carlo Approximation

The policy-gradient theorem still contains an expectation.

We normally cannot calculate it exactly.

Suppose we collect `N` trajectories.

Then:

`∇θJ(θ) ≈ (1/N) Σᵢ Σₜ ∇θ log πθ(Aₜ⁽ⁱ⁾|Sₜ⁽ⁱ⁾) Gₜ⁽ⁱ⁾`

This is a Monte Carlo estimate.

It follows from the Law of Large Numbers:

```text
More Independent Samples

↓

Sample Average Approaches True Expectation
```

More trajectories generally improve the estimate but increase computation.

---

# 220. REINFORCE

REINFORCE is the direct sample-based implementation of the policy-gradient estimator.

```text
1. Run the current policy.

2. Collect trajectories.

3. Record actions and rewards.

4. Compute future returns.

5. Compute log probabilities of selected actions.

6. Multiply each log probability by its return.

7. Update the policy.
```

Deep-learning libraries normally minimize a loss.

Therefore, we use:

`L_REINFORCE(θ) = −E[Σₜ log πθ(Aₜ|Sₜ) Gₜ]`

The minus sign converts gradient ascent into ordinary gradient descent.

---

# 221. Worked Example and Minimal Pseudocode

Suppose the robot generates:

| Time | State | Action | Reward |
|---|---|---|---:|
| 0 | Reception | Forward | −0.1 |
| 1 | Corridor | Right | −0.1 |
| 2 | Room B | Deliver | +20 |

With `γ = 1`:

`G₂ = 20`

`G₁ = −0.1 + 20 = 19.9`

`G₀ = −0.1 − 0.1 + 20 = 19.8`

All three selected actions are followed by positive future return.

REINFORCE increases their probabilities in similar states.

If the robot had hit a wall and received a large negative return, the selected actions would be discouraged.

---

## Minimal Pseudocode

```python
for iteration in range(num_iterations):
    trajectories = collect_trajectories(policy)

    loss = 0.0

    for trajectory in trajectories:
        returns = discounted_returns(
            rewards=trajectory.rewards,
            gamma=gamma,
        )

        for log_prob, return_t in zip(
            trajectory.log_probs,
            returns,
        ):
            loss = loss - log_prob * return_t

    loss = loss / len(trajectories)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Only the policy log probabilities need to be differentiable.

The environment can remain a black box.

---

# 222. Why Plain REINFORCE Is Not Enough

REINFORCE is elegant, but its gradient estimate can have high variance.

The same policy may produce very different returns:

```text
Trajectory 1 → Return +20

Trajectory 2 → Return −5

Trajectory 3 → Return +8

Trajectory 4 → Return −2
```

The resulting gradients may point in very different directions.

```text
One Batch

↓

Large Positive Update
```

```text
Next Batch

↓

Large Negative Update
```

Consequences include:

- unstable gradients,
- slow learning,
- sensitivity to random samples,
- large differences between updates.

The next chapter introduces:

- baselines,
- value functions,
- Q functions,
- advantage functions,
- actor–critic intuition.

These ideas reduce variance and prepare the path toward PPO-style alignment.

---

## Connection to LLM Alignment

For an autoregressive LLM:

```text
Agent

=

Language Model
```

```text
State

=

Prompt + Generated Prefix
```

```text
Action

=

Next Token
```

```text
Policy

=

Next-Token Probability Distribution
```

```text
Trajectory

=

Complete Generated Response
```

```text
Reward

=

Quality Score for the Response
```

The Transformer already provides:

`πθ(Aₜ|Sₜ)`

because its Softmax output is a probability distribution over the next token.

This is why an autoregressive LLM can naturally be treated as a policy.

The detailed mapping is developed in Chapter 17.

---

# Common Misconceptions

## Misconception 1

**The policy is part of the MDP.**

It is not.

The MDP describes the environment's rules.

The policy describes the agent's strategy.

---

## Misconception 2

**A stochastic policy performs several actions at once.**

It does not.

It assigns probabilities to actions and then selects one action.

---

## Misconception 3

**The reward tells the agent the correct action.**

Usually it does not.

It evaluates the outcome.

The agent must infer which actions led to that outcome.

---

## Misconception 4

**Markov means the past is ignored.**

It does not.

The current state should summarize all relevant information from the past.

---

## Misconception 5

**Policy gradient differentiates through the environment.**

It does not require that.

It differentiates the log probability of selected actions.

---

## Misconception 6

**One successful trajectory proves that a policy is good.**

It does not.

A stochastic trajectory may be lucky.

Policy quality is measured using expected return over many trajectories.

---

## Misconception 7

**A high reward guarantees every action in the trajectory was good.**

Not necessarily.

Some actions may have been unnecessary or harmful even if the final outcome was successful.

This is part of the credit-assignment problem.

---

# Chapter Summary

In this chapter, we built the foundations of policy-based Reinforcement Learning.

We learned that:

- Reinforcement Learning is useful when behavior is evaluated through feedback rather than exact target labels.
- The agent chooses actions and the environment produces next states and rewards.
- A state describes the current situation.
- An action is a decision made by the agent.
- A reward evaluates the consequence of an action.
- A policy is a probability distribution over actions given a state.
- An MDP describes the rules of the world, while the policy describes the agent's strategy.
- A trajectory is a complete sequence of states, actions and rewards.
- A return summarizes future rewards.
- The objective is to maximize expected return.
- The log-derivative trick rewrites the gradient into a sample-friendly expectation.
- Environment terms disappear because they do not depend on the policy parameters.
- The Policy Gradient Theorem reinforces actions associated with high return.
- REINFORCE estimates the gradient using sampled trajectories.
- Plain REINFORCE has high variance, motivating value and advantage functions.

---

# Key Takeaways

✅ Reinforcement Learning learns from consequences rather than exact answer keys.

✅ `πθ(a|s)` represents the agent's behavior.

✅ `P(s'|s,a)` represents the environment's response.

✅ A trajectory is the data generated by agent–environment interaction.

✅ The Reinforcement Learning objective is:

`J(θ) = Eτ~pθ(τ)[G₀]`

✅ The log-derivative trick is:

`∇θpθ(τ) = pθ(τ)∇θ log pθ(τ)`

✅ The trajectory-form policy gradient is:

`∇θJ(θ) = E[Σₜ ∇θ log πθ(Aₜ|Sₜ) Gₜ]`

✅ High-return actions become more likely.

✅ Low-return actions become less likely.

✅ REINFORCE is mathematically simple but often has high variance.

---

# Looking Ahead

The next chapter is:

```text
17_Value_Q_and_Advantage_Functions_for_LLM_Alignment.md
```

It explains how to make policy-gradient learning more stable using:

- baselines,
- value functions,
- Q functions,
- advantage functions,
- actor–critic intuition,
- the exact mapping between an autoregressive LLM and an RL policy.
