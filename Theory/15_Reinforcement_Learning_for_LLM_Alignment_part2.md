# 39. Why Do We Need a Mathematical Framework?

In Part 1, we described Reinforcement Learning intuitively.

```text
Agent

↓

Chooses an Action

↓

Environment Responds

↓

Agent Receives a Reward
```

This explanation is useful, but it is not precise enough to build an algorithm.

We need to specify:

- what situations the agent may encounter,
- what actions it may take,
- how the environment changes,
- how rewards are produced,
- how the agent chooses actions.

The mathematical framework used to represent this interaction is called a:

**Markov Decision Process**

or simply:

**MDP**

---

# 40. What Is a Markov Decision Process?

A Markov Decision Process describes a system in which:

1. An agent is currently in some state.
2. The agent selects an action.
3. The environment produces a reward.
4. The environment moves the agent to a new state.
5. The process repeats.

Conceptually:

```text
Current State Sₜ

↓

Agent Chooses Action Aₜ

↓

Environment Produces Reward Rₜ

↓

Environment Produces Next State Sₜ₊₁

↓

Repeat
```

The MDP provides the mathematical rules governing this interaction.

---

# 41. Components of an MDP

An MDP can be represented using the tuple:

`M = (S, A, P, R, ρ₀, γ)`

where:

- `S` is the state space.
- `A` is the action space.
- `P` is the transition probability.
- `R` is the reward function.
- `ρ₀` is the initial-state distribution.
- `γ` is the discount factor.

We will understand each component separately.

---

# 42. The State Space

The **state space** contains every possible situation in which the agent may find itself.

It is written as:

`S = {s₁, s₂, ..., sM}`

A particular state at time `t` is written as:

`Sₜ`

---

## Example: Video Game

A state may include:

- the player's position,
- the locations of enemies,
- the current score,
- the remaining lives,
- the objects visible on the screen.

Conceptually:

```text
State

=

Complete Description of the Current Game Situation
```

---

# 43. A State Can Be a Vector

A state does not need to be one number.

For a self-driving car, the state might be:

`Sₜ = [speed, position, traffic-light status, nearby vehicles, road condition]`

For example:

```text
Sₜ = [40 km/h, lane 2, green light, car ahead, dry road]
```

Therefore, a state is often represented as a vector containing several measurements.

---

# 44. What Should a State Contain?

A useful state should contain enough information for the agent to make its next decision.

Suppose a robot must decide whether to move forward.

Knowing only its current position may not be sufficient.

It may also need:

- the locations of obstacles,
- its direction,
- its remaining battery,
- its destination.

The state should summarize the information relevant to the next decision.

---

# 45. The Action Space

The **action space** contains every action that the agent is allowed to perform.

It is written as:

`A = {a₁, a₂, ..., aK}`

A particular action taken at time `t` is written as:

`Aₜ`

---

## Example: Robot Navigation

The action space may be:

```text
Move Left

Move Right

Move Forward

Move Backward

Stop
```

Mathematically:

`A = {left, right, forward, backward, stop}`

---

# 46. Available Actions May Depend on the State

Not every action must be valid in every state.

For example, in a chess game:

- a pawn cannot move like a bishop,
- a piece cannot move outside the board,
- some moves may place the king in danger.

Therefore, the set of valid actions may depend on the current state.

Conceptually:

`ValidActions = A(Sₜ)`

For introductory analysis, we often describe one overall action space and assume invalid actions are excluded or masked.

---

# 47. Does the Agent Perform Every Possible Action?

No.

The policy may assign probabilities to several actions, but the agent chooses only one action at a given time step.

Example:

| Action | Policy probability |
|---|---:|
| Move left | 0.10 |
| Move right | 0.60 |
| Move forward | 0.25 |
| Stop | 0.05 |

The agent samples or selects one action from this distribution.

It does not perform all four actions simultaneously.

---

# 48. The Policy

The **policy** describes the agent's behavior.

It gives a probability distribution over actions for a given state.

`πθ(a | s) = probability of choosing action a while in state s`

The parameters of the policy are represented by `θ`.

In deep Reinforcement Learning, the policy is frequently represented by a neural network.

---

# 49. Simple Policy Example

Suppose a robot detects an obstacle directly ahead.

Its policy may produce:

| Action | Probability |
|---|---:|
| Move forward | 0.02 |
| Move left | 0.43 |
| Move right | 0.45 |
| Stop | 0.10 |

The policy expresses what the agent is likely to do in that state.

---

# 50. Deterministic Policy

A deterministic policy always chooses one particular action for a given state.

It can be written as:

`Aₜ = μθ(Sₜ)`

Example:

```text
If obstacle ahead:

Always turn right.
```

The same state always produces the same action.

---

# 51. Stochastic Policy

A stochastic policy produces probabilities over possible actions.

It is written as:

`Aₜ ~ πθ(Aₜ | Sₜ)`

Example:

```text
If obstacle ahead:

Turn left  → 45%

Turn right → 45%

Stop       → 10%
```

The action may differ across repeated visits to the same state.

Language models naturally behave like stochastic policies because they produce probability distributions over tokens.

---

# 52. The Transition Probability

After the agent performs an action, the environment moves to a new state.

This behavior is described using the **transition probability**:

`P(Sₜ₊₁ = s' | Sₜ = s, Aₜ = a)`

Read this as:

> Given that the agent is in state `s` and performs action `a`, what is the probability that the next state will be `s'`?

The transition probability is also called:

- transition model,
- transition kernel,
- environment dynamics.

---

# 53. Deterministic Transition Example

Consider chess.

Suppose the current board configuration is `Sₜ`.

The agent chooses a legal move `Aₜ`.

The next board configuration is determined by the rules of chess.

Conceptually:

```text
Current Board

+

Chosen Move

↓

One Specific Next Board
```

In this case, the transition is mostly deterministic.

The chosen move determines the next state.

---

# 54. Stochastic Transition Example

Imagine a robot moving on a slippery floor.

The agent chooses:

```text
Move Right
```

However, the possible results are:

| Next state | Probability |
|---|---:|
| Move right successfully | 0.80 |
| Slip upward | 0.10 |
| Slip downward | 0.10 |

The agent controls the action.

It does not completely control the result.

This uncertainty is represented by:

`P(Sₜ₊₁ | Sₜ, Aₜ)`

---

# 55. Policy Versus Transition Probability

This distinction is extremely important.

## Policy

The policy describes the agent's choice.

`πθ(Aₜ | Sₜ)`

It answers:

> What action will the agent choose in the current state?

---

## Transition Probability

The transition probability describes the environment's response.

`P(Sₜ₊₁ | Sₜ, Aₜ)`

It answers:

> After the agent chooses an action, what state will the environment produce?

---

## Simple Analogy

Suppose you are at home.

Your possible actions are:

- go to work,
- go shopping,
- stay at home.

Your decision rule is your **policy**.

Suppose you choose to go to work.

Traffic, weather and road conditions determine how the journey unfolds.

Those external consequences belong to the **environment transition**.

```text
Your Decision

=

Policy
```

```text
What the World Does After Your Decision

=

Transition
```

---

# 56. Which Part Is Learned?

In the formulation used in this chapter, the policy is parameterized by `θ`.

`πθ(Aₜ | Sₜ)`

The goal is to learn good values of `θ`.

The transition probability is treated as part of the environment:

`P(Sₜ₊₁ | Sₜ, Aₜ)`

It is not controlled by the agent.

Some Reinforcement Learning methods also learn a model of the environment, but that is not required for the present discussion.

---

# 57. The Reward Function

After observing the state and action, the environment produces a reward.

A simple reward function is:

`Rₜ = R(Sₜ, Aₜ)`

The reward is a real number.

It may be:

- positive,
- zero,
- negative.

---

## Example: Navigation

```text
Reach the destination → +10

Move one step         → −0.1

Hit an obstacle       → −5
```

The reward communicates which outcomes are desirable.

---

# 58. Reward Is Not the Same as the Action

The agent chooses the action.

The environment provides the reward.

```text
Agent

↓

Chooses Action

↓

Environment

↓

Produces Reward
```

The reward is feedback about the action's consequence.

The agent does not normally choose its own reward.

---

# 59. Immediate Reward May Be Misleading

Suppose a robot must travel through a long corridor to reach a charging station.

For many steps, it receives no reward.

Only at the final step does it receive:

```text
+100
```

An early action may therefore be important even if it produces no immediate reward.

This is called the **delayed reward problem**.

Reinforcement Learning must learn which earlier actions contributed to the later success.

---

# 60. The Initial-State Distribution

The system must begin somewhere.

The distribution over possible starting states is written as:

`S₀ ~ ρ₀(S₀)`

where `ρ₀` is the initial-state distribution.

---

## Example: Game

A game may always begin from the same state.

Then:

```text
Probability of standard starting screen = 1
```

Alternatively, a training environment may begin from several different levels or positions.

Then `ρ₀` defines how frequently each starting state is selected.

---

# 61. Initial State in an LLM

For a language model, the initial state contains the user's prompt.

Example:

```text
Explain gradient descent using a real-life analogy.
```

During alignment training, prompts may be sampled from a dataset.

Therefore, the prompt dataset induces an initial-state distribution.

Conceptually:

`S₀ ~ PromptDistribution`

---

# 62. The Discount Factor

The MDP also includes a discount factor:

`0 ≤ γ < 1`

The discount factor determines how strongly future rewards are weighted.

A reward received far in the future is multiplied by a larger power of `γ`.

For now, remember:

```text
γ close to 0

↓

Focus mainly on immediate rewards
```

```text
γ close to 1

↓

Give importance to long-term rewards
```

We will derive the discounted return in Part 3.

---

# 63. The Markov Property

The word **Markov** means that the next state depends only on:

- the current state,
- the current action.

Once these are known, the complete earlier history is not additionally required.

The Markov property is:

`P(Sₜ₊₁ | S₀,A₀,...,Sₜ,Aₜ) = P(Sₜ₊₁ | Sₜ,Aₜ)`

---

# 64. Markov Property Intuition

Suppose the state of a chess game contains the complete current board configuration and all rule-relevant information.

To determine the next board after a move, we do not need to replay the entire game from the beginning.

We need only:

```text
Current Board

+

Current Move
```

The current state already summarizes the relevant past.

---

# 65. Markov Does Not Mean the Past Is Unimportant

The Markov property does **not** mean that earlier events never matter.

It means that their relevant effects must already be represented in the current state.

For example, in an LLM:

```text
Current State

=

Prompt + All Tokens Generated So Far
```

The earlier tokens still matter.

They are included inside the current state.

Therefore, the next-token decision can still depend on the complete generated prefix while satisfying the Markov property.

---

# 66. One Complete Interaction Step

One Reinforcement Learning step can now be written as:

```text
1. Agent observes Sₜ

2. Agent samples Aₜ ~ πθ(Aₜ | Sₜ)

3. Environment produces Rₜ = R(Sₜ, Aₜ)

4. Environment samples Sₜ₊₁ ~ P(Sₜ₊₁ | Sₜ, Aₜ)

5. Repeat
```

This is the fundamental MDP interaction loop.

---

# 67. What Is a Trajectory?

A **trajectory** is one complete sequence of states and actions experienced by the agent.

It is commonly written as:

`τ = (S₀, A₀, R₀, S₁, A₁, R₁, ..., S_T)`

The Greek letter `τ`, pronounced **tau**, represents the trajectory.

---

# 68. Real-Life Trajectory Example

Consider a robot learning to reach a charging station.

```text
State 0:

Robot is in Room A

Action 0:

Move right

Reward 0:

−0.1

State 1:

Robot enters the corridor

Action 1:

Move forward

Reward 1:

−0.1

State 2:

Robot reaches charging station

Reward 2:

+10
```

The complete sequence is one trajectory.

---

# 69. Trajectory Versus State

A state is one snapshot.

A trajectory is the entire experience.

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

This is an important distinction.

---

# 70. Episode Versus Trajectory

An **episode** is an interaction that has a beginning and an ending.

Examples include:

- one complete game,
- one robot-navigation attempt,
- one generated LLM response.

The sequence recorded during that episode is its trajectory.

In many introductory discussions, the words episode and trajectory are used closely together.

---

# 71. Data in Reinforcement Learning

In supervised learning, data may look like:

`(input, label)`

In Reinforcement Learning, data consists of trajectories.

```text
Trajectory 1

Trajectory 2

Trajectory 3

...

Trajectory N
```

The agent interacts with the environment multiple times and records what happened.

These trajectories are then used to improve the policy.

---

# 72. Different Policies Produce Different Trajectories

Suppose one robot policy frequently moves toward the destination.

It will generate trajectories that often reach the goal.

Another policy may move randomly.

It will generate very different trajectories.

Therefore, the probability of observing a trajectory depends on the policy:

`Pθ(τ)`

The subscript `θ` appears because the policy is parameterized by `θ`.

---

# 73. Probability of a Trajectory

The probability of a trajectory is:

`Pθ(τ) = ρ₀(S₀) × ∏ₜ₌₀ᵀ⁻¹ πθ(Aₜ | Sₜ) × P(Sₜ₊₁ | Sₜ, Aₜ)`

This is one of the most important equations in Reinforcement Learning.

It combines:

1. The probability of the initial state.
2. The probability of each selected action.
3. The probability of each state transition.

---

# 74. Understanding the Trajectory Equation

The first term is:

`ρ₀(S₀)`

This tells us how likely the starting state is.

At each time step, two things happen.

The agent selects an action:

`πθ(Aₜ | Sₜ)`

The environment produces the next state:

`P(Sₜ₊₁ | Sₜ, Aₜ)`

Multiplying these probabilities across time gives the probability of the complete trajectory.

---

# 75. Why Does the Product Appear?

A trajectory consists of a sequence of events.

```text
Start in S₀

↓

Choose A₀

↓

Move to S₁

↓

Choose A₁

↓

Move to S₂

↓

...
```

The Chain Rule of Probability tells us to multiply the relevant conditional probabilities.

The Markov property simplifies the transition terms so that the next state depends only on the current state and action.

---

# 76. Small Numerical Example

Suppose:

`ρ₀(S₀) = 1`

The policy selects the first action with probability:

`πθ(A₀ | S₀) = 0.7`

The environment reaches the desired next state with probability:

`P(S₁ | S₀,A₀) = 0.9`

At the second state:

`πθ(A₁ | S₁) = 0.8`

The final transition has probability:

`P(S₂ | S₁,A₁) = 0.95`

The trajectory probability is:

`Pθ(τ) = 1 × 0.7 × 0.9 × 0.8 × 0.95 = 0.4788`

This means that this exact sequence of states and actions occurs with probability `0.4788` under the policy and environment.

---

# 77. Why Is Reward Missing from the Trajectory Probability?

In our simplified formulation, the reward is determined by the state and action:

`Rₜ = R(Sₜ,Aₜ)`

Therefore, once the state and action are known, the reward is already determined.

It does not require a separate probability term.

If rewards were stochastic, we could additionally include a term such as:

`P(Rₜ | Sₜ,Aₜ)`

For this chapter, we use the simpler reward-function formulation.

---

# 78. Policy Parameters Affect the Trajectory Distribution

The environment transition is:

`P(Sₜ₊₁ | Sₜ,Aₜ)`

The agent does not directly control it.

However, changing `θ` changes:

`πθ(Aₜ | Sₜ)`

Changing the action probabilities changes which trajectories become likely.

Therefore:

```text
Change Policy Parameters θ

↓

Change Action Choices

↓

Change Trajectory Distribution

↓

Change Rewards Obtained
```

This is how policy learning influences the agent's behavior.

---

# 79. LLM Response as a Trajectory

An autoregressive language model can also be viewed as producing a trajectory.

Suppose the prompt is:

```text
Machine learning is
```

The initial state is:

`S₀ = "Machine learning is"`

The model chooses the action:

`A₀ = "useful"`

The new state becomes:

`S₁ = "Machine learning is useful"`

The model then chooses:

`A₁ = "because"`

The new state becomes:

`S₂ = "Machine learning is useful because"`

This continues until an end-of-sequence token is generated.

---

# 80. LLM Transition Is Usually Deterministic

At token level, after the language model chooses token `Aₜ`, the next state is formed by appending that token.

`Sₜ₊₁ = Append(Sₜ,Aₜ)`

For example:

```text
Current State:

Deep learning is

Action:

powerful

Next State:

Deep learning is powerful
```

The policy is uncertain because it produces probabilities over tokens.

The state update itself is usually deterministic.

---

# 81. Complete LLM Mapping

| Reinforcement Learning concept | Language-model interpretation |
|---|---|
| Agent | Language model |
| Initial state | User prompt |
| State | Prompt plus generated tokens |
| Action | Next token |
| Policy | Next-token probability distribution |
| Transition | Append selected token |
| Trajectory | Complete generated response |
| Reward | Evaluation of response quality |
| Terminal state | End-of-sequence or stopping condition |

This correspondence allows language-model generation to be treated as a Reinforcement Learning problem.

---

# 82. Common Misconceptions

## Misconception 1

**The policy and transition probability are the same thing.**

They are different.

The policy describes the agent's decision:

`πθ(Aₜ | Sₜ)`

The transition describes the environment's response:

`P(Sₜ₊₁ | Sₜ,Aₜ)`

---

## Misconception 2

**A stochastic policy performs several actions at once.**

It does not.

It defines probabilities over actions and then selects one action.

---

## Misconception 3

**Markov means that the past is ignored.**

It does not.

The current state should summarize all relevant information from the past.

---

## Misconception 4

**Reward tells the agent the correct action.**

Not necessarily.

Reward evaluates the consequence of an action.

The agent must learn which behavior produces larger rewards.

---

## Misconception 5

**A trajectory is one state-action pair.**

It is not.

A trajectory is the complete sequence of states, actions and rewards.

---

## Misconception 6

**The agent controls the next state completely.**

Not always.

In stochastic environments, the same state and action may lead to different next states.

---

# Key Insights

- An MDP formally represents agent–environment interaction.
- The state describes the agent's current situation.
- The action is the decision made by the agent.
- The policy specifies action probabilities given the state.
- The transition probability describes how the environment changes.
- The reward evaluates the consequence of an action.
- The initial-state distribution determines how episodes begin.
- The Markov property says the next state depends on the current state and action.
- A trajectory is a complete sequence of states and actions.
- Reinforcement Learning data consists of multiple trajectories.
- The probability of a trajectory depends on both the policy and environment.
- An LLM response can be interpreted as a trajectory of next-token actions.

---

# What's Next?

We now know how a policy generates trajectories.

The next question is:

> How do we determine whether one policy is better than another?

In Part 3, we will introduce:

- cumulative reward,
- discounted return,
- expected return,
- the Reinforcement Learning objective,
- policy optimization,
- the Policy Gradient idea.

The central goal will become:

`Find θ that maximizes the expected discounted return.`
