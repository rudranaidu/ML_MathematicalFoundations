# 83. What Makes One Policy Better Than Another?

In Part 2, we learned that a policy determines how an agent chooses actions.

`πθ(a | s)`

Different policies produce different trajectories.

For example, consider a robot trying to reach a charging station.

One policy may repeatedly move toward the charger.

Another policy may move randomly.

Both policies are valid, but one is clearly more useful.

Therefore, we need a mathematical way to answer:

> Which policy produces better long-term behavior?

The answer is based on the rewards collected during a trajectory.

---

# 84. Immediate Reward

At every time step, the environment may provide a reward.

`Rₜ = R(Sₜ, Aₜ)`

An immediate reward evaluates the consequence of one action.

Example:

```text
Robot moves closer to charger  → +1

Robot hits an obstacle         → −5

Robot reaches charger          → +20
```

However, evaluating only the immediate reward is not enough.

An action may be useful even if it does not produce an immediate positive reward.

---

# 85. Why Immediate Reward Is Not Enough

Imagine that you are travelling to attend an important exam.

At the beginning, you:

- wake up early,
- take a bus,
- walk to the examination hall,
- write the exam.

The final reward may be:

```text
Successfully completing the exam
```

The earlier actions did not directly produce that reward.

However, they were necessary.

Similarly, in Reinforcement Learning, an action may contribute to a reward received much later.

Therefore, the policy must consider the total reward collected over the entire trajectory.

---

# 86. Cumulative Reward

The simplest way to evaluate a trajectory is to add all its rewards.

`TotalReward(τ) = R₀ + R₁ + R₂ + ... + R_T`

Suppose one trajectory produces:

```text
R₀ = −1

R₁ = −1

R₂ = +10
```

Then:

`TotalReward(τ) = −1 − 1 + 10 = 8`

A trajectory with a larger cumulative reward is generally considered better.

---

# 87. Why Future Rewards Are Often Discounted

In many problems, a reward received immediately is considered more valuable than the same reward received much later.

For example:

```text
Receive ₹100 today
```

may be preferred over:

```text
Receive ₹100 after ten years
```

There may also be uncertainty about whether a distant reward will ever be reached.

To represent this preference, Reinforcement Learning introduces a **discount factor**.

---

# 88. The Discount Factor

The discount factor is represented by:

`γ`

with:

`0 ≤ γ ≤ 1`

Future rewards are multiplied by powers of `γ`.

The farther a reward is in the future, the more times it is discounted.

---

# 89. Discounted Return

The **discounted return** of a trajectory is:

`G(τ) = R₀ + γR₁ + γ²R₂ + γ³R₃ + ...`

For a finite trajectory:

`G(τ) = Σₜ₌₀ᵀ γᵗRₜ`

This single number summarizes the quality of the complete trajectory.

---

# 90. Understanding the Powers of Gamma

The immediate reward receives weight:

`γ⁰ = 1`

The next reward receives weight:

`γ¹`

The reward after two steps receives weight:

`γ²`

Therefore:

```text
Immediate Reward    → Weight 1

One-Step Future     → Weight γ

Two-Step Future     → Weight γ²

Three-Step Future   → Weight γ³
```

If `γ < 1`, rewards farther in the future receive smaller weights.

---

# 91. Numerical Example

Suppose:

`γ = 0.9`

and the trajectory rewards are:

```text
R₀ = 2

R₁ = 3

R₂ = 10
```

The discounted return is:

`G(τ) = 2 + 0.9 × 3 + 0.9² × 10`

Therefore:

`G(τ) = 2 + 2.7 + 8.1 = 12.8`

Without discounting, the total reward would be:

`2 + 3 + 10 = 15`

The distant reward contributes slightly less under discounting.

---

# 92. What Happens When Gamma Is Zero?

If:

`γ = 0`

then:

`G(τ) = R₀`

The agent considers only the immediate reward.

It completely ignores future rewards.

This creates a very short-sighted agent.

---

# 93. What Happens When Gamma Is Close to One?

If:

`γ ≈ 1`

then future rewards retain most of their importance.

For example, with:

`γ = 0.99`

a reward received several steps later still contributes significantly.

This encourages long-term planning.

---

# 94. Gamma Intuition

Think of `γ` as the agent's level of patience.

```text
Small γ

↓

Impatient Agent

↓

Focuses on Immediate Rewards
```

```text
Large γ

↓

Patient Agent

↓

Considers Long-Term Consequences
```

The appropriate value depends on the problem.

---

# 95. Why Discounting Is Mathematically Useful

Discounting also helps when trajectories may continue for a very long time.

Without discounting, an infinite sum of positive rewards could grow without bound.

When:

`0 ≤ γ < 1`

the powers:

`1, γ, γ², γ³, ...`

become progressively smaller.

This can keep the return mathematically well behaved.

---

# 96. Return From a Particular Time Step

Sometimes we want to calculate the return beginning at time `t`.

This is called the **return-to-go**.

`Gₜ = Rₜ + γRₜ₊₁ + γ²Rₜ₊₂ + ...`

It measures the future reward obtained after reaching state `Sₜ`.

Actions taken at time `t` cannot change rewards that occurred before time `t`.

Therefore, future rewards are especially relevant when evaluating that action.

---

# 97. One Trajectory Is Not Enough

Suppose we run the same stochastic policy twice.

The first run may succeed.

The second run may fail.

For example:

```text
Trajectory 1 Return = +20

Trajectory 2 Return = −5

Trajectory 3 Return = +8
```

Which number represents the quality of the policy?

We cannot judge a stochastic policy using only one trajectory.

We must consider its average behavior across many possible trajectories.

---

# 98. Expected Return

The quality of a policy is measured using the **expected return**.

`J(θ) = Eτ~Pθ(τ)[G(τ)]`

Read this as:

> The average discounted return over trajectories generated by policy `πθ`.

Here:

- `θ` represents the policy parameters.
- `Pθ(τ)` is the trajectory distribution.
- `G(τ)` is the return of a trajectory.
- `J(θ)` is the objective value of the policy.

---

# 99. Expected Return Intuition

Imagine evaluating a student using only one examination.

The student may have:

- had an unusually good day,
- received familiar questions,
- made accidental mistakes.

A more reliable estimate comes from averaging performance over many examinations.

Similarly:

```text
One Trajectory

↓

May Be Lucky or Unlucky
```

```text
Average Over Many Trajectories

↓

Better Estimate of Policy Quality
```

---

# 100. Small Expected-Return Example

Suppose a policy can generate three possible trajectories.

| Trajectory | Probability | Return |
|---|---:|---:|
| `τ₁` | 0.5 | 10 |
| `τ₂` | 0.3 | 5 |
| `τ₃` | 0.2 | −2 |

The expected return is:

`J(θ) = 0.5 × 10 + 0.3 × 5 + 0.2 × (−2)`

Therefore:

`J(θ) = 5 + 1.5 − 0.4 = 6.1`

The expected return of this policy is `6.1`.

---

# 101. Defining a Good Policy

We can now define what Prathosh referred to as a **good policy**.

A good policy is one that produces a large expected discounted return.

Therefore, the Reinforcement Learning objective is:

`θ* = argmaxθ J(θ)`

Equivalently:

`θ* = argmaxθ Eτ~Pθ(τ)[G(τ)]`

This is the central optimization problem in Reinforcement Learning.

---

# 102. Understanding the Objective

The objective does not say:

> Find the action with the highest immediate reward.

It says:

> Find policy parameters that produce trajectories with high average long-term reward.

This distinction is extremely important.

The policy must consider the consequences of an entire sequence of actions.

---

# 103. How Does Theta Affect the Return?

The reward function may not directly contain `θ`.

However, the policy depends on `θ`.

`πθ(Aₜ | Sₜ)`

Changing `θ` changes the action probabilities.

Changing the actions changes the trajectories.

Changing the trajectories changes the rewards collected.

Conceptually:

```text
Change θ

↓

Change Policy

↓

Change Action Probabilities

↓

Change Trajectories

↓

Change Expected Return
```

Therefore, the expected return is indirectly a function of `θ`.

---

# 104. Connection to the Trajectory Distribution

Recall the trajectory probability:

`Pθ(τ) = ρ₀(S₀) × ∏ₜ πθ(Aₜ | Sₜ) × P(Sₜ₊₁ | Sₜ,Aₜ)`

Only the policy term depends on `θ`.

The environment transition is not controlled directly by the policy parameters.

Therefore, learning changes the trajectory distribution by changing:

`πθ(Aₜ | Sₜ)`

---

# 105. Why Ordinary Supervised Learning Does Not Directly Apply

In supervised learning, we normally know the correct output.

Example:

```text
Input Image

↓

Correct Label = Cat
```

The loss can directly compare the prediction with the correct label.

In Reinforcement Learning, we may only know the final reward.

Example:

```text
Sequence of 50 Actions

↓

Final Reward = +10
```

We are not explicitly told:

- which individual action was correct,
- which action was unnecessary,
- which action caused success.

This creates the **credit assignment problem**.

---

# 106. The Credit Assignment Problem

The credit assignment problem asks:

> Which actions in a trajectory deserve credit or blame for the final outcome?

Suppose a chess agent wins after 80 moves.

Was the victory caused by:

- the first move,
- a move in the middle,
- the final move,
- a combination of many moves?

The final reward evaluates the complete trajectory.

The learning algorithm must determine how to update the individual action probabilities.

---

# 107. Credit Assignment in an LLM

Suppose a language model generates a 500-token answer.

A human gives the complete answer a low reward.

Which tokens caused the problem?

Possibilities include:

- an incorrect statement near the beginning,
- an irrelevant paragraph,
- an unsafe recommendation,
- a poor conclusion,
- repeated text.

The reward may apply to the complete response rather than to each individual token.

The model must still update the probability of the token choices that produced that response.

---

# 108. The Main Optimization Challenge

We want to calculate:

`∇θJ(θ)`

This tells us how to change the policy parameters to increase the expected return.

However:

`J(θ) = Eτ~Pθ(τ)[G(τ)]`

contains an expectation over all possible trajectories.

In realistic environments, the number of possible trajectories can be enormous.

We cannot enumerate all of them.

---

# 109. Why Backpropagation Through the Environment May Be Impossible

The environment may contain operations that are:

- unknown,
- non-differentiable,
- stochastic,
- controlled by external systems,
- based on human feedback.

For example, a human rating a response is not a differentiable mathematical function available to the model.

Therefore, we need a method that can improve the policy using sampled trajectories and rewards.

This leads to the **Policy Gradient** idea.

---

# 110. What Is Policy Optimization?

Policy optimization means directly adjusting the parameters `θ` of the policy to increase expected return.

Conceptually:

```text
Generate Trajectories

↓

Observe Rewards

↓

Determine Which Actions Were Useful

↓

Increase Their Probabilities

↓

Reduce Probabilities of Poor Actions

↓

Repeat
```

A policy-gradient method performs this update using gradients.

---

# 111. The Basic Policy-Gradient Intuition

Suppose an action appears in a trajectory with a large positive return.

The algorithm should make that action more likely in similar states.

Suppose an action appears in a trajectory with a poor return.

The algorithm should make that action less likely.

Conceptually:

```text
High-Return Trajectory

↓

Reinforce Its Actions
```

```text
Low-Return Trajectory

↓

Discourage Its Actions
```

This is the central intuition behind policy-gradient methods.

---

# 112. The Log-Probability of an Action

The policy produces:

`πθ(Aₜ | Sₜ)`

Policy-gradient methods work with its logarithm:

`log πθ(Aₜ | Sₜ)`

The gradient:

`∇θ log πθ(Aₜ | Sₜ)`

indicates how the parameters should change to increase the probability of the selected action.

Multiplying this gradient by a return creates a learning signal.

---

# 113. Simplified Policy-Gradient Form

A simplified policy-gradient expression is:

`∇θJ(θ) = Eτ~Pθ(τ)[Σₜ ∇θ log πθ(Aₜ | Sₜ) × Gₜ]`

This equation may initially look complicated.

Its meaning is simple:

1. Generate trajectories using the current policy.
2. Calculate the return following each action.
3. Compute the gradient of the selected action's log-probability.
4. Scale the gradient by the return.
5. Average across trajectories.

---

# 114. Understanding the Sign of the Update

Suppose:

`Gₜ > 0`

Then gradient ascent increases:

`log πθ(Aₜ | Sₜ)`

Therefore, the selected action becomes more likely.

If:

`Gₜ < 0`

the update moves in the opposite direction.

Therefore, the selected action becomes less likely.

---

# 115. Easy Example

Suppose a robot is in a state with two actions.

| Action | Current probability |
|---|---:|
| Move toward charger | 0.40 |
| Move away | 0.60 |

The robot chooses:

```text
Move toward charger
```

and later receives a large positive return.

The policy-gradient update increases the probability of:

```text
Move toward charger
```

when the robot encounters a similar state again.

Over many attempts, useful actions become increasingly likely.

---

# 116. Sampling Instead of Enumerating All Trajectories

The expected policy gradient involves all possible trajectories.

In practice, we approximate it using sampled trajectories.

Suppose we collect `N` trajectories.

A Monte Carlo estimate is:

`∇θJ(θ) ≈ (1/N) Σᵢ Σₜ ∇θ log πθ(Aₜ⁽ⁱ⁾ | Sₜ⁽ⁱ⁾) × Gₜ⁽ⁱ⁾`

This follows the same principle used throughout the course:

```text
Unknown Expectation

↓

Approximate Using Sample Average
```

This is another application of the Law of Large Numbers.

---

# 117. Why the Logarithm Appears

The probability of a trajectory contains a product of policy probabilities.

`Pθ(τ) ∝ πθ(A₀ | S₀) × πθ(A₁ | S₁) × ...`

Taking the logarithm converts the product into a sum.

`log Pθ(τ) = constant + Σₜ log πθ(Aₜ | Sₜ)`

Differentiating a sum is easier than differentiating a long product.

This is one reason log-probabilities appear naturally in policy-gradient algorithms.

---

# 118. The Log-Derivative Trick

A useful identity is:

`∇θPθ(τ) = Pθ(τ)∇θ log Pθ(τ)`

This is called the:

**Log-Derivative Trick**

or:

**Score-Function Trick**

It allows the gradient of an expectation to be rewritten as another expectation that can be estimated using sampled trajectories.

This avoids the need to differentiate directly through the environment.

---

# 119. High-Level Derivation

Start with:

`J(θ) = Στ Pθ(τ)G(τ)`

Differentiate:

`∇θJ(θ) = Στ ∇θPθ(τ)G(τ)`

Apply the log-derivative identity:

`∇θJ(θ) = Στ Pθ(τ)∇θ log Pθ(τ)G(τ)`

Recognize the expectation:

`∇θJ(θ) = Eτ~Pθ(τ)[∇θ log Pθ(τ)G(τ)]`

Since only the policy depends on `θ`:

`∇θ log Pθ(τ) = Σₜ ∇θ log πθ(Aₜ | Sₜ)`

Therefore:

`∇θJ(θ) = Eτ~Pθ(τ)[Σₜ ∇θ log πθ(Aₜ | Sₜ)G(τ)]`

This is the basic policy-gradient idea.

---

# 120. Why Environment Gradients Disappear

Recall:

`Pθ(τ) = ρ₀(S₀) × ∏ₜ πθ(Aₜ | Sₜ) × P(Sₜ₊₁ | Sₜ,Aₜ)`

The initial-state distribution and environment transitions do not depend on `θ`.

Therefore:

`∇θ log ρ₀(S₀) = 0`

and:

`∇θ log P(Sₜ₊₁ | Sₜ,Aₜ) = 0`

Only the policy terms remain.

This allows policy gradients to learn even when the environment itself is non-differentiable.

---

# 121. Policy Gradient as Weighted Maximum Likelihood

There is an intuitive connection to maximum-likelihood training.

In ordinary language-model training, we increase the probability of target tokens.

In policy-gradient training, we increase the probability of sampled actions according to their rewards.

Conceptually:

```text
Maximum Likelihood

↓

Increase Probability of Correct Training Targets
```

```text
Policy Gradient

↓

Increase Probability of Actions Producing High Return
```

The reward acts like a weight.

---

# 122. Connection to Language Models

For an LLM:

`πθ(Aₜ | Sₜ) = Pθ(next token | prompt and previous tokens)`

A complete response is a trajectory.

Suppose the model generates:

```text
Gradient descent is like walking downhill by repeatedly taking steps in the direction that reduces the height.
```

A human evaluator may assign a high reward because the response is:

- correct,
- clear,
- relevant,
- easy to understand.

Policy optimization then increases the probability of the token decisions that produced this kind of response.

---

# 123. LLM Policy-Gradient Mapping

| Policy-gradient concept | LLM interpretation |
|---|---|
| State `Sₜ` | Prompt and tokens generated so far |
| Action `Aₜ` | Next token |
| Policy `πθ` | Next-token distribution |
| Trajectory `τ` | Complete response |
| Return `G(τ)` | Quality score of the response |
| Policy update | Change token probabilities |

---

# 124. Sequence-Level Reward

In many LLM alignment settings, the reward is given only after the complete response is generated.

Example:

```text
Prompt

↓

Generate 300 Tokens

↓

Human or Reward Model Evaluates Complete Answer

↓

Reward = 8.5
```

The same sequence-level reward may initially be associated with many token actions.

This makes credit assignment difficult.

Later algorithms introduce additional techniques to make training more stable.

---

# 125. Why Policy Gradients Can Be Noisy

Suppose two responses are generated from the same prompt.

One receives a high reward.

Another receives a low reward.

The difference may be caused by:

- random token sampling,
- evaluator inconsistency,
- one small phrase,
- genuine quality differences.

Therefore, sampled policy-gradient estimates can have high variance.

This means that the estimated gradient may change considerably from one batch of trajectories to another.

---

# 126. What Is Variance Here?

Variance measures how much the gradient estimate changes across different trajectory samples.

```text
Low Variance

↓

Different Samples Produce Similar Gradient Directions
```

```text
High Variance

↓

Different Samples Produce Very Different Gradient Directions
```

High variance makes optimization unstable and slow.

This is one reason practical policy-gradient methods use additional techniques such as:

- baselines,
- value functions,
- advantages,
- constrained policy updates.

---

# 127. Why We Need More Advanced Algorithms

The basic policy-gradient idea is powerful, but directly applying it can cause problems.

The policy may:

- change too much in one update,
- forget useful behavior,
- exploit weaknesses in the reward function,
- become unstable,
- produce less diverse responses.

Therefore, practical LLM alignment uses more controlled optimization methods.

One important family is:

**Proximal Policy Optimization**

or:

**PPO**

---

# 128. Reward Is Difficult to Define for Language

For a video game, the reward may be easy to measure.

```text
Score Increased by 100
```

For a language response, quality is much more subjective.

Questions include:

- Is the answer factually correct?
- Is it relevant?
- Is it clear?
- Is it safe?
- Is it too long?
- Does it follow the user's instructions?

Different humans may assign different scores.

Therefore, directly defining a numerical reward is difficult.

---

# 129. Preference Instead of Absolute Score

Instead of asking a human:

```text
Give this response a score from 0 to 10.
```

we may show two responses and ask:

```text
Which response is better?
```

Example:

```text
Response A

vs.

Response B
```

The result is a preference:

`A > B`

or:

`B > A`

Pairwise preferences are often easier and more consistent than absolute scores.

---

# 130. From Preferences to Learning

Preference-based alignment methods try to answer:

> How can we update the language model when humans provide comparisons rather than exact numerical rewards?

Different methods solve this problem differently.

Examples include:

- RLHF with a learned reward model,
- PPO,
- Direct Preference Optimization,
- Group Relative Policy Optimization.

These methods will be studied in the next part.

---

# 131. Common Misconceptions

## Misconception 1

**The policy should maximize the immediate reward at every step.**

Not necessarily.

The objective is to maximize expected long-term return.

An action with a small immediate cost may lead to a much larger future reward.

---

## Misconception 2

**Gamma controls the learning rate.**

It does not.

`γ` controls the importance of future rewards.

The learning rate controls the size of parameter updates.

---

## Misconception 3

**One high-reward trajectory proves that the policy is good.**

It does not.

The trajectory may have been lucky.

Policy quality is measured using expected return over many trajectories.

---

## Misconception 4

**The reward directly tells the agent the correct action.**

Usually it does not.

Reward evaluates the outcome.

The algorithm must determine which actions deserve credit.

---

## Misconception 5

**Policy gradient requires differentiating through the environment.**

It does not.

The log-derivative formulation allows learning using sampled actions and rewards.

---

## Misconception 6

**Every token receives a separate human reward.**

Usually not.

Human feedback often evaluates the complete response.

---

## Misconception 7

**A larger reward always means every action in the trajectory was good.**

Not necessarily.

Some actions may have been irrelevant or harmful even if the final trajectory succeeded.

This is part of the credit assignment problem.

---

# Key Insights

- A trajectory is evaluated using the rewards it accumulates.
- Discounted return combines immediate and future rewards.
- The discount factor controls how strongly future rewards matter.
- Policy quality is measured using expected return across trajectories.
- Reinforcement Learning finds parameters that maximize expected return.
- Changing the policy changes the distribution of trajectories.
- Credit assignment determines which actions deserve credit for an outcome.
- Policy-gradient methods increase the probability of actions associated with high returns.
- The log-derivative trick makes policy optimization possible without differentiating through the environment.
- Sample averages approximate otherwise intractable expectations.
- An LLM can be optimized as a policy over next-token actions.
- Human preference is often easier to obtain as pairwise comparisons than as absolute numerical rewards.

---

# What's Next?

The basic policy-gradient framework explains how rewards can modify a language model.

However, practical alignment introduces several additional questions:

- Where does the language-model reward come from?
- How can human comparisons be converted into a reward model?
- How do we prevent the policy from changing too aggressively?
- Can we optimize preferences without training a separate reward model?

In Part 4, we will study the high-level ideas behind:

- Reward Modeling,
- Reinforcement Learning from Human Feedback,
- Proximal Policy Optimization,
- Direct Preference Optimization,
- Group Relative Policy Optimization.

These methods connect human preferences to practical LLM post-training.
