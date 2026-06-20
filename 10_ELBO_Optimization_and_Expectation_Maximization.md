Based on Prof. Prathosh A.P's lecture on Generative AI Principles and Applications.

Core Theme:

How do we learn a latent variable model?

The answer is:

Maximize the Evidence Lower Bound (ELBO).

1. Where We Are in the Story

In the previous note we introduced latent variable models.

A latent variable model assumes:

\int P_\theta(x,z),dz
$$

where:

(x) = observed variable
(z) = hidden (latent) variable

The objective remains unchanged:

$$
P_\theta \approx P_X
$$

where:

(P_X) = true data distribution
(P_\theta) = model distribution
2. Intuition First

Imagine you are a detective.

You observe:

Evidence (X)

but the true cause:

Hidden Cause (Z)

is invisible.

Examples:

Observed Data X	Hidden Variable Z
Face Image	Pose, Age, Lighting
Customer Purchases	Customer Type
Speech Signal	Phoneme Sequence
Medical Measurements	Disease Category

The challenge:

We must learn:

Model parameters (\theta)
Hidden causes (Z)

simultaneously.

3. Real World Example

Suppose a hospital collects:

Temperature
Blood Pressure
Heart Rate

for thousands of patients.

But patients secretly belong to:

Healthy
Mild Infection
Severe Infection

Nobody tells us these categories.

They are latent variables.

Our goal is to discover these hidden groups while learning the data distribution.

4. From KL Divergence to Maximum Likelihood

Prathosh starts from:

\arg\min_\theta
D_{KL}(P_X | P_\theta)
$$

Expanding KL divergence:

\int
P_X(x)
\log
\frac{P_X(x)}
{P_\theta(x)}
,dx
$$

Expanding further:

\int P_X(x)\log P_X(x),dx

\int P_X(x)\log P_\theta(x),dx
$$

The first term does not depend on (\theta).

Therefore:

\arg\max_\theta
\mathbb E_{P_X}
\left[
\log P_\theta(x)
\right]
$$

5. Key Insight

Prathosh emphasizes:

Minimizing KL Divergence
=
Maximizing Log Likelihood

These are mathematically equivalent.

This procedure is called:

Maximum Likelihood Estimation (MLE)
6. The New Problem

Under a latent variable model:

\int
P_\theta(x,z)
,dz
$$

Therefore:

\log
\left(
\int
P_\theta(x,z)
,dz
\right)
$$

The logarithm sits outside the integral.

This makes optimization difficult.

7. Why Is It Difficult?

We know how to estimate:

$$
\mathbb E[\cdot]
$$

using samples.

But we do not know how to efficiently estimate:

$$
\log \mathbb E[\cdot]
$$

This is the central difficulty in latent variable models.

8. Introducing Q(z|x)

Prathosh introduces:

$$
Q(z|x)
$$

which is any probability density function over the latent variable.

Think of it as:

A guess about the hidden cause.

We multiply and divide by (Q(z|x)):

P_\theta(x,z)
\frac{Q(z|x)}
{Q(z|x)}
$$

which does not change the value.

9. Rewriting the Likelihood

We obtain:

\log
\left(
\int
Q(z|x)
\frac
{P_\theta(x,z)}
{Q(z|x)}
,dz
\right)
$$

Recognize:

\mathbb E_{Q(z|x)}
[f(z)]
$$

Therefore:

\log
\left(
\mathbb E_{Q(z|x)}
\left[
\frac
{P_\theta(x,z)}
{Q(z|x)}
\right]
\right)
$$

10. Jensen's Inequality

Prathosh now uses Jensen's inequality.

For the logarithm:

$$
\log \mathbb E[X]
\ge
\mathbb E[\log X]
$$

Applying Jensen:

$$
\log P_\theta(x)
\ge
\mathbb E_{Q(z|x)}
\left[
\log
\frac
{P_\theta(x,z)}
{Q(z|x)}
\right]
$$

11. Evidence Lower Bound (ELBO)

The right-hand side is called:

Evidence Lower Bound

or

ELBO

We define:

\mathbb E_{Q(z|x)}
\left[
\log
\frac
{P_\theta(x,z)}
{Q(z|x)}
\right]
$$

Thus:

$$
\log P_\theta(x)
\ge
J_\theta(Q)
$$

12. Why Is It Called ELBO?

The likelihood:

$$
P_\theta(x)
$$

is also called:

Evidence

because it measures:

How well does the model explain the observed data?

Since we derived a lower bound on evidence:

Evidence Lower Bound
=
ELBO
13. The Fundamental Optimization Problem

Originally:

$$
\max_\theta
\log P_\theta(x)
$$

Now:

\arg\max_{\theta,Q}
J_\theta(Q)
$$

Notice:

The ELBO depends on:

Model parameters

$$
\theta
$$

Variational distribution

$$
Q(z|x)
$$

Therefore:

We must learn both simultaneously.

14. Variational Latent Posterior

Prathosh calls:

$$
Q(z|x)
$$

the

Variational Latent Posterior

because:

It is a distribution.
It is optimized.
It belongs to a family of functions.

Optimization over functions comes from:

Variational Calculus
15. The Most Important Result

Every latent variable generative model solves:

\arg\max_{\theta,Q}
J_\theta(Q)
$$

Prathosh explicitly states:

This is the foundational problem behind
all latent variable models.

Including:

Gaussian Mixture Models (GMM)
Variational Autoencoders (VAE)
Diffusion Models
16. Example: Gaussian Mixture Model (GMM)

Consider a latent variable:

$$
Z
$$

which is discrete:

$$
Z \in {1,2,\ldots,M}
$$

Then:

\sum_{j=1}^{M}
P_\theta(z=j)
P_\theta(x|z=j)
$$

17. GMM Definition

For a GMM:

\alpha_j
$$

and

\mathcal N(x;\mu_j,\Sigma_j)
$$

Therefore:

\sum_{j=1}^{M}
\alpha_j
,
\mathcal N(x;\mu_j,\Sigma_j)
$$

This is why it is called:

Gaussian Mixture Model
18. Parameters of a GMM

We must estimate:

{
\alpha_j,\mu_j,\Sigma_j
}_{j=1}^{M}
$$

subject to:

$$
\alpha_j \ge 0
$$

and

1
$$

19. How Do We Optimize ELBO?

Prathosh introduces:

Expectation Maximization (EM)

Idea:

Alternate optimization.

Step 1:

Fix (\theta)

Optimize (Q)

Step 2:

Fix (Q)

Optimize (\theta)

Repeat.

20. EM Algorithm

Initialize:

$$
\theta^{(0)},
Q^{(0)}
$$

arbitrarily.

For iteration (t):

E-Step

\arg\max_Q
J_{\theta^{(t)}}(Q)
$$

M-Step

\arg\max_\theta
J_\theta
\left(
Q^{(t+1)}
\right)
$$

Repeat until convergence.

21. Why EM Works

Prathosh states:

EM guarantees:

$$
L^{(t+1)}
\ge
L^{(t)}
$$

Meaning:

Likelihood never decreases.

EM may not reach the global optimum.

But every iteration improves or preserves the likelihood.

22. Solving the E-Step for GMM

A beautiful result:

P_\theta(z|x)
$$

The optimal variational posterior equals the exact posterior.

Using Bayes rule:

\frac{
P_\theta(x|z)
P_\theta(z)
}{
P_\theta(x)
}
$$

Since all terms are known for GMM:

The posterior can be computed exactly.

Therefore:

The E-Step becomes tractable.

23. Solving the M-Step

After obtaining:

$$
Q^*(z|x)
$$

we maximize:

$$
J_\theta(Q^*)
$$

with respect to:

{
\alpha_j,\mu_j,\Sigma_j
}
$$

This becomes a standard optimization problem.

Differentiate.

Set gradients to zero.

Solve.

24. Why EM Eventually Fails

EM works only when:

$$
P_\theta(z|x)
$$

can be computed.

For GMM:

Yes.

For simple latent variable models:

Yes.

For neural-network-based latent variable models:

Usually no.

25. The Big Limitation

Suppose:

$$
P_\theta(z|x)
$$

cannot be computed.

Then:

P_\theta(z|x)
$$

cannot be found.

Therefore:

E-Step fails.

EM can no longer be applied.

26. The Big Question

Prathosh ends with:

How do we learn latent variable models
when Pθ(z|x) is unknown?

This is exactly the motivation for:

Variational Autoencoders (VAEs)
27. Bridge to VAEs

VAEs solve the problem by:

Approximating
P_\theta(z|x)
using neural networks.

Instead of computing:

$$
P_\theta(z|x)
$$

exactly,

they learn an approximation.

Final Takeaway

Every latent variable model starts with:

\int
P_\theta(x,z)
,dz
$$

and ultimately solves:

\arg\max_{\theta,Q}
ELBO
$$

For simple models:

GMM
C-Means

the ELBO is optimized using:

Expectation Maximization (EM)

For deep models:

Variational Autoencoders
Diffusion Models

the posterior

$$
P_\theta(z|x)
$$

is no longer tractable.

This leads directly to the next topic:
