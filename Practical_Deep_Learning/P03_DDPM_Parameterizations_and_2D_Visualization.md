This practical note compares four common DDPM prediction targets:

- clean-sample prediction,
- noise prediction,
- reverse-mean prediction,
- score prediction.

It also provides a small 2D PyTorch lab for visualizing:

- forward diffusion,
- the approach toward an isotropic Gaussian,
- conversion between prediction targets,
- reverse sampling behavior,
- training progress across epochs.

The code is a representative mini-lab designed to make the tutorial concepts reproducible.

---

## Learning Objectives

After completing this practical, you should be able to:

- implement the DDPM forward noising equation,
- create a linear beta schedule,
- visualize a 2D dataset at different diffusion times,
- derive one prediction target from another,
- train a time-conditioned network to predict `X₀` or noise,
- understand why equivalent parameterizations may behave differently in optimization,
- inspect reverse samples across time and training epochs.

---

## Prerequisites

You should already know:

- the DDPM forward process,
- `βₜ`, `αₜ` and `ᾱₜ`,
- the role of a time-conditioned neural network,
- basic PyTorch tensors and optimization,
- the difference between training and reverse-time sampling.

---

# 1. The Four Prediction Targets

A DDPM network receives a noisy sample and a time step:

`model input = (Xₜ,t)`

The network can be trained to predict different but mathematically related quantities.

| Parameterization | Network output |
|---|---|
| Sample prediction | `X̂₀,θ(Xₜ,t)` |
| Noise prediction | `ε̂θ(Xₜ,t)` |
| Mean prediction | `μθ(Xₜ,t)` |
| Score prediction | `sθ(Xₜ,t)` |

The network architecture may be similar in all four cases.

The main difference is the target and the formula used during reverse sampling.

---

# 2. Forward Diffusion Equation

The direct noising equation is:

`Xₜ = √ᾱₜ X₀ + √(1−ᾱₜ) ε`

where:

- `X₀` is a clean data sample,
- `ε ∼ N(0,I)` is Gaussian noise,
- `αₜ = 1−βₜ`,
- `ᾱₜ = ∏ₛ₌₁ᵗ αₛ`.

This lets us sample `Xₜ` directly without simulating every earlier step.

---

# 3. Intuition for Different Times

At a small time step:

`ᾱₜ ≈ 1`

so `Xₜ` contains mostly data.

At a large time step:

`ᾱₜ ≈ 0`

so `Xₜ` contains mostly Gaussian noise.

```text
Structured Data

↓ Add Small Noise

Slightly Blurred Distribution

↓ Add More Noise

Weak Structure

↓ Add Much More Noise

Approximately Isotropic Gaussian
```

---

# 4. Why Use 2D Data?

Image diffusion is difficult to inspect directly because each sample has thousands of dimensions.

A 2D dataset makes the complete geometry visible.

You can directly observe:

- clusters spreading,
- modes merging,
- the distribution becoming rounder,
- reverse samples moving from noise back toward data modes.

This is extremely useful for debugging.

---

# 5. Environment Setup

Install the required packages:

```bash
pip install torch matplotlib numpy
```

Imports:

```python
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
```

Select a device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
```

---

# 6. Create a Synthetic 2D Dataset

The following function creates eight Gaussian clusters arranged on a ring.

```python
def make_eight_gaussians(
    n_samples: int = 20_000,
    radius: float = 4.0,
    cluster_std: float = 0.25,
    seed: int = 42,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)

    component_ids = torch.randint(
        low=0,
        high=8,
        size=(n_samples,),
        generator=generator,
    )

    angles = 2.0 * math.pi * component_ids.float() / 8.0
    centers = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)],
        dim=1,
    )

    noise = cluster_std * torch.randn(
        n_samples,
        2,
        generator=generator,
    )

    samples = centers + noise

    # Normalizing keeps the scale convenient for a simple beta schedule.
    return samples / radius


x0_data = make_eight_gaussians()
print(x0_data.shape)
```

Visualize the clean data:

```python
plt.figure(figsize=(6, 6))
plt.scatter(x0_data[:, 0], x0_data[:, 1], s=3, alpha=0.5)
plt.title("Clean 2D Data: Eight Gaussian Modes")
plt.axis("equal")
plt.show()
```

---

# 7. Build the Diffusion Schedule

We will use `T=200`, matching the tutorial's 2D visualization example.

```python
@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    posterior_variance: torch.Tensor


def make_linear_schedule(
    timesteps: int = 200,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    alpha_bars_prev = torch.cat(
        [torch.ones(1), alpha_bars[:-1]],
        dim=0,
    )

    posterior_variance = (
        betas
        * (1.0 - alpha_bars_prev)
        / (1.0 - alpha_bars)
    )

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_prev=alpha_bars_prev,
        posterior_variance=posterior_variance,
    )


schedule = make_linear_schedule(timesteps=200)
T = schedule.betas.numel()
```

---

# 8. Utility for Extracting Time-Dependent Values

Each sample in a batch may use a different time step.

```python
def extract(values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    selected = values.to(x.device)[t]
    return selected.reshape(t.shape[0], *([1] * (x.ndim - 1)))
```

For 2D data with shape `[batch, 2]`, this converts a vector of time-dependent scalars into shape `[batch, 1]` for broadcasting.

---

# 9. Implement Direct Forward Sampling

```python
def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bar_t = extract(schedule.alpha_bars, t, x0)

    xt = (
        torch.sqrt(alpha_bar_t) * x0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )

    return xt, noise
```

This implements:

`Xₜ = √ᾱₜ X₀ + √(1−ᾱₜ) ε`

---

# 10. Visualize Forward Diffusion

```python
@torch.no_grad()
def plot_forward_snapshots(
    x0: torch.Tensor,
    schedule: DiffusionSchedule,
    times: list[int],
    max_points: int = 5_000,
) -> None:
    subset = x0[:max_points]

    for time_index in times:
        t = torch.full(
            (subset.shape[0],),
            fill_value=time_index,
            dtype=torch.long,
        )

        xt, _ = q_sample(subset, t, schedule)

        plt.figure(figsize=(6, 6))
        plt.scatter(xt[:, 0], xt[:, 1], s=3, alpha=0.5)
        plt.title(f"Forward Diffusion at t={time_index}")
        plt.axis("equal")
        plt.show()


plot_forward_snapshots(
    x0=x0_data,
    schedule=schedule,
    times=[0, 20, 60, 120, 199],
)
```

Expected observation:

```text
t = 0
Distinct Modes

↓

t = 60
Modes Spread and Overlap

↓

t = 199
Approximately Isotropic Gaussian
```

---

# 11. Sample-Prediction Parameterization

The network predicts the original clean sample:

`X̂₀,θ = fθ(Xₜ,t)`

A simple training loss is:

`L_X₀ = E[‖X̂₀,θ(Xₜ,t) − X₀‖²]`

## Intuition

The network receives a corrupted point and tries to locate the clean point that produced it.

For small 2D datasets, this target can be easy to visualize and may train well.

---

# 12. Noise-Prediction Parameterization

The network predicts the noise used in the forward equation:

`ε̂θ = fθ(Xₜ,t)`

The loss is:

`L_ε = E[‖ε̂θ(Xₜ,t) − ε‖²]`

The clean sample can be recovered using:

`X̂₀,θ = (Xₜ − √(1−ᾱₜ) ε̂θ) / √ᾱₜ`

Noise prediction is widely used because it gives a convenient, standardized target.

---

# 13. Convert Sample Prediction to Noise Prediction

If the network predicts `X̂₀,θ`, the implied noise estimate is:

`ε̂θ = (Xₜ − √ᾱₜ X̂₀,θ) / √(1−ᾱₜ)`

Implementation:

```python
def predict_noise_from_x0(
    xt: torch.Tensor,
    t: torch.Tensor,
    x0_pred: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    alpha_bar_t = extract(schedule.alpha_bars, t, xt)

    denominator = torch.sqrt(1.0 - alpha_bar_t).clamp_min(1e-8)

    return (
        xt - torch.sqrt(alpha_bar_t) * x0_pred
    ) / denominator
```

---

# 14. Convert Noise Prediction to Sample Prediction

```python
def predict_x0_from_noise(
    xt: torch.Tensor,
    t: torch.Tensor,
    noise_pred: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    alpha_bar_t = extract(schedule.alpha_bars, t, xt)

    denominator = torch.sqrt(alpha_bar_t).clamp_min(1e-8)

    return (
        xt - torch.sqrt(1.0 - alpha_bar_t) * noise_pred
    ) / denominator
```

This implements:

`X̂₀,θ = (Xₜ − √(1−ᾱₜ) ε̂θ) / √ᾱₜ`

At very large `t`, `ᾱₜ` can be small, so numerical care is important.

---

# 15. Score-Prediction Parameterization

The score is the gradient of the log density with respect to the noisy sample.

For the forward conditional Gaussian:

`∇ₓₜ log q(Xₜ|X₀) = −ε / √(1−ᾱₜ)`

Therefore, a score target is:

`s_target = −ε / √(1−ᾱₜ)`

A score network predicts:

`sθ(Xₜ,t) ≈ ∇ₓₜ log qₜ(Xₜ)`

The score points toward regions of higher probability density.

Implementation of the conditional score target:

```python
def score_target(
    noise: torch.Tensor,
    t: torch.Tensor,
    xt: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    alpha_bar_t = extract(schedule.alpha_bars, t, xt)
    std_t = torch.sqrt(1.0 - alpha_bar_t).clamp_min(1e-8)
    return -noise / std_t
```

---

# 16. Reverse-Mean Parameterization

The true posterior is:

`q(Xₜ₋₁|Xₜ,X₀) = N(μ̃ₜ(Xₜ,X₀), β̃ₜI)`

Its mean is:

`μ̃ₜ(Xₜ,X₀) = [√ᾱₜ₋₁ βₜ/(1−ᾱₜ)]X₀ + [√αₜ(1−ᾱₜ₋₁)/(1−ᾱₜ)]Xₜ`

A mean-prediction model directly estimates:

`μθ(Xₜ,t) ≈ μ̃ₜ(Xₜ,X₀)`

Implementation:

```python
def posterior_mean_target(
    x0: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    beta_t = extract(schedule.betas, t, xt)
    alpha_t = extract(schedule.alphas, t, xt)
    alpha_bar_t = extract(schedule.alpha_bars, t, xt)
    alpha_bar_prev_t = extract(schedule.alpha_bars_prev, t, xt)

    denominator = (1.0 - alpha_bar_t).clamp_min(1e-8)

    coefficient_x0 = (
        torch.sqrt(alpha_bar_prev_t) * beta_t / denominator
    )

    coefficient_xt = (
        torch.sqrt(alpha_t)
        * (1.0 - alpha_bar_prev_t)
        / denominator
    )

    return coefficient_x0 * x0 + coefficient_xt * xt
```

---

# 17. Unified Target Builder

```python
def build_target(
    parameterization: str,
    x0: torch.Tensor,
    xt: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    if parameterization == "x0":
        return x0

    if parameterization == "noise":
        return noise

    if parameterization == "score":
        return score_target(noise, t, xt, schedule)

    if parameterization == "mean":
        return posterior_mean_target(x0, xt, t, schedule)

    raise ValueError(
        "parameterization must be one of: x0, noise, score, mean"
    )
```

This lets the same network and training loop compare all four targets.

---

# 18. Time Embedding for a 2D Network

The model must know the noise level.

```python
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Time embedding dimension must be even.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2

        frequency_scale = math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            -frequency_scale
            * torch.arange(half_dim, device=t.device)
        )

        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)

        return torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1,
        )
```

---

# 19. Tiny Time-Conditioned MLP

```python
class TimeConditionedMLP(nn.Module):
    def __init__(
        self,
        data_dim: int = 2,
        time_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.network = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        return self.network(torch.cat([x, t_emb], dim=1))
```

For image data, this MLP would be replaced by a time-conditioned U-Net.

---

# 20. Training Loop

```python
def train_model(
    parameterization: str,
    x0_data: torch.Tensor,
    schedule: DiffusionSchedule,
    epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
) -> tuple[nn.Module, list[float]]:
    dataset = TensorDataset(x0_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = TimeConditionedMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for (x0_batch,) in loader:
            x0_batch = x0_batch.to(device)

            t = torch.randint(
                low=0,
                high=T,
                size=(x0_batch.shape[0],),
                device=device,
            )

            xt, noise = q_sample(x0_batch, t, schedule)

            target = build_target(
                parameterization=parameterization,
                x0=x0_batch,
                xt=xt,
                noise=noise,
                t=t,
                schedule=schedule,
            )

            prediction = model(xt, t)
            loss = loss_function(prediction, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(loader)
        losses.append(average_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"{parameterization=} | epoch={epoch:03d} "
                f"| loss={average_loss:.6f}"
            )

    return model, losses
```

Train a clean-sample predictor:

```python
x0_model, x0_losses = train_model(
    parameterization="x0",
    x0_data=x0_data,
    schedule=schedule,
    epochs=100,
)
```

Train a noise predictor:

```python
noise_model, noise_losses = train_model(
    parameterization="noise",
    x0_data=x0_data,
    schedule=schedule,
    epochs=100,
)
```

---

# 21. Convert Any Model Output into an `X₀` Estimate

Reverse sampling is easiest when every parameterization is converted to a common quantity.

```python
def model_output_to_x0(
    model_output: torch.Tensor,
    parameterization: str,
    xt: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    if parameterization == "x0":
        return model_output

    if parameterization == "noise":
        return predict_x0_from_noise(
            xt=xt,
            t=t,
            noise_pred=model_output,
            schedule=schedule,
        )

    if parameterization == "score":
        alpha_bar_t = extract(schedule.alpha_bars, t, xt)
        noise_pred = -model_output * torch.sqrt(1.0 - alpha_bar_t)
        return predict_x0_from_noise(
            xt=xt,
            t=t,
            noise_pred=noise_pred,
            schedule=schedule,
        )

    raise NotImplementedError(
        "Direct mean output should be used as the posterior mean rather than "
        "converted through this helper."
    )
```

---

# 22. Reverse Mean from an `X₀` Estimate

```python
def posterior_mean_from_x0_prediction(
    xt: torch.Tensor,
    t: torch.Tensor,
    x0_pred: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    return posterior_mean_target(
        x0=x0_pred,
        xt=xt,
        t=t,
        schedule=schedule,
    )
```

For noise prediction, the equivalent reverse mean can also be written as:

`μθ(Xₜ,t) = 1/√αₜ · [Xₜ − βₜ/√(1−ᾱₜ) · εθ(Xₜ,t)]`

---

# 23. Reverse Sampling

```python
@torch.no_grad()
def sample_reverse_process(
    model: nn.Module,
    parameterization: str,
    schedule: DiffusionSchedule,
    n_samples: int = 5_000,
    save_times: set[int] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    if save_times is None:
        save_times = {199, 150, 100, 50, 10, 0}

    model.eval()

    xt = torch.randn(n_samples, 2, device=device)
    snapshots: dict[int, torch.Tensor] = {}

    for time_index in reversed(range(T)):
        t = torch.full(
            (n_samples,),
            fill_value=time_index,
            dtype=torch.long,
            device=device,
        )

        model_output = model(xt, t)

        if parameterization == "mean":
            mean = model_output
        else:
            x0_pred = model_output_to_x0(
                model_output=model_output,
                parameterization=parameterization,
                xt=xt,
                t=t,
                schedule=schedule,
            )

            # Optional clipping can improve stability for normalized data.
            x0_pred = x0_pred.clamp(-2.0, 2.0)

            mean = posterior_mean_from_x0_prediction(
                xt=xt,
                t=t,
                x0_pred=x0_pred,
                schedule=schedule,
            )

        if time_index > 0:
            variance_t = extract(
                schedule.posterior_variance,
                t,
                xt,
            ).clamp_min(1e-12)

            xt = mean + torch.sqrt(variance_t) * torch.randn_like(xt)
        else:
            xt = mean

        if time_index in save_times:
            snapshots[time_index] = xt.detach().cpu()

    return xt.detach().cpu(), snapshots
```

Generate samples:

```python
final_samples, reverse_snapshots = sample_reverse_process(
    model=x0_model,
    parameterization="x0",
    schedule=schedule,
)
```

---

# 24. Visualize the Reverse Process

```python
def plot_reverse_snapshots(snapshots: dict[int, torch.Tensor]) -> None:
    for time_index in sorted(snapshots.keys(), reverse=True):
        points = snapshots[time_index]

        plt.figure(figsize=(6, 6))
        plt.scatter(points[:, 0], points[:, 1], s=3, alpha=0.5)
        plt.title(f"Reverse Process at t={time_index}")
        plt.axis("equal")
        plt.show()


plot_reverse_snapshots(reverse_snapshots)
```

Expected qualitative behavior:

```text
Gaussian Noise

↓

Weak Geometric Structure

↓

Separated Modes Begin to Form

↓

Samples Concentrate Near the Data Distribution
```

---

# 25. Visualize Training Progress Across Epochs

To reproduce the tutorial idea of watching the model improve over epochs, save checkpoints periodically.

```python
checkpoint_epochs = {0, 5, 10, 25, 50, 99}
checkpoints: dict[int, dict[str, torch.Tensor]] = {}
```

Inside the training loop, add:

```python
if epoch in checkpoint_epochs:
    checkpoints[epoch] = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
```

Then load and sample each checkpoint:

```python
for epoch, state_dict in checkpoints.items():
    checkpoint_model = TimeConditionedMLP().to(device)
    checkpoint_model.load_state_dict(state_dict)

    samples, _ = sample_reverse_process(
        model=checkpoint_model,
        parameterization="x0",
        schedule=schedule,
        n_samples=3_000,
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5)
    plt.title(f"Generated Samples after Epoch {epoch}")
    plt.axis("equal")
    plt.show()
```

This reveals whether modes form gradually or collapse during training.

---

# 26. Compare Loss Curves Carefully

```python
plt.figure(figsize=(8, 5))
plt.plot(x0_losses, label="X0 prediction")
plt.plot(noise_losses, label="Noise prediction")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
```

Do not compare raw MSE values across parameterizations as if they were directly equivalent.

The targets have different scales.

A lower numerical loss does not automatically imply better generated samples.

Evaluate:

- reverse-sample geometry,
- mode coverage,
- diversity,
- stability,
- target-normalized error.

---

# 27. Why Equivalent Parameterizations Behave Differently

Mathematically, the targets can be converted into one another when the schedule is known.

Optimization is still different because:

- target scales vary with time,
- errors are amplified differently during conversion,
- large-time and small-time examples receive different effective weights,
- network initialization interacts with target magnitude,
- finite data and finite capacity break ideal equivalence,
- numerical stability differs.

Therefore:

```text
Mathematical Convertibility

≠

Identical Training Dynamics
```

---

# 28. Tutorial Observation Versus General Rule

In the tutorial's small 2D experiments:

- sample prediction produced the best results among the demonstrated runs,
- mean prediction performed poorly in that specific setup,
- noise and score prediction were also compared,
- noise prediction was discussed as a common choice for larger diffusion systems.

These are empirical observations, not universal laws.

Results depend on:

- dataset size,
- architecture,
- schedule,
- normalization,
- loss weighting,
- optimizer,
- training duration.

---

# 29. Common Implementation Mistakes

## Mistake 1 — Using `αₜ` Instead of `ᾱₜ`

Direct noising requires cumulative alpha:

`Xₜ = √ᾱₜX₀ + √(1−ᾱₜ)ε`

---

## Mistake 2 — Forgetting Per-Sample Time Steps

A batch should usually contain randomly sampled time steps.

```python
t = torch.randint(0, T, (batch_size,), device=device)
```

---

## Mistake 3 — Omitting Time Conditioning

The same noisy point can require different denoising behavior at different noise levels.

The network must receive `t` or a noise-level embedding.

---

## Mistake 4 — Using the Wrong Conversion Formula

Noise-to-sample and sample-to-noise conversions contain different scaling factors.

Check `√ᾱₜ` and `√(1−ᾱₜ)` carefully.

---

## Mistake 5 — Adding Noise at `t=0` During Reverse Sampling

At the final step, return the mean without injecting additional Gaussian noise.

---

## Mistake 6 — Comparing Only Training Loss

A low MSE can coexist with poor generation.

Always inspect generated samples.

---

## Mistake 7 — Treating One Small Experiment as a Universal Ranking

A parameterization that works best for a small 2D dataset may not be best for a large image model.

---

# 30. Practice Exercises

## Exercise 1 — Forward Diffusion Animation

Create an animation for `t=0` through `t=199` showing the ring distribution becoming Gaussian.

Verify that the final distribution has approximately zero mean and similar variance in both axes.

---

## Exercise 2 — Train All Four Targets

Train four separate models using:

```text
x0
noise
score
mean
```

Use the same:

- architecture,
- random seed,
- number of epochs,
- optimizer,
- batch size.

Compare reverse samples rather than raw losses alone.

---

## Exercise 3 — Change Dataset Complexity

Repeat the experiment using:

- two moons,
- concentric circles,
- a spiral,
- a checkerboard mixture.

Observe whether the best-performing parameterization changes.

---

## Exercise 4 — Compare Schedules

Compare:

- linear beta schedule,
- cosine-style alpha-bar schedule,
- fewer diffusion steps,
- more diffusion steps.

Measure how quickly the structure disappears during the forward process.

---

## Exercise 5 — Time-Dependent Error

Evaluate prediction error separately for different time ranges:

```text
Early:  t ∈ [0, 49]
Middle: t ∈ [50, 149]
Late:   t ∈ [150, 199]
```

This reveals whether the model is strong only at certain noise levels.

---

## Exercise 6 — Mode Coverage

Assign each generated point to the nearest true cluster center.

Count how many samples reach each of the eight modes.

A good model should not place all samples in only one or two modes.

---

# Practical Summary

The DDPM forward process creates noisy samples using:

`Xₜ = √ᾱₜX₀ + √(1−ᾱₜ)ε`

A model can predict:

- `X₀`,
- `ε`,
- the reverse mean,
- the score.

These quantities are mathematically related, but their optimization behavior can differ substantially.

A 2D experiment makes forward diffusion, reverse denoising, mode coverage and training progression visually inspectable.

For small datasets, sample prediction may be easy to train and understand. Noise prediction remains a widely used practical target for larger diffusion systems.

---

# Key Takeaways

✅ Direct forward sampling uses cumulative `ᾱₜ`.

✅ Random time-step sampling trains one model across all noise levels.

✅ Sample, noise, mean and score prediction are convertible but not optimization-identical.

✅ Time conditioning is essential.

✅ A 2D dataset is an excellent debugging environment.

✅ Reverse snapshots reveal failures that training loss may hide.

✅ Empirical results depend on dataset scale, architecture, schedule and weighting.

✅ The best parameterization for one experiment is not automatically the best for every diffusion model.
