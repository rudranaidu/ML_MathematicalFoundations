# PyTorch Neural Network Training: Beginner Guide

This guide assumes you know basic Python and are new to deep learning.

Use it with the runnable project in [mnist_beginner_project.py](mnist_beginner_project.py).

## 1. The Big Picture

A neural network is a function with adjustable numbers inside it.

Traditional programming:

```text
human writes rules + input data -> computer produces answer
```

Neural network training:

```text
input data + correct answers -> computer learns useful rules
```

Real-life intuition: imagine teaching a child to recognize handwritten digits. You do not write exact rules like "if the top loop is this wide, it is an 8." You show many examples and correct mistakes. Over time, the child adjusts their internal sense of what each digit looks like. A neural network does the same, but its "internal sense" is stored as weights and biases.

Why this exists: many problems are too messy for hand-written rules. Spam detection, image recognition, speech recognition, recommendations, and fraud detection all involve patterns that are hard to describe with simple if/else logic.

Visual mental model:

```text
input features -> layer 1 -> activation -> layer 2 -> activation -> output scores
```

Each layer transforms the data. Early layers learn simple patterns. Later layers combine those simple patterns into bigger patterns.

## 2. What Is a Neural Network?

A neural network is built from neurons. A neuron takes input numbers, multiplies them by weights, adds a bias, and usually applies an activation function.

Simple neuron:

```text
z = x1*w1 + x2*w2 + b
output = activation(z)
```

House price example:

```text
size_sqft, bedrooms, location_score -> neural network -> predicted_price
```

Email spam example:

```text
word counts, sender features, link counts -> neural network -> spam probability
```

Handwritten digit example:

```text
28x28 pixel values -> neural network -> scores for digits 0 through 9
```

Cat vs dog example:

```text
image pixels -> neural network -> cat score, dog score
```

Summary: a neural network is a learnable pattern-finding machine.

## 3. Dataset Basics

| Term | Meaning | Example |
| --- | --- | --- |
| Dataset | A collection of examples. | 60,000 handwritten digit images. |
| Sample | One example from the dataset. | One image of a handwritten 7. |
| Feature | Input information used for prediction. | Pixel values, house size, email word counts. |
| Label | The correct answer for a sample. | The digit is 7. |
| Target | Another name for label, often used in PyTorch docs/code. | `labels` or `targets`. |
| Training set | Data used to update model parameters. | 55,000 MNIST images. |
| Validation set | Data used during development to check generalization. | 5,000 MNIST images. |
| Test set | Final holdout data used after training decisions are done. | 10,000 MNIST images. |

Real-life intuition: training set is homework practice, validation set is a mock exam, test set is the final exam.

Why split the data: if you only measure performance on examples the model trained on, the model may look good because it memorized. Validation and test sets check whether it learned patterns that work on new data.

Common beginner mistake: tuning your model repeatedly on the test set. That leaks final-exam information into your design decisions. Use validation for tuning and test for the final check.

## 4. Batches, Epochs, and Iterations

| Term | Meaning |
| --- | --- |
| Batch | A group of samples processed together. |
| Mini-batch | Usually what people mean by "batch" in neural network training. |
| Batch size | Number of samples in one batch. |
| Epoch | One complete pass through the training set. |
| Iteration / step | One parameter update, usually one batch. |

Example with MNIST:

```text
training samples = 60,000
batch size = 64
steps per epoch = ceil(60,000 / 64) = 938
epochs = 5
total optimizer updates = 938 * 5 = 4,690
```

Why batches exist:

- Processing one sample at a time is noisy and slow.
- Processing the full dataset at once may not fit in memory.
- Mini-batches are a practical compromise.

Memory/GPU intuition:

```text
one MNIST image shape = [1, 28, 28]
one batch of 64 images shape = [64, 1, 28, 28]
```

The GPU likes batches because it can do many matrix operations in parallel. Larger batches use more memory. Smaller batches produce noisier gradient estimates.

Common beginner mistake: thinking one epoch means one update. One epoch usually contains many updates.

## 5. Parameters vs Hyperparameters

Parameters are learned automatically.

| Parameter | Meaning | Learned by training? |
| --- | --- | --- |
| Weight | Controls how strongly one input affects the next layer. | Yes |
| Bias | Extra adjustable offset added to a neuron. | Yes |

Hyperparameters are chosen by the human before or during training.

| Hyperparameter | Meaning | Chosen by human? |
| --- | --- | --- |
| Learning rate | Size of each optimizer step. | Yes |
| Batch size | Samples per batch. | Yes |
| Epochs | Number of passes through the training set. | Yes |
| Hidden layers | Layers between input and output. | Yes |
| Number of neurons | Width of each hidden layer. | Yes |

Real-life analogy: parameters are the knobs the machine learns to turn. Hyperparameters are the settings you choose for how the learning process should run.

Common beginner mistake: expecting the model to learn the learning rate. Standard training does not do that. You choose it or use a scheduler.

## 6. Forward Pass

The forward pass is the computation from input to output.

Why "forward": information moves from the input side through the layers toward the output side.

Tiny network example:

```text
input x = [2, 3]

hidden neuron 1:
h1_raw = 0.2*2 + 0.5*3 - 0.1 = 1.8
h1 = ReLU(1.8) = 1.8

hidden neuron 2:
h2_raw = -0.3*2 + 0.8*3 + 0.2 = 2.0
h2 = ReLU(2.0) = 2.0

output class 0 logit:
z0 = 0.4*h1 - 0.6*h2 + 0.1 = -0.38

output class 1 logit:
z1 = -0.2*h1 + 0.9*h2 - 0.3 = 1.14

logits = [-0.38, 1.14]
```

The model predicts class 1 because `1.14` is larger than `-0.38`.

PyTorch implementation:

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 2),
)

x = torch.tensor([[2.0, 3.0]])
logits = model(x)
```

What PyTorch does internally:

- `nn.Linear` stores weights and biases as `nn.Parameter`.
- Calling `model(x)` calls each layer's `forward` method.
- PyTorch creates tensor operations and, if gradients are enabled, records a computation graph.
- Parameters do not change during the forward pass.

Summary: forward pass produces predictions; training has not happened yet.

## 7. Activation Functions

Without activation functions, stacking linear layers is still just one big linear function:

```text
Linear2(Linear1(x)) = another_linear_function(x)
```

That means the network cannot learn curved, complex decision boundaries. Activation functions add non-linearity.

| Activation | Formula intuition | Mental graph | Advantages | Disadvantages | Common use |
| --- | --- | --- | --- | --- | --- |
| ReLU | `max(0, x)` | Flat at 0 for negative values, diagonal line for positive values. | Fast, simple, works well in hidden layers. | Negative neurons can become inactive. | Default hidden-layer activation. |
| Sigmoid | `1 / (1 + exp(-x))` | S-shaped curve from 0 to 1. | Good for binary probabilities. | Gradients can become tiny for large positive/negative values. | Binary output probability, usually via `BCEWithLogitsLoss` instead of manual sigmoid during training. |
| Tanh | Similar to sigmoid but from -1 to 1. | S-shaped curve centered at 0. | Zero-centered output. | Can also suffer tiny gradients. | Some recurrent networks and older architectures. |
| Softmax | Converts many scores into probabilities summing to 1. | Several bars normalized into a distribution. | Good for multi-class probabilities. | Should not be applied before `CrossEntropyLoss`. | Inference after logits for multi-class classification. |

PyTorch examples:

```python
x = torch.tensor([-2.0, 0.0, 3.0])

relu = torch.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)

logits = torch.tensor([[2.0, 1.0, 0.1]])
probabilities = torch.softmax(logits, dim=1)
```

Common beginner mistakes:

- Putting `Softmax` in the model before `nn.CrossEntropyLoss`.
- Using `Sigmoid` for a 10-class single-label problem like MNIST.
- Thinking ReLU "learns." ReLU has no learned parameters. It only transforms values.

## 8. Logits

Logits are raw output scores from the final layer before probabilities.

Example:

```text
logits for digits 0, 1, 2:
[2.0, 1.0, 0.1]
```

These are not probabilities because:

- They can be negative.
- They do not need to be between 0 and 1.
- They do not sum to 1.

Why neural networks produce logits instead of probabilities:

- Logits are easier and more numerically stable for loss functions.
- `CrossEntropyLoss` combines `LogSoftmax` and negative log-likelihood internally.
- Keeping logits raw avoids duplicated softmax calculations.

Difference:

```text
logits        = raw scores
probabilities = normalized confidence values
prediction    = class with largest score/probability
```

Small numerical example:

```text
logits = [2.0, 1.0, 0.1]
exp    = [7.389, 2.718, 1.105]
sum    = 11.212
softmax probabilities:
[7.389/11.212, 2.718/11.212, 1.105/11.212]
= [0.659, 0.242, 0.099]
```

Summary: logits are model scores; softmax turns them into probabilities.

## 9. Softmax and Probability Distributions

A probability distribution assigns probabilities to possible outcomes.

For digit classification:

```text
P(0)=0.01
P(1)=0.02
P(2)=0.03
P(3)=0.01
P(4)=0.01
P(5)=0.04
P(6)=0.01
P(7)=0.82
P(8)=0.03
P(9)=0.02
sum = 1.00
```

Why probabilities sum to one: for one handwritten digit, the model is choosing among mutually exclusive classes. The digit should be one of 0 through 9, so total probability mass is distributed across those classes.

PyTorch:

```python
logits = model(images)
probabilities = torch.softmax(logits, dim=1)
predictions = probabilities.argmax(dim=1)
```

Use softmax for interpreting predictions after the model produces logits. For training with `nn.CrossEntropyLoss`, pass logits directly.

## 10. Loss Functions

A loss function measures how wrong the model is.

Real-life intuition: if the model is a student, the loss is the grade penalty. A smaller loss means the answer is closer to correct.

Why loss exists: neural networks need one number that says "how bad was this prediction?" Backpropagation starts from this number.

### MSE Loss

Mean Squared Error is common for regression.

```text
prediction = 220,000
target     = 200,000
error      = 20,000
squared error = 400,000,000
```

Use for predicting continuous numbers: house price, temperature, sales amount.

PyTorch:

```python
criterion = nn.MSELoss()
loss = criterion(predicted_prices, true_prices)
```

### BCE Loss

Binary Cross Entropy is common for binary classification.

Examples:

- Spam vs not spam.
- Disease vs no disease.
- Fraud vs not fraud.

Best PyTorch practice is usually `nn.BCEWithLogitsLoss()`, which expects raw logits and applies sigmoid internally in a stable way.

```python
criterion = nn.BCEWithLogitsLoss()
loss = criterion(binary_logits, binary_labels.float())
```

### CrossEntropy Loss

CrossEntropy is common for single-label multi-class classification.

Examples:

- MNIST digit 0 through 9.
- Cat, dog, horse, car, etc.
- One correct class per sample.

If the true class probability is high:

```text
true class probability = 0.70
loss = -log(0.70) = 0.357
```

If the true class probability is low:

```text
true class probability = 0.10
loss = -log(0.10) = 2.303
```

PyTorch:

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
```

Important: `labels` should usually be integer class indices like `[7, 2, 0]`, not one-hot vectors, for ordinary classification.

Common beginner mistakes:

- Passing softmax probabilities into `CrossEntropyLoss`.
- Using `MSELoss` for class labels like digit 7.
- Keeping labels as shape `[batch_size, 1]` instead of `[batch_size]` for `CrossEntropyLoss`.

## 11. Gradients and Gradient Descent

A derivative tells you the slope of a function at one point.

For one variable:

```text
positive derivative -> function increases if x increases
negative derivative -> function decreases if x increases
zero derivative     -> flat point
```

A gradient is the multi-variable version of a derivative. Neural networks have many parameters, so the gradient tells us how each weight and bias affects the loss.

Real-life intuition: walking downhill while blindfolded.

- Your height above sea level is the loss.
- Your position is the model parameters.
- The slope under your feet is the gradient.
- You step in the downhill direction to reduce loss.

Important detail: the gradient points toward steepest increase. To go downhill, gradient descent subtracts the gradient.

```text
new_weight = old_weight - learning_rate * gradient
```

Learning rate:

- Too small: training is slow.
- Too large: training may jump around or diverge.
- Reasonable: loss tends to decrease.

Local vs global minimum:

- Global minimum: best possible lowest loss.
- Local minimum: lower than nearby points but not necessarily best overall.

In modern neural networks, we often care more about finding a useful low-loss region than a mathematically perfect global minimum.

## 12. Backpropagation

After computing loss, the model needs to know which parameters caused the error.

Backpropagation computes gradients by moving backward through the computation graph.

Chain rule intuition:

```text
if loss depends on prediction
and prediction depends on weight
then loss depends on weight
```

Tiny math example:

```text
y = w*x + b
loss = (y - target)^2

d loss / d y = 2*(y - target)
d y / d w = x
d loss / d w = d loss / d y * d y / d w
```

PyTorch:

```python
loss.backward()
```

What PyTorch computes automatically:

- It starts at the scalar `loss`.
- It walks backward through recorded tensor operations.
- It applies derivatives for each operation.
- It accumulates results into each parameter's `.grad`.

What changes during `loss.backward()`:

- Parameter values do not change yet.
- Parameter `.grad` tensors change.
- Intermediate activations may be freed after backward unless the graph is retained.

Common beginner mistake: thinking `backward()` updates weights. It does not. The optimizer updates weights.

## 13. Optimizers

An optimizer decides how to use gradients to update parameters.

| Optimizer | Intuition | Advantages | Disadvantages | When to use |
| --- | --- | --- | --- | --- |
| Gradient Descent | Use the full dataset gradient each update. | Stable direction. | Expensive for large datasets. | Mostly for teaching. |
| SGD | Use mini-batch gradients. | Simple, often generalizes well. | Can be noisy and slower. | Strong baseline, especially with momentum. |
| Momentum | Roll a ball downhill with memory of past direction. | Reduces zig-zagging. | Adds another hyperparameter. | Vision models, SGD training. |
| RMSProp | Scale updates by recent gradient sizes. | Helps with uneven gradient scales. | Less common than Adam today. | Some sequence/reinforcement learning settings. |
| Adam | Momentum plus adaptive step sizes. | Beginner-friendly, often works quickly. | Can generalize worse than SGD in some settings. | Good default for many beginner projects. |

PyTorch examples:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Summary: gradients say "which way"; optimizers decide "how far and with what strategy."

## 14. PyTorch Basics

| PyTorch concept | What it does | What happens internally |
| --- | --- | --- |
| Tensor | N-dimensional array, like NumPy with GPU/autograd support. | Stores data, shape, dtype, device, and sometimes graph info. |
| Autograd | Automatic differentiation engine. | Records operations and computes gradients during `backward()`. |
| `nn.Module` | Base class for models/layers. | Registers parameters, submodules, buffers, and train/eval mode. |
| Dataset | Object that returns one sample at a time. | Implements `__len__` and `__getitem__`. |
| DataLoader | Creates batches from a Dataset. | Handles batching, shuffling, optional worker processes, memory pinning. |
| Model | Your neural network. | Runs forward computation and owns parameters. |
| Loss | Scalar error measure. | Builds the final part of the computation graph. |
| Optimizer | Updates parameters. | Reads `.grad` values and modifies parameter tensors. |

Minimal tensor example:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)  # 4.0 because d(x^2)/dx = 2x
```

## 15. Training Loop: Line by Line

Canonical loop:

```python
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

Line explanations:

| Line | Why needed | What changes |
| --- | --- | --- |
| `for epoch in range(num_epochs):` | Repeats training over the dataset multiple times. | Nothing by itself. |
| `for inputs, labels in train_loader:` | Gets one mini-batch. | New batch tensors are loaded. |
| `optimizer.zero_grad()` | Clears old gradients. | Parameter `.grad` buffers are cleared. |
| `outputs = model(inputs)` | Runs forward pass. | Output tensor and computation graph are created. Parameters do not change. |
| `loss = criterion(outputs, labels)` | Measures prediction error. | Loss tensor is created. Parameters do not change. |
| `loss.backward()` | Computes gradients. | Parameter `.grad` buffers are filled/accumulated. Parameter values do not change. |
| `optimizer.step()` | Updates weights and biases. | Parameter values change. Inputs and labels do not change. |

Beginner detail: PyTorch accumulates gradients by default. If you forget `zero_grad()`, gradients from previous batches add into the current batch's gradients.

## 16. `model.train()`, `model.eval()`, and `torch.no_grad()`

`model.train()`:

- Sets the model to training mode.
- Affects layers like Dropout and BatchNorm.
- Does not itself train anything. It only changes mode flags.

`model.eval()`:

- Sets the model to evaluation mode.
- Dropout stops randomly dropping neurons.
- BatchNorm uses stored running statistics.
- It does not disable gradient tracking by itself.

`torch.no_grad()`:

- Temporarily tells PyTorch not to record operations for autograd.
- Saves memory and compute during validation/testing.
- Does not change Dropout/BatchNorm behavior by itself.

Correct evaluation pattern:

```python
model.eval()
with torch.no_grad():
    logits = model(images)
```

Common beginner mistake: using only `model.eval()` and assuming gradients are disabled. They are separate controls.

## 17. Complete Training Workflow

```text
Dataset
  -> DataLoader
  -> Batch
  -> Model
  -> Forward pass
  -> Logits
  -> Activation if needed for interpretation
  -> Loss
  -> Backward pass
  -> Gradients
  -> Optimizer
  -> Updated parameters
  -> Next batch
  -> Repeat until all batches finish
  -> One epoch complete
  -> Repeat epochs
  -> Final trained model
```

For MNIST with `CrossEntropyLoss`, the training path is:

```text
image batch -> model -> logits -> CrossEntropyLoss -> backward -> optimizer step
```

Softmax is used for human-readable probabilities after training or during reporting, not before `CrossEntropyLoss`.

## 18. Complete Project: MNIST Digit Classification

File: [mnist_beginner_project.py](mnist_beginner_project.py)

Goal: train a small neural network to classify handwritten digits from 0 to 9.

### Folder structure

After running the project, expect:

```text
otbi-lcm-jc/
  mnist_beginner_project.py
  pytorch_neural_networks_beginner_guide.md
  data/
    MNIST/
  mnist_mlp.pt
```

### Install and run

Use the official PyTorch install selector if `torch` and `torchvision` are not installed.

Typical CPU install command:

```bash
python3 -m pip install torch torchvision
```

Run:

```bash
python3 mnist_beginner_project.py
```

Expected output shape:

```text
Using device: cpu
epoch 1/5 | train loss ... | train acc ... | val loss ... | val acc ...
...
test loss ... | test acc ...
Saved model weights to mnist_mlp.pt
Sample predictions:
image 0: label=7, prediction=7, confidence=...
```

Exact numbers vary by hardware, PyTorch version, random seed behavior, and downloaded data order.

### Model architecture

```text
input image shape: [batch_size, 1, 28, 28]
flattened shape:   [batch_size, 784]
linear layer:      784 -> 128
ReLU
linear layer:      128 -> 64
ReLU
linear layer:      64 -> 10
output:            10 logits, one for each digit
```

### Script walkthrough

Blank lines are only spacing. Multi-line Python statements are grouped when they form one logical command.

| Lines | Explanation |
| --- | --- |
| 1 | Imports `Path` so file/folder paths can be represented cleanly. |
| 3 | Imports the main PyTorch package. |
| 4 | Imports `torch.nn` as `nn`, which contains layers and loss functions. |
| 5 | Imports `DataLoader` for batches and `random_split` for train/validation splitting. |
| 6 | Imports torchvision datasets and transforms. MNIST comes from `datasets.MNIST`. |
| 9 | Sets a seed to make random choices more repeatable. |
| 10 | Sets batch size to 64 samples per training step. |
| 11 | Sets training to 5 epochs. |
| 12 | Sets Adam's learning rate to `0.001`. |
| 13 | Sets the data download folder to `data`. |
| 14 | Sets the saved model weight file to `mnist_mlp.pt`. |
| 17 | Defines `get_device()`, a helper that chooses available hardware. |
| 18-19 | Uses CUDA if an NVIDIA GPU is available. |
| 20-21 | Uses Apple Silicon MPS if available. |
| 22 | Falls back to CPU. |
| 25 | Defines `DigitMLP`, our neural network class. It inherits from `nn.Module`. |
| 26 | Defines the constructor, where layers are created. |
| 27 | Calls the parent `nn.Module` constructor so PyTorch can register layers correctly. |
| 28 | Creates a flatten layer to convert `[batch, 1, 28, 28]` into `[batch, 784]`. |
| 29-35 | Creates the hidden and output layers in order. The final layer returns 10 logits. |
| 30 | First linear layer maps 784 pixel inputs to 128 hidden values. |
| 31 | ReLU adds non-linearity after the first linear layer. |
| 32 | Second linear layer maps 128 values to 64 values. |
| 33 | Another ReLU adds more non-linearity. |
| 34 | Final linear layer maps 64 values to 10 digit logits. |
| 37 | Defines the forward pass. PyTorch calls this when you call `model(images)`. |
| 38 | Flattens image tensors. |
| 39 | Sends flattened images through all layers. |
| 40 | Returns raw logits, not probabilities. |
| 43 | Defines a helper to create train, validation, and test loaders. |
| 44-49 | Builds preprocessing transforms. |
| 46 | Converts PIL images to PyTorch tensors with values roughly between 0 and 1. |
| 47 | Normalizes MNIST using common mean and standard deviation values. |
| 51-56 | Downloads/loads the training portion of MNIST. |
| 57-62 | Downloads/loads the test portion of MNIST. |
| 64 | Creates a seeded generator for reproducible train/validation split. |
| 65-69 | Splits 60,000 training images into 55,000 training and 5,000 validation samples. |
| 71 | Enables pinned memory only for CUDA, which can speed CPU-to-GPU transfers. |
| 72-77 | Creates the training DataLoader. `shuffle=True` changes batch order each epoch. |
| 78-83 | Creates the validation DataLoader. `shuffle=False` keeps evaluation stable. |
| 84-89 | Creates the test DataLoader. `shuffle=False` is standard for final evaluation. |
| 90 | Returns the three loaders. |
| 93 | Defines one epoch of training. |
| 94 | Puts the model in training mode. |
| 95-97 | Initializes counters for loss, correct predictions, and sample count. |
| 99 | Iterates over mini-batches from the training loader. |
| 100-101 | Moves images and labels to the selected device. |
| 103 | Clears old gradients before the new backward pass. |
| 104 | Runs the forward pass and gets logits. |
| 105 | Computes classification loss from logits and integer labels. |
| 106 | Backpropagates and fills parameter `.grad` tensors. |
| 107 | Updates model parameters using the optimizer. |
| 109 | Reads the current batch size. The final batch can be smaller than 64. |
| 110 | Adds this batch's total loss contribution. |
| 111 | Chooses the largest logit as the predicted class. |
| 112 | Counts correct predictions in the batch. |
| 113 | Adds this batch size to the running total. |
| 115 | Converts accumulated loss into average loss per sample. |
| 116 | Computes accuracy as correct divided by total. |
| 117 | Returns training metrics for this epoch. |
| 120 | Defines evaluation for validation or test data. |
| 121 | Puts the model in evaluation mode. |
| 122-124 | Initializes evaluation counters. |
| 126 | Disables gradient tracking for faster, lighter evaluation. |
| 127 | Iterates over evaluation batches. |
| 128-129 | Moves batch tensors to the selected device. |
| 131 | Runs the forward pass. |
| 132 | Computes loss for reporting only. |
| 134-138 | Updates loss, prediction, correct-count, and total-count metrics. |
| 140-142 | Returns average evaluation loss and accuracy. |
| 145 | Defines a helper to print example predictions. |
| 146 | Uses evaluation mode for prediction. |
| 147 | Gets one batch from the test loader. |
| 148 | Moves images to the selected device. |
| 150 | Disables gradient tracking. |
| 151 | Gets logits from the model. |
| 152 | Converts logits to probabilities for human-readable confidence. |
| 153 | Chooses predicted classes from the probabilities. |
| 154 | Extracts the highest probability as confidence. |
| 156-158 | Moves labels, predictions, and confidence values back to CPU for printing. |
| 160-167 | Prints 10 sample predictions. |
| 170 | Defines the main program entry function. |
| 171 | Sets PyTorch's random seed. |
| 172 | Chooses CPU/GPU/MPS device. |
| 173 | Prints the selected device. |
| 175 | Builds the DataLoaders. |
| 176 | Creates the model and moves it to the selected device. |
| 177 | Creates multi-class classification loss. |
| 178 | Creates Adam optimizer over all model parameters. |
| 180 | Starts the epoch loop. |
| 181-187 | Runs one training epoch. |
| 188 | Evaluates on the validation set. |
| 190-196 | Prints training and validation metrics. |
| 198 | Evaluates on the test set after training is finished. |
| 199 | Prints final test metrics. |
| 201 | Saves learned weights. |
| 202 | Prints the save path. |
| 204 | Prints sample predictions. |
| 207-208 | Runs `main()` only when this file is executed directly. |

### What changes during training

| Object | During forward | During backward | During optimizer step |
| --- | --- | --- | --- |
| Images | Read only | Read only | Unchanged |
| Labels | Read only | Read only | Unchanged |
| Logits | Created | Used for gradients | Discarded after loop moves on |
| Loss | Created | Starting point for backprop | Discarded after metrics |
| Parameter `.grad` | Usually unchanged until backward | Filled with gradients | Usually kept until next `zero_grad` |
| Parameter values | Unchanged | Unchanged | Updated |

## 19. Common Beginner Debugging Checklist

- Loss is `nan`: check learning rate, invalid operations, bad normalization, or labels in the wrong format.
- Accuracy stays random: check that labels match data, model output size matches class count, and optimizer receives `model.parameters()`.
- `CrossEntropyLoss` shape error: logits should usually be `[batch_size, num_classes]`; labels should be `[batch_size]`.
- Model does not improve: try lower learning rate, more epochs, simpler model, or verify data preprocessing.
- CUDA/MPS device error: make sure model and tensors are on the same device.
- Validation accuracy much lower than training accuracy: possible overfitting; use more data, augmentation, regularization, or smaller model.
- Very slow training: use mini-batches, use available GPU/MPS, avoid unnecessary Python work inside the batch loop.

## 20. Practice Exercises

### Exercise 1: Tensor and Autograd Warm-up

Objective: understand tensors, `requires_grad`, and `.backward()`.

Concepts reinforced: tensor creation, scalar loss, gradients, gradient accumulation.

Dataset: no external dataset. Use synthetic numbers.

Source: generated in Python.

Folder structure:

```text
exercise_01_autograd/
  autograd_practice.py
```

Steps:

1. Create `x = torch.tensor([2.0], requires_grad=True)`.
2. Compute `y = x*x + 3*x + 1`.
3. Call `y.backward()`.
4. Print `x.grad`.
5. Repeat without clearing gradients and observe accumulation.

Expected output: gradient should be `7.0` for `x=2` because derivative of `x^2 + 3x + 1` is `2x + 3`.

Common errors: calling `.backward()` on a non-scalar tensor without passing a gradient; forgetting that gradients accumulate.

Challenge: manually calculate gradients for `y = 4*x*x - 2*x`.

### Exercise 2: Iris Classification

Objective: classify iris flower species from four numeric features.

Concepts reinforced: tabular data, train/test split, `CrossEntropyLoss`, small MLP.

Dataset: Iris, 150 samples, 4 features, 3 classes.

Source: `sklearn.datasets.load_iris()` at https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html or UCI Machine Learning Repository.

Folder structure:

```text
exercise_02_iris/
  train_iris.py
```

Steps:

1. Load Iris data.
2. Standardize features.
3. Split into train/validation/test.
4. Build `nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))`.
5. Train with `CrossEntropyLoss`.
6. Print test accuracy.

Expected output: often above 90 percent accuracy.

Common errors: using one-hot labels with ordinary `CrossEntropyLoss`; forgetting to convert labels to `torch.long`.

Challenge: compare 0, 1, and 2 hidden layers.

### Exercise 3: Wine Classification

Objective: classify wine cultivar using chemical measurements.

Concepts reinforced: feature scaling, multi-class classification, validation monitoring.

Dataset: Wine, 178 samples, 13 features, 3 classes.

Source: `sklearn.datasets.load_wine()` at https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html or UCI Machine Learning Repository.

Folder structure:

```text
exercise_03_wine/
  train_wine.py
```

Steps:

1. Load the dataset.
2. Standardize every feature.
3. Build a small MLP with input size 13 and output size 3.
4. Train for 100 to 300 epochs.
5. Track training and validation loss.

Expected output: high validation accuracy if scaling is correct.

Common errors: training without scaling; using too large a learning rate on a tiny dataset.

Challenge: plot loss curves.

### Exercise 4: Breast Cancer Binary Classification

Objective: predict benign vs malignant diagnosis.

Concepts reinforced: binary classification, logits, `BCEWithLogitsLoss`.

Dataset: Breast Cancer Wisconsin, 30 numeric features, 2 classes.

Source: `sklearn.datasets.load_breast_cancer()` at https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html or UCI Machine Learning Repository.

Folder structure:

```text
exercise_04_breast_cancer/
  train_breast_cancer.py
```

Steps:

1. Load and standardize features.
2. Convert labels to float tensors shaped `[batch_size, 1]`.
3. Build model output with one logit.
4. Use `nn.BCEWithLogitsLoss()`.
5. Convert logits to probabilities with `torch.sigmoid()` for reporting.

Expected output: strong accuracy, often above 90 percent.

Common errors: using `CrossEntropyLoss` with one output neuron; applying sigmoid before `BCEWithLogitsLoss`.

Challenge: compute precision and recall.

### Exercise 5: California Housing Regression

Objective: predict house values from tabular features.

Concepts reinforced: regression, `MSELoss`, output size 1.

Dataset: California Housing.

Source: `sklearn.datasets.fetch_california_housing()` at https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html.

Folder structure:

```text
exercise_05_california_housing/
  train_housing.py
```

Steps:

1. Load data.
2. Standardize features and target.
3. Build a model ending in `nn.Linear(hidden_size, 1)`.
4. Train with `nn.MSELoss()`.
5. Report mean absolute error after converting predictions back to original scale.

Expected output: decreasing MSE over epochs.

Common errors: treating regression target as class labels; forgetting target shape `[batch_size, 1]`.

Challenge: compare MSE and MAE as reporting metrics.

### Exercise 6: FashionMNIST Classification

Objective: classify clothing images.

Concepts reinforced: image tensors, DataLoader, same pipeline as MNIST.

Dataset: FashionMNIST, 10 clothing classes, 28x28 grayscale images.

Source: `torchvision.datasets.FashionMNIST` at https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html.

Folder structure:

```text
exercise_06_fashionmnist/
  train_fashionmnist.py
  data/
```

Steps:

1. Copy the MNIST project.
2. Replace `datasets.MNIST` with `datasets.FashionMNIST`.
3. Keep output size 10.
4. Train and compare accuracy with MNIST.

Expected output: lower accuracy than MNIST because clothing classes are more visually similar.

Common errors: assuming MNIST-level accuracy is guaranteed; forgetting to update class names for printing.

Challenge: add dropout and compare validation accuracy.

### Exercise 7: CIFAR-10 CNN

Objective: classify color images using a convolutional neural network.

Concepts reinforced: channels, convolution, pooling, image classification.

Dataset: CIFAR-10, 10 classes, 32x32 RGB images.

Source: `torchvision.datasets.CIFAR10` at https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html.

Folder structure:

```text
exercise_07_cifar10/
  train_cifar10_cnn.py
  data/
```

Steps:

1. Load CIFAR-10 with `transforms.ToTensor()`.
2. Build a CNN using `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, and `nn.Linear`.
3. Train with `CrossEntropyLoss`.
4. Track train and validation accuracy.

Expected output: a simple CNN should beat random guessing by a large margin.

Common errors: flattening to the wrong size before the final linear layer; forgetting RGB input has 3 channels.

Challenge: add data augmentation with random crop and horizontal flip.

### Exercise 8: Titanic Binary Classification

Objective: predict passenger survival from tabular features.

Concepts reinforced: real-world data cleaning, categorical encoding, binary classification.

Dataset: Titanic.

Source: Kaggle Titanic competition at https://www.kaggle.com/c/titanic/data.

Folder structure:

```text
exercise_08_titanic/
  data/
    train.csv
    test.csv
  train_titanic.py
```

Steps:

1. Download CSVs from Kaggle.
2. Select features such as passenger class, sex, age, siblings/spouses, parents/children, fare, and embarkation port.
3. Fill missing values.
4. One-hot encode categorical features.
5. Standardize numeric features.
6. Train a binary classifier with `BCEWithLogitsLoss`.

Expected output: validation accuracy better than simple guessing.

Common errors: leaking target column into features; failing to apply the same preprocessing to train and test data.

Challenge: compare neural network results to logistic regression.

## 21. Best Practices to Build Confidence

- Start with a tiny model that can run quickly.
- Print tensor shapes often.
- Make sure loss decreases on a small subset before training on the full dataset.
- Use logits for training losses and probabilities for human interpretation.
- Keep train, validation, and test responsibilities separate.
- Save model weights only after you know what metric they represent.
- Change one hyperparameter at a time when learning.

## 22. Quick Summary

Neural network training is a repeated feedback loop:

```text
predict -> measure error -> compute gradients -> update parameters -> repeat
```

The forward pass produces logits. The loss turns prediction error into one scalar. Backpropagation computes gradients. The optimizer updates weights and biases. Batches make training practical. Epochs repeat learning over the dataset. Evaluation checks whether the model learned useful patterns instead of memorizing.

## Sources Checked

- PyTorch `CrossEntropyLoss`: https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html
- PyTorch autograd mechanics: https://docs.pytorch.org/docs/2.12/notes/autograd.html
- PyTorch `nn.Module` train/eval behavior: https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html
- Torchvision MNIST dataset: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
- Torchvision FashionMNIST dataset: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
- Torchvision CIFAR-10 dataset: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
- Scikit-learn dataset loaders: https://scikit-learn.org/stable/datasets.html
- Kaggle Titanic data: https://www.kaggle.com/c/titanic/data
