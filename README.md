# Using a projectional method for reconstruction

This is code accompanying the paper [A Projectional Ansatz to Reconstruction](https://arxiv.org/abs/1907.04675).

## Basic idea
We want to solve the inverse problem
$$Ax + \eta = y + \eta = y^\delta,$$
where $A:X\to Y$ a continuous linear operator,
$\eta$ some noise with $||\eta||=\delta$ and we want to reconstuct $x$ from $y^\delta$.

## Approach
Train a neural network

$$G_\theta(A, y^\delta, \delta, x_0) := P_{\overline V(A, y^\delta, \delta)} \circ g_{\theta, n-1} \circ P_{\overline V(A, y^\delta, \delta)} \circ \cdots \circ g_{\theta, 0} \circ P_{\overline V(A, y^\delta, \delta)} (0),$$

where $P_\overline V$ the projection into the set
$$\overline V(A, y^\delta, \delta):= \{x\in X:\|Ax-y^\delta\|\le \delta\}$$

over a set of samples $(x, y^\delta)$.