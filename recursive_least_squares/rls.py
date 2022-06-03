import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import typing as t


class RLS(object):
    def __init__(self, lmbda=0.3, alpha=1e-3):
        self.lmbda = lmbda
        self.alpha = alpha  # learning rate
        self.H = None

    def update(self, y, yhat, thetas: t.List[torch.Tensor], verbose=False):
        # should be equal to x b/c loss function is squared L2 norm / 2
        # NOTE: we take gradient wrt yhat, NOT Loss = squared error
        yhat = yhat.flatten()
        y = y.flatten()
        length = y.shape[0]

        # Get gradient of thetas wrt outputs (ie: Jacobian)
        # Reference: https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments/47026836
        # More efficient way? https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
        all_Gs = []
        from tqdm import tqdm
        for i in tqdm(range(length)):
            dL_dyhat = torch.nn.functional.one_hot(
                torch.tensor(i), num_classes=length).float().to(y.device)
            G = autograd.grad(yhat, thetas, dL_dyhat,
                              allow_unused=True, retain_graph=True)
            all_Gs.append(G)

        orig_grads = None
        if verbose:
            debug_loss = 0.5 * (y - yhat).pow(2).mean()
            debug_loss.backward(retain_graph=True)
            orig_grads = [theta.grad.clone() for theta in thetas]

        with torch.no_grad():
            # T x num_thetas
            # https://stackoverflow.com/questions/59149275/stacking-tensors-in-a-list-of-tuples-of-tensors
            all_Gs = torch.hstack(tuple(map(torch.stack, zip(*all_Gs))))

            e = (y - yhat).detach()
            if self.H is None:
                self.H = torch.eye(len(thetas), device=y.device)
            self.H = all_Gs.T @ all_Gs + self.lmbda * self.H
            dthetas = torch.linalg.pinv(self.H) @ all_Gs.T @ e
            for i in range(len(thetas)):
                thetas[i].data = thetas[i].data + self.alpha * dthetas[i]

        if verbose:
            debug_loss = (y - yhat).pow(2).sum()
            debug_loss.backward()
            for i, theta in enumerate(thetas):
                print("Original Grad: %.3f, RLS Grad: %.3f, New P: %.3f" % (
                    -orig_grads[i].item(), dthetas[i].item(), theta.item()))


def second_order_grad_example():
    # NOTE: THIS IS NOT NEEDED FOR RLS
    # 	We only approximate the Hessian, never directly calculate it
    x = torch.tensor(1., requires_grad=True)
    y = 2 * x ** 3 + 5 * x ** 2 + 8
    y.backward(retain_graph=True, create_graph=True)
    loss = 8 - y

    # dL/dx
    # correct answer: -16 = -(6*x**2 + 10*x)
    first_derivative = autograd.grad(loss, x, create_graph=True)[0]

    # d^2L/dx^2
    # correct answer: -22 = -(12*x + 10)
    second_derivative = autograd.grad(first_derivative, x)[0]


def rls_example_1D():
    """
    Edge cases:
        - be careful with lambda = 0. If G0 == 0, then H1 = G0*G0 + lambda*H0 = 0, inv(H1) will be undef
        - be carefeul with H0 = 0 for similar reason.

    """
    # initial parameters and function f(x)
    # NOTE: should not be 0 initially if f = (theta^k)*x with k > 1, otherwise df/dtheta = k*(theta^(k-1))*x)) = 0 bc theta = 0
    theta = torch.tensor([0.1], requires_grad=True)

    # fhat is estimated function, f is true/unknown function to match
    # def fhat(x, theta): return (theta ** 1) * x
    def fhat(x, theta): return theta ** 2 * x ** 2
    def f(x): return x ** 2

    alpha = 0.3  # learning rate
    lmbda = 1.0  # forgetting factor
    rls = RLS(lmbda=lmbda, alpha=alpha)

    N = 10  # number of data points
    extra = 5  # visualize future predictions
    xs = torch.arange(0, N + extra, dtype=torch.float32)
    ys = f(xs)  # ground truth to fit
    prediction_rollouts = []
    thetas = []

    for i in range(N):
        x, y = xs[i], ys[i]
        yhat = fhat(x, theta)
        rls.update(y, yhat, thetas=[theta])

        # Visualize predictions after each update
        prediction_rollouts.append(
            (xs[i:], fhat(xs[i:], theta).detach().numpy()))
        thetas.append(theta.item())

    # Plotting
    plt.plot(xs, ys, label="Ground Truth", color="black", linewidth=2)
    plt.title("RLS(lambda=%.2f, alpha=%.2f)" % (lmbda, alpha))
    colors = cm.rainbow(np.linspace(0, 1, N))
    for i in range(N-1):
        x_vals = prediction_rollouts[i][0]
        yhats = prediction_rollouts[i][1]
        plt.plot(x_vals, yhats, label="step %d, theta=%.3f" %
                 (i, thetas[i]), color=colors[i])
        plt.scatter([x_vals[0]], [yhats[0]], color=colors[i])

    plt.legend()
    plt.show()


if __name__ == "__main__":
    rls_example_1D()
