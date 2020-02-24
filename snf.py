import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np
import toy_data
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
import visualize_flow
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def to_numpy(x):
    return x.cpu().detach().numpy()


def sgld_mean(x, f, epsilon):
    fx = f(x)
    dfdx = utils.keep_grad(fx.sum(), x)
    x_t_mu = x + epsilon * dfdx
    return x_t_mu


def sgld_step(x, f, epsilon, beta):
    # compute x_t
    fx = f(x)
    dfdx = utils.keep_grad(fx.sum(), x)
    x_t_mu = x + epsilon * dfdx  # mean of forward distribution
    eta = torch.randn_like(x)
    x_t = x_t_mu + eta * (2 * epsilon / beta) ** .5

    fx_t = f(x_t)
    dfdx_t = utils.keep_grad(fx_t.sum(), x_t)
    x_mu = x_t + epsilon * dfdx_t  # mean of reverse distribution

    std = (2 * epsilon / beta) ** .5

    rev_dist = distributions.Normal(x_mu, std)
    for_dist = distributions.Normal(x_t_mu, std)

    rev_logprob = rev_dist.log_prob(x).sum(-1)
    for_logprob = for_dist.log_prob(x_t).sum(-1)
    return x_t, -(rev_logprob - for_logprob)


class SNF(nn.Module):
    def __init__(self, f, mu, std, n_steps=10):
        super().__init__()
        self.f = f
        self.mu = nn.Parameter(mu)
        self.logstd = nn.Parameter(std.log())
        self.logbetas = torch.zeros((n_steps,))  # dont learn temperature???
        self.logeps = nn.Parameter(torch.zeros((n_steps,)))
        self.logitlam = nn.Parameter(torch.zeros((n_steps,)) + 4)  # initialize lambda approx 1.0
        self.n_steps = n_steps

    def base_dist(self):
        return distributions.Normal(self.mu, self.logstd.exp())

    def f_lam(self, x, lam):
        bd = self.base_dist()
        fx = self.f(x)[:, None]
        logpz = bd.log_prob(x).sum(-1)[:, None]
        return fx * lam + logpz * (1 - lam)

    def forward(self, n):
        bd = self.base_dist()
        z = bd.rsample((n,)).to(device)
        x = z
        logpx = bd.log_prob(z).sum(-1)
        epss = self.logeps.exp()
        betas = self.logbetas.exp()
        lams = torch.sigmoid(self.logitlam)
        for t in range(self.n_steps):

            eps = epss[t]
            beta = betas[t]
            lam = lams[t]

            f = lambda x: self.f_lam(x, lam)
            x_t, delta_logp = sgld_step(x, f, eps, beta)
            logpx += delta_logp
            x = x_t
        return x, logpx


def gaussian_grid_2d(size=2, std=.25):
    comps = []
    for i in range(size):
        for j in range(size):
            center = np.array([i, j])
            center = torch.from_numpy(center).float()
            center = center.to(device)
            scale = (torch.ones((2,)) * std).to(device)
            comp = distributions.Normal(center, scale)
            comps.append(comp)

    pi = torch.ones((size**2,)) / (size**2)
    pi = pi.to(device)
    mog = toy_data.Mixture(comps, pi)
    return mog

def main(args):
    utils.makedirs(args.save)
    dist = gaussian_grid_2d(3, .1)

    init_batch = dist.sample(10000)
    mu = init_batch.mean(0)
    std = init_batch.std(0)
    lp_gt = dist.logprob(init_batch).mean().item()

    snf = SNF(dist.logprob, mu, std)

    optim = torch.optim.Adam(snf.parameters(), lr=args.lr)
    snf.to(device)
    for i in range(args.niters):
        x, logpx = snf(args.batch_size)
        lpx = dist.logprob(x)
        obj = lpx - logpx
        loss = -obj.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 10 == 0:
            print(i, loss.item(), dist.logprob(x).mean().item(), lp_gt)

        if i % 100 == 0:
            data = to_numpy(dist.sample(10000))
            x = to_numpy(x)
            print(data.min(), data.max(), x.min(), x.max())
            visualize_flow.visualize_transform([data, x],
                                               ["samples", "data"],
                                               [lambda x: dist.logprob(x.to(device))],
                                               ["px"], low=data.min(), high=data.max())
            plt.savefig("{}/test_{}.png".format(args.save, i))
            print(snf.logbetas.exp())
            print(snf.logeps.exp())
            print(snf.logitlam.sigmoid())
            print(snf.mu)
            print(snf.logstd.exp())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['mnist', 'fashionmnist'], type=str, default='mnist')
    parser.add_argument('--niters', type=int, default=100001)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=.5)
    parser.add_argument('--grad_l2', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--save', type=str, default='/tmp/test_snf')
    parser.add_argument('--load', type=str)
    parser.add_argument('--viz_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--base_dist', action="store_true")
    parser.add_argument('--fixed_base_dist', action="store_true")
    parser.add_argument('--k_iters', type=int, default=1)
    parser.add_argument('--e_iters', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=1000)
    parser.add_argument('--logit', action="store_true")
    parser.add_argument('--data_type', type=str, default="continuous", choices=["continuous", "binary"])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--z_dim', type=int, default=32)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--quadratic', action="store_true")
    parser.add_argument('--data_init', action="store_true")
    parser.add_argument('--full_rank_mass', action="store_true")
    parser.add_argument('--dropout', action="store_true")
    parser.add_argument('--exact_trace', action="store_true")
    parser.add_argument('--t_scaled', action="store_true")
    parser.add_argument('--both_scaled', action="store_true")
    parser.add_argument('--rbm', action="store_true")
    parser.add_argument('--resnet', action="store_true")
    parser.add_argument('--convnet', action="store_true")
    parser.add_argument('--grad_crit', action="store_true")
    parser.add_argument('--e_squared', action="store_true")
    parser.add_argument('--t_squared', action="store_true")
    parser.add_argument('--tanh', action="store_true")
    parser.add_argument('--num_const', type=float, default=1e-6)
    parser.add_argument('--burn_in', type=int, default=2000)
    parser.add_argument('--arch', default='mlp', choices=["mlp", "mlp-large",
                                                          "convnet-large", "resnet", "convnet-small"])

    args = parser.parse_args()
    main(args)