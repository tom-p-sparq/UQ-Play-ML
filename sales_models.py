import numpy as np
import copy
import scipy.stats as stats
from scipy.special import logsumexp
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt
import wquantiles
import ipywidgets as widgets


def decaying_sigma(x):
    return 1 / (1 + np.exp(x))


class SalesModel:
    def __init__(self, *, mu_N, L_half, elasticity):
        self.mu_N = mu_N
        self.L_half = L_half
        self.elasticity = elasticity
        return

    def alpha(self):
        return -2 * self.elasticity / self.L_half

    def parameters(self):
        return [self.mu_N, self.L_half, self.elasticity]

    # purchase probability is deterministic
    def purchase_probability(self, price):
        return decaying_sigma(self.alpha() * (price - self.L_half))

    # elasticity is deterministic
    def price_elasticity_of_demand(self, price):
        return -self.alpha() * price * (1 - self.purchase_probability(price))

    def optimal_price(self, cost=0):
        def grad_profit(price):
            return 1 - self.alpha() * (price - cost) * (1 - self.purchase_probability(price))
        root = fsolve(grad_profit, self.L_half)
        return root[0]

    # OUTPUTS
    #
    # sales numbers are random

    def sales_distribution(self, price):
        mu = self.expected_sales(price)
        return stats.poisson(mu=mu)

    def expected_sales(self, price):
        p = self.purchase_probability(price)
        return p * self.mu_N

    def std_sales(self, price):
        mu = self.expected_sales(price)
        return np.sqrt(mu)

    def quantile_sales(self, price, quantile):
        d = self.sales_distribution(price)
        return d.ppf(quantile)

    def log_likelihoods(self, prices, sales):
        return np.array([self.sales_distribution(L).logpmf(S) for (L, S) in zip(prices, sales)])

    def log_likelihood(self, prices, sales):
        return sum(self.log_likelihoods(prices, sales))

    # profit numbers are random

    def expected_profit(self, price, cost):
        return (price - cost) * self.expected_sales(price)

    def std_profit(self, price, cost):
        return (price - cost) * self.std_sales(price)

    def quantile_profit(self, price, cost, quantile):
        return (price - cost) * self.quantile_sales(price, quantile)

    # revenue numbers are random

    def expected_revenue(self, price):
        return self.expected_profit(price, 0)

    def std_revenue(self, price):
        return self.std_profit(price, 0)

    def quantile_revenue(self, price, quantile):
        return self.quantile_profit(price, 0, quantile)

    def show_profits(self, costs, *, price_decisions=None, **kwargs):
        prices = np.arange(101).reshape(-1, 1)
        data = self.expected_profit(prices, np.array(costs).reshape(1, -1))
        fig = plt.figure()
        plt.plot(prices, data, **kwargs)
        plt.gca().set_ylim((0, None))
        if price_decisions is not None:
            y0, y1 = plt.gca().get_ylim()
            plt.vlines(price_decisions, y0, y1, color='black', linestyles='dotted')
        plt.legend()
        plt.gca().set_xlabel('price')
        plt.gca().set_ylabel('profit')
        return fig

    def show(self):
        prices = np.arange(101)
        revmax_price = self.revmax_price()
        revmax = self.expected_revenue(revmax_price)

        probs = self.purchase_probability(prices)
        elasticities = self.price_elasticity_of_demand(prices)

        expected_sales = self.expected_sales(prices)
        lq_sales = self.min_sales(prices, alpha=0.25)
        uq_sales = self.max_sales(prices, alpha=0.25)
        min_sales = self.min_sales(prices, alpha=0.05)
        max_sales = self.max_sales(prices, alpha=0.05)

        expected_revenue = self.expected_revenue(prices)
        lq_revenue = self.min_revenue(prices, alpha=0.25)
        uq_revenue = self.max_revenue(prices, alpha=0.25)
        min_revenue = self.min_revenue(prices, alpha=0.05)
        max_revenue = self.max_revenue(prices, alpha=0.05)

        plt.ioff()
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(10, 8)

        axs[0, 0].plot(prices, probs, color='blue')
        axs[0, 0].set_ylabel('probability')
        axs[0, 0].set_ylim((-0.05, 1.05))
        axs[0, 0].set_title('Purchase probability')
        axs[0, 0].set_xlabel('price')

        axs[0, 1].plot(prices, elasticities, color='orange')
        axs[0, 1].vlines(revmax_price, -5, 0, linestyles='dotted', color='black')
        axs[0, 1].hlines(-1, 0, 100, linestyles='dotted', color='black')
        axs[0, 1].set_ylabel('elasticity')
        axs[0, 1].yaxis.tick_right()
        axs[0, 1].set_ylim((-5.25, 0.25))
        axs[0, 1].set_title('Elasticity of demand')
        axs[0, 1].set_xlabel('price')

        axs[1, 0].plot(prices, expected_sales, color='green')
        axs[1, 0].fill_between(prices, lq_sales, uq_sales, alpha=0.2, color='green')
        axs[1, 0].fill_between(prices, min_sales, max_sales, alpha=0.1, color='green')
        axs[1, 0].set_ylabel('sales')
        axs[1, 0].set_ylim((-60, 1260))
        axs[1, 0].set_title('Random realised sales')
        axs[1, 0].set_xlabel('price')

        axs[1, 1].plot(prices, expected_revenue, color='red')
        axs[1, 1].fill_between(prices, lq_revenue, uq_revenue, alpha=0.2, color='red')
        axs[1, 1].fill_between(prices, min_revenue, max_revenue, alpha=0.1, color='red')
        axs[1, 1].vlines(revmax_price, 0, 50_000, linestyles='dotted', color='black')
        axs[1, 1].hlines(revmax, 0, 100, linestyles='dotted', color='black')
        axs[1, 1].set_ylabel('revenue')
        axs[1, 1].yaxis.tick_right()
        axs[1, 1].set_ylim((-1000, 51000))
        axs[1, 1].set_title('Random realised revenue')
        axs[1, 1].set_xlabel('price')

        plt.tight_layout()

        return fig


class SalesModelDistribution:
    def __init__(self, *, mu_N, L_half, elasticity):
        self.mu_N = mu_N
        self.L_half = L_half
        self.elasticity = elasticity
        self.q = stats.qmc.Sobol(3, seed=0)
        return

    def sample(self, N):
        probs = self.q.random_base2(N)
        mu_Ns = self.mu_N.ppf(probs[:, 0])
        L_halfs = self.L_half.ppf(probs[:, 1])
        elasticitys = self.elasticity.ppf(probs[:, 2])
        self.q.reset()
        return SalesModelSample(np.zeros(2 ** N), mu_Ns=mu_Ns, L_halfs=L_halfs, elasticitys=elasticitys)

    def logpdf(self, *, mu_Ns, L_halfs, elasticitys):
        return self.mu_N_dist.logpdf(mu_Ns) + self.L_half_dist.logpdf(L_halfs) + self.elasticity_dist.logpdf(
            elasticitys)

    def show(self):
        mu_N_dom, L_half_dom, elasticity_dom = np.linspace(800, 1200, num=2 ** 10), np.linspace(0, 100,
                                                                                                num=2 ** 10), np.linspace(
            -3, 0, num=2 ** 10)
        fig, axs = plt.subplots(1, 3)
        fig.set_figwidth(15)
        # ps = [d.pdf(d.loc()) for d in [self.mu_N_dist, self.L_half_dist, self.elasticity_dist]]
        axs[0].plot(mu_N_dom, self.mu_N.pdf(mu_N_dom), label='prior')
        axs[0].set_xlabel('Daily arrival rate')
        axs[1].set_title('Prior uncertainty')
        axs[1].plot(L_half_dom, self.L_half.pdf(L_half_dom), label='prior')
        axs[1].set_xlabel('Reference price')
        axs[2].plot(elasticity_dom, self.elasticity.pdf(elasticity_dom), label='prior')
        axs[2].set_xlabel('Reference elasticity')
        for ax in axs:
            #    ax.set_ylim((-0.05*p, 1.05*p))
            ax.tick_params(left=False, labelleft=False)
        return fig, axs


class SalesModelSample:
    def __init__(self, log_weights, *, mu_Ns, L_halfs, elasticitys):
        self.data = np.array([mu_Ns, L_halfs, elasticitys]).T
        self.models = np.array([SalesModel(mu_N=a, L_half=b, elasticity=c) for [a, b, c] in self.data])
        self.log_weights = log_weights
        self._normalize_weights()
        return

    def _normalize_weights(self):
        normalizer = logsumexp(self.log_weights)
        self.log_weights -= normalizer
        return

    def _log_ess(self):
        return 2 * logsumexp(self.log_weights) - logsumexp(2 * self.log_weights)

    def ess(self):
        return np.exp(self._log_ess())

    def _integrate_sample(self, arr):
        logabsEs, sgns = logsumexp(self.log_weights, axis=-1, b=arr, return_sign=True)
        return sgns * np.exp(logabsEs)

    def _take_quantiles(self, arr, q):
        return wquantiles.quantile(arr, np.exp(self.log_weights), q)

    def purchase_probabilities(self, price):
        arr = np.array([m.purchase_probability(price) for m in self.models])
        return np.moveaxis(arr, 0, -1)

    def expected_purchase_probability(self, price):
        mat = self.purchase_probabilities(price)
        return self._integrate_sample(mat)

    def max_purchase_probability(self, price, alpha=0.01):
        mat = self.purchase_probabilities(price)
        return self._take_quantiles(mat, 1 - alpha)

    def min_purchase_probability(self, price, alpha=0.01):
        mat = self.purchase_probabilities(price)
        return self._take_quantiles(mat, alpha)

    def price_elasticities_of_demand(self, price):
        arr = np.array([m.price_elasticity_of_demand(price) for m in self.models])
        return np.moveaxis(arr, 0, -1)

    def expected_elasticity(self, price):
        mat = self.price_elasticities_of_demand(price)
        return self._integrate_sample(mat)

    def max_elasticity(self, price, alpha=0.01):
        mat = self.price_elasticities_of_demand(price)
        return self._take_quantiles(mat, 1 - alpha)

    def min_elasticity(self, price, alpha=0.01):
        mat = self.price_elasticities_of_demand(price)
        return self._take_quantiles(mat, alpha)

    def Esales(self, price):
        arr = np.array([m.expected_sales(price) for m in self.models])
        return np.moveaxis(arr, 0, -1)

    def expected_Esales(self, price):
        Esales = self.Esales(price)
        return self._integrate_sample(Esales)

    def min_Esales(self, price, alpha=0.01):
        Esales = self.Esales(price)
        return self._take_quantiles(Esales, alpha)

    def max_Esales(self, price, alpha=0.01):
        Esales = self.Esales(price)
        return self._take_quantiles(Esales, 1 - alpha)

    def expected_Erevenue(self, price):
        expected_Esales = self.expected_Esales(price)
        return price * expected_Esales

    def min_Erevenue(self, price, alpha=0.01):
        min_Esales = self.min_Esales(price, alpha)
        return price * min_Esales

    def max_Erevenue(self, price, alpha=0.01):
        max_Esales = self.max_Esales(price, alpha)
        return price * max_Esales

    def expected_Eprofit(self, price, cost):
        expected_Esales = self.expected_Esales(price)
        return (price - cost) * expected_Esales

    def min_Eprofit(self, price, cost, alpha=0.01):
        min_Esales = self.min_Esales(price, alpha)
        return (price - cost) * min_Esales

    def max_Eprofit(self, price, cost, alpha=0.01):
        max_Esales = self.max_Esales(price, alpha)
        return (price - cost) * max_Esales

    def pass_on_cost_delta(self, price, cost, delta):
        return np.array([m.pass_on_cost_delta(price, cost, delta) for m in self.models])

    def decide_pass_on_cost_delta(self, price, cost, delta, threshold=0):
        benefits = self.pass_on_cost_delta(price, cost, delta)
        decisions = benefits > threshold
        E_decision = self._integrate_sample(decisions)

    def show_pass_on_cost_delta(self, price, cost, delta, true_model=None):
        benefits = self.pass_on_cost_delta(price, cost, delta)
        wts = np.exp(self.log_weights)
        fig = plt.figure()
        ax = plt.gca()
        plt.hist(benefits, weights=wts, density=True, color='purple', alpha=0.3, bins=25)
        ax.set_xlabel('benefit')
        # ax.set_xlim((0, 100))
        ax.set_title('Uncertain benefit of passing cost increase')
        ax.set_ylabel('probability')
        plt.tick_params(left=False, labelleft=False)
        if true_model:
            ax.vlines(true_model.pass_on_cost_delta(price, cost, delta), *ax.get_ylim(), color='black',
                      linestyles='dotted', label='true')
        return fig

    def revmax_prices(self):
        return np.array([m.revmax_price() for m in self.models])

    def show(self, variable, true_model=None):
        prices = np.arange(101, step=2)

        if variable == 'Purchase probability':
            return self.show_purchase_probability(prices, true_model)
        elif variable == 'Elasticity of demand':
            return self.show_elasticity(prices, true_model)
        elif variable == 'Sales':
            return self.show_sales(prices, true_model)
        elif variable == 'Revenue':
            return self.show_revenue(prices, true_model)
        else:
            raise Exception('Choose variable only from Purchase probability, Elasticity of demand, Sales, Revenue')
        return

    def show_purchase_probability(self, prices, true_model=None):
        E_purchase_probability = self.expected_purchase_probability(prices)
        m_purchase_probability = self.min_purchase_probability(prices, alpha=0.5)
        lq_purchase_probability = self.min_purchase_probability(prices, alpha=0.25)
        uq_purchase_probability = self.max_purchase_probability(prices, alpha=0.25)
        min_purchase_probability = self.min_purchase_probability(prices, alpha=0.05)
        max_purchase_probability = self.max_purchase_probability(prices, alpha=0.05)

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(prices, m_purchase_probability, color='blue')
        ax.fill_between(prices, lq_purchase_probability, uq_purchase_probability, alpha=0.2, color='blue')
        ax.fill_between(prices, min_purchase_probability, max_purchase_probability, alpha=0.1, color='blue')
        ax.set_ylabel('probability')
        ax.set_ylim((-0.05, 1.05))
        ax.set_title('Uncertain purchase probability')
        ax.set_xlabel('price')
        if true_model:
            ax.plot(prices, true_model.purchase_probability(prices), color='black', linestyle='dotted', label='true')
        return fig

    def show_elasticity(self, prices, true_model=None):
        E_elasticity = self.expected_elasticity(prices)
        m_elasticity = self.min_elasticity(prices, alpha=0.5)
        lq_elasticity = self.min_elasticity(prices, alpha=0.25)
        uq_elasticity = self.max_elasticity(prices, alpha=0.25)
        min_elasticity = self.min_elasticity(prices, alpha=0.05)
        max_elasticity = self.max_elasticity(prices, alpha=0.05)

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(prices, m_elasticity, color='orange')
        ax.fill_between(prices, lq_elasticity, uq_elasticity, alpha=0.2, color='orange')
        ax.fill_between(prices, min_elasticity, max_elasticity, alpha=0.1, color='orange')
        ax.set_ylabel('elasticity')
        ax.set_ylim((-5.25, 0.25))
        ax.set_title('Uncertain elasticity of demand')
        ax.set_xlabel('price')
        if true_model:
            ax.plot(prices, true_model.price_elasticity_of_demand(prices), color='black', linestyle='dotted',
                    label='true')
        return fig

    def show_sales(self, prices, true_model=None):
        E_Esales = self.expected_Esales(prices)
        m_Esales = self.min_Esales(prices, 0.5)
        lq_Esales = self.min_Esales(prices, 0.25)
        uq_Esales = self.max_Esales(prices, 0.25)
        min_Esales = self.min_Esales(prices, 0.05)
        max_Esales = self.max_Esales(prices, 0.05)

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(prices, m_Esales, color='green')
        ax.fill_between(prices, lq_Esales, uq_Esales, alpha=0.2, color='green')
        ax.fill_between(prices, min_Esales, max_Esales, alpha=0.1, color='green')
        ax.set_ylabel('expected sales')
        ax.set_ylim((-60, 1260))
        ax.set_title('Uncertain expected sales')
        ax.set_xlabel('price')
        if true_model:
            ax.plot(prices, true_model.expected_sales(prices), color='black', linestyle='dotted', label='true')
        return fig

    def show_revenue(self, prices, true_model=None):
        E_Erevenue = self.expected_Erevenue(prices)
        m_Erevenue = self.min_Erevenue(prices, 0.5)
        lq_Erevenue = self.min_Erevenue(prices, 0.25)
        uq_Erevenue = self.max_Erevenue(prices, 0.25)
        min_Erevenue = self.min_Erevenue(prices, 0.05)
        max_Erevenue = self.max_Erevenue(prices, 0.05)

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(prices, m_Erevenue, color='red')
        ax.fill_between(prices, lq_Erevenue, uq_Erevenue, alpha=0.2, color='red')
        ax.fill_between(prices, min_Erevenue, max_Erevenue, alpha=0.1, color='red')
        ax.set_ylabel('expected revenue')
        ax.set_ylim((-1000, 51000))
        ax.set_title('Uncertain expected revenue')
        ax.set_xlabel('price')
        if true_model:
            ax.plot(prices, true_model.expected_revenue(prices), color='black', linestyle='dotted', label='true')
        return fig

    def show_posterior_histograms(self, prior, true_model=None):
        mu_Ns, L_halfs, elasticitys = self.data.T
        wts = np.exp(self.log_weights)
        fig, axs = prior.show()
        for ax, vec in zip(axs, [mu_Ns, L_halfs, elasticitys]):
            a, bins, patches = ax.hist(vec, weights=wts, bins=20, density=True, color='red')
            y0, y1 = ax.get_ylim()
            Y1 = max(a) if max(a) > y1 else y1
            ax.set_ylim((-0.05 * Y1, 1.05 * Y1))
        axs[1].set_title('Uncertain parameter distribution')
        axs[2].legend()
        if true_model:
            axs[0].vlines(true_model.mu_N, *axs[0].get_ylim(), color='black', linestyles='dotted', label='true')
            axs[1].vlines(true_model.L_half, *axs[1].get_ylim(), color='black', linestyles='dotted', label='true')
            axs[2].vlines(true_model.elasticity, *axs[2].get_ylim(), color='black', linestyles='dotted', label='true')
        return fig, axs

    def show_revmax_price_histogram(self, true_model=None):
        revmax_prices = self.revmax_prices()
        wts = np.exp(self.log_weights)
        fig = plt.figure()
        ax = plt.gca()
        plt.hist(revmax_prices, weights=wts, density=True, color='purple', alpha=0.3, bins=25)
        ax.set_xlabel('optimal price')
        ax.set_xlim((0, 100))
        ax.set_title('Uncertain revenue-maximising price')
        ax.set_ylabel('probability')
        plt.tick_params(left=False, labelleft=False)
        if true_model:
            ax.vlines(true_model.revmax_price(), *ax.get_ylim(), color='black', linestyles='dotted', label='true')
        return fig

    def condition(self, price, n_sales):
        ds = [m.sales_distribution(price) for m in self.models]
        self.log_weights += np.array([d.logpmf(n_sales) for d in ds])
        self._normalize_weights()
        return

    def get_posterior(self):
        return SalesModelPosterior(self)

    def resample(self, seed=None, *, prior_support=None):
        N = len(self.models)
        Q = self.get_posterior()
        return Q.sample(N, seed, prior_support=prior_support)


class SalesModelPosterior:
    def __init__(self, sample):
        self.q = stats.gaussian_kde(sample.data.T, weights=np.exp(sample.log_weights))
        return

    def sample(self, N, seed=None, *, prior_support=None):
        the_rng = np.random.default_rng(seed)
        X0 = np.array([prior_support[k][0] for k in ['mu_N', 'L_half', 'elasticity']] if prior_support else [0, 0,
                                                                                                             -float(
                                                                                                                 'inf')]).reshape(
            3, 1)
        X1 = np.array(
            [prior_support[k][1] for k in ['mu_N', 'L_half', 'elasticity']] if prior_support else [float('inf'),
                                                                                                   float('inf'),
                                                                                                   0]).reshape(3, 1)
        X = self.q.resample(N, the_rng)

        def invalid(X):
            return (X0 > X) | (X > X1)

        while invalid(X).any():
            invalid_idx = invalid(X).any(axis=0)
            N_new = np.sum(invalid_idx)
            X[:, invalid_idx] = self.q.resample(N_new, the_rng)
        mu_N, L_half, elasticity = X
        return SalesModelSample(np.zeros(N), mu_Ns=mu_N, L_halfs=L_half, elasticitys=elasticity)


def trunc_support_to_stds(support_pair, *, loc, scale):
    a, b = support_pair
    return (a - loc) / scale, (b - loc) / scale


def truncnorm(support_pair, *, loc, scale):
    return stats.truncnorm(*trunc_support_to_stds(support_pair, loc=loc, scale=scale), loc=loc, scale=scale)


class PriorsPlayML:
    def __init__(self, log2_N=5):
        l = widgets.Layout(width='600px')
        s = widgets.SliderStyle(description_width='200px')
        self.log2_N = log2_N
        self.params = ['mu_N', 'L_half', 'elasticity']
        self.days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        self.tomorrow_idx = 0

        # Init values for the parametric prior (take mean as initial nominal model)
        # means
        self.nominal = {'mu_N': 1000, 'L_half': 50, 'elasticity': -1.5}
        # stds
        self.sigma = {'mu_N': 50, 'L_half': 10, 'elasticity': 0.3}
        # supports of truncated prior
        self.supports = {'mu_N': (800, 1200), 'L_half': (0, 100), 'elasticity': (-3, -0.1)}
        # Parameter values for the data generating process
        self.true = {'mu_N': 1100, 'L_half': 60, 'elasticity': -2}

        ## Sliders
        # Single model
        self.nominal_sliders = {
            'mu_N': widgets.IntSlider(min=750, max=1250, step=10, description='Arrival rate:',
                                      value=self.nominal['mu_N'], continuous_update=False, style=s, layout=l),
            'L_half': widgets.IntSlider(min=20, max=80, step=1, description='Reference price:',
                                        value=self.nominal['L_half'], continuous_update=False, style=s, layout=l),
            'elasticity': widgets.FloatSlider(min=-2.5, max=-0.5, step=0.1, description='Reference elasticity:',
                                              value=self.nominal['elasticity'], continuous_update=False, style=s,
                                              layout=l)
        }
        # Prior uncertainty
        self.sigma_sliders = {
            'mu_N': widgets.IntSlider(min=1, max=100, step=1, description='Arrival rate uncertainty:',
                                      value=self.sigma['mu_N'], continuous_update=False, style=s, layout=l),
            'L_half': widgets.FloatSlider(min=0.5, max=20, step=0.5, description='Ref. price uncertainty:',
                                          value=self.sigma['L_half'], continuous_update=False, style=s, layout=l),
            'elasticity': widgets.FloatSlider(min=0.01, max=0.5, step=0.01, description='Ref. elasticity uncertainty:',
                                              value=self.sigma['elasticity'], continuous_update=False, style=s,
                                              layout=l)
        }
        # Data prices
        self.data_sliders = {
            d: widgets.IntSlider(min=1, max=100, step=1, description=f'{d}: ', value=60,
                                 continuous_update=False, layout=l) for d in self.days}
        # Learn button
        self.btn = widgets.Button(description='LEARN!')
        self.reset = widgets.Button(description='Reset to prior')
        # Predictive variable
        self.prior_predictive_selector = widgets.RadioButtons(
            options=['Purchase probability', 'Elasticity of demand', 'Sales', 'Revenue'],
            value='Purchase probability',
            description='Prior predictive distributions',
            disabled=False
        )
        self.posterior_predictive_selector = widgets.RadioButtons(
            options=['Purchase probability', 'Elasticity of demand', 'Sales', 'Revenue'],
            value='Purchase probability',
            description='Posterior predictive distributions',
            disabled=False
        )

        ## Objects
        self.the_nominal_model = self.nominal_model()
        self.the_prior = self.prior()
        self.the_sample = self.sample()
        self.the_true_model = self.true_model()
        self.the_data = self.synthetic_data()
        self.the_MLEs = self.MLEs()

        ## Outputs
        # Create
        self.nominal_predictive = widgets.Output()
        self.nominal_decision_fig = widgets.Output()
        self.data_fig = widgets.Output()
        self.MLE_comparison = widgets.Output()
        self.prior_parameters = widgets.Output()
        self.prior_predictive = widgets.Output()
        self.prior_revmax = widgets.Output()
        self.posterior_parameters = widgets.Output()
        self.posterior_predictive = widgets.Output()
        self.posterior_revmax = widgets.Output()
        self.view_button = widgets.Output()
        # Initialise
        self.true_revmax = widgets.HTML(value=self.report_revmax('true'))
        self.nominal_predictive.append_display_data(self.the_nominal_model.show())
        self.nominal_decision_fig.append_display_data(self.the_nominal_model.show_profits([30, 40], price_decisions=[60, 70], label=['Cost £30', 'Cost £40']))
        self.nominal_decision_report = widgets.HTML(value=self.report_decision('nominal'))
        self.nominal_revmax = widgets.HTML(value=self.report_revmax('nominal'))
        self.data_fig.append_display_data(self.do_data_fig())
        self.MLE_comparison.append_display_data(self.do_MLE_fig())
        self.prior_parameters.append_display_data(self.the_prior.show()[0])
        self.prior_predictive.append_display_data(self.the_sample.show(self.prior_predictive_selector.value))
        self.prior_revmax.append_display_data(self.the_sample.show_revmax_price_histogram())
        self.posterior_parameters.append_display_data(
            self.the_sample.show_posterior_histograms(self.the_prior, self.the_true_model)[0])
        self.posterior_predictive.append_display_data(
            self.the_sample.show(self.posterior_predictive_selector.value, self.the_true_model))
        self.posterior_revmax.append_display_data(self.the_sample.show_revmax_price_histogram(self.the_true_model))
        self.view_button.append_display_data(self.btn)
        self.upcoming_data = widgets.HTML(value=self.report_upcoming_data())

        ## Observers
        # Single model
        for k in self.params:
            self.nominal_sliders[k].observe(self.nominal_model_slider_handler(k), names='value')
            self.sigma_sliders[k].observe(self.prior_slider_handler(k), names='value')
        # Prices
        for d in self.days:
            self.data_sliders[d].observe(self.data_slider_handler(d), names='value')
        # Learn button
        self.btn.on_click(self.button_handler())
        self.reset.on_click(self.reset_handler())
        # Selectors
        self.prior_predictive_selector.observe(self.selector_handler(True), names='value')
        self.posterior_predictive_selector.observe(self.selector_handler(False), names='value')
        return

    def nominal_model(self):
        return SalesModel(**self.nominal)

    def true_model(self):
        return SalesModel(**self.true)

    def prior(self):
        dists = {p: truncnorm(self.supports[p], loc=self.nominal[p], scale=self.sigma[p]) for p in self.params}
        return SalesModelDistribution(**dists)

    def sample(self):
        return self.the_prior.sample(self.log2_N)

    def synthetic_data(self):
        return {d: (
            self.data_sliders[d].value,  # price
            self.the_true_model.sales_distribution(self.data_sliders[d].value).rvs(
                random_state=np.prod([ord(c) for c in d])),  # random sales at that price
        ) for d in self.days}

    def the_prices(self):
        return np.array([self.the_data[d][0] for d in self.days])

    def the_sales(self):
        return np.array([self.the_data[d][1] for d in self.days])

    def cost_vector_function(self, theta):
        prices = self.the_prices()
        sales = self.the_sales()
        mu_N, L_half, elasticity = theta
        m = SalesModel(mu_N=mu_N, L_half=L_half, elasticity=elasticity)
        return -1 * m.log_likelihoods(prices, sales).cumsum()

    def MLEs(self):
        x0 = [self.nominal[p] for p in self.params]
        mle_models = []
        for n_days in range(7):
            f = lambda x: self.cost_vector_function(x)[n_days]
            xmin = minimize(f, x0, bounds=[self.supports[p] for p in self.params]).x
            # print(sol.message)
            x0 = copy.copy(xmin)
            mle_models.append(SalesModel(mu_N=xmin[0], L_half=xmin[1], elasticity=xmin[2]))
        return mle_models

    def learn_posteriors(self):
        self.the_sample = self.sample()
        for d in self.days:
            self.step_posterior(d)
        return

    def step_posterior(self, day):
        obs_price, obs_sales = self.the_data[day]
        self.the_sample.condition(obs_price, obs_sales)
        self.the_sample = self.the_sample.resample(seed=ord(day[0]), prior_support=self.supports)
        return

    def report_upcoming_data(self):
        if self.tomorrow_idx < 7:
            d = self.days[self.tomorrow_idx]
            next_price, next_sales = self.the_data[d]
            output_html = f'Next observation is from <b>{self.days[self.tomorrow_idx]}</b>:<br>{next_sales} sales at price £{next_price}'
        else:
            output_html = '<b>No more observations available!</b><br>'
        return output_html

    def report_revmax(self, thetype):
        d = {
            'nominal': self.the_nominal_model,
            'MLE': self.the_MLEs[-1],
            'true': self.the_true_model,
        }
        P = d[thetype].revmax_price()
        return f'Price to maximise revenue (<b>{thetype}</b> model):<br>£{P:.2f}'

    def report_decision(self, thetype):
        d = {
            'nominal': self.the_nominal_model,
            'MLE': self.the_MLEs[-1],
            'true': self.the_true_model,
        }
        P_60 = d[thetype].expected_profit(60, 40)
        s_60 = f'Profit at £60: £{P_60:,.2f}'
        P_70 = d[thetype].expected_profit(70, 40)
        s_70 = f'Profit at £70: £{P_70:,.2f}'
        if P_60 >= P_70:
            s_decision = 'Keep price at £60'
        else:
            s_decision = 'Increase price to £70'
        return s_60 + '<br>' + s_70 + '<br><b>Decision</b>: ' + s_decision

    def nominal_model_slider_handler(self, field_name):
        def _handler(change):
            self.nominal_predictive.clear_output(wait=True)
            self.nominal_decision_fig.clear_output(wait=True)
            setattr(self.the_nominal_model, field_name, change.new)
            with self.nominal_predictive:
                fig = self.the_nominal_model.show()
                display(fig)
                plt.close(fig)
            with self.nominal_decision_fig:
                fig = self.the_nominal_model.show_profits([30, 40], price_decisions=[60,70], label=['Cost £30', 'Cost £40'])
                display(fig)
                plt.close(fig)
            self.nominal_decision_report.value = self.report_decision('nominal')
            return

        return _handler

    def prior_slider_handler(self, param):
        def _handler(change):
            new_dist = truncnorm(self.supports[param], loc=self.nominal[param], scale=change.new)
            setattr(self.the_prior, param, new_dist)
            self.prior_parameters.clear_output(wait=True)
            self.prior_predictive.clear_output(wait=True)
            self.prior_revmax.clear_output(wait=True)
            self.the_sample = self.sample()
            self.tomorrow_idx = 0
            with self.prior_parameters:
                fig, axs = self.the_prior.show()
                display(fig)
                plt.close(fig)
            with self.prior_revmax:
                fig = self.the_sample.show_revmax_price_histogram()
                display(fig)
                plt.close(fig)
            with self.prior_predictive:
                fig = self.the_sample.show(self.prior_predictive_selector.value)
                display(fig)
                plt.close(fig)
            self.set_posterior_to_prior()
            return

        return _handler

    def set_posterior_to_prior(self):
        self.posterior_parameters.clear_output(wait=True)
        self.posterior_predictive.clear_output(wait=True)
        self.posterior_revmax.clear_output(wait=True)
        self.view_button.clear_output(wait=True)
        # Update the prior
        with self.posterior_parameters:
            fig, axs = self.the_sample.show_posterior_histograms(self.the_prior, self.the_true_model)
            display(fig)
            plt.close(fig)
        # Update the revmax distribution (deterministic map of the prior)
        with self.posterior_revmax:
            fig = self.the_sample.show_revmax_price_histogram(self.the_true_model)
            display(fig)
            plt.close(fig)
        # Update the predictive distribution
        with self.posterior_predictive:
            fig = self.the_sample.show(self.posterior_predictive_selector.value, self.the_true_model)
            display(fig)
            plt.close(fig)
        with self.view_button:
            display(self.btn)
        self.upcoming_data.value = self.report_upcoming_data()

        return

    def data_slider_handler(self, d):
        def _handler(change):
            self.data_fig.clear_output(wait=True)
            self.MLE_comparison.clear_output(wait=True)
            self.the_data[d] = (
                change.new,
                self.the_true_model.sales_distribution(change.new).rvs(random_state=np.prod([ord(c) for c in d]))
            )
            self.upcoming_data.value = self.report_upcoming_data()
            self.the_MLEs = self.MLEs()
            with self.data_fig:
                fig = self.do_data_fig()
                display(fig)
                plt.close(fig)
            with self.MLE_comparison:
                fig = self.do_MLE_fig()
                display(fig)
                plt.close(fig)
            return

        return _handler

    def button_handler(self):
        def _handler(obj):
            self.posterior_parameters.clear_output(wait=True)
            self.posterior_revmax.clear_output(wait=True)
            self.posterior_predictive.clear_output(wait=True)

            self.step_posterior(self.days[self.tomorrow_idx])
            self.tomorrow_idx += 1

            with self.posterior_parameters:
                fig, axs = self.the_sample.show_posterior_histograms(self.the_prior, self.the_true_model)
                display(fig)
                plt.close(fig)
            with self.posterior_revmax:
                fig = self.the_sample.show_revmax_price_histogram(self.the_true_model)
                display(fig)
                plt.close(fig)
            with self.posterior_predictive:
                fig = self.the_sample.show(self.posterior_predictive_selector.value, self.the_true_model)
                display(fig)
                plt.close(fig)
            if self.tomorrow_idx == 7:
                self.view_button.clear_output(wait=True)
                with self.view_button:
                    display(self.reset)
            self.upcoming_data.value = self.report_upcoming_data()
            return

        return _handler

    def reset_handler(self):
        def _handler(obj):
            self.the_sample = self.sample()
            self.tomorrow_idx = 0
            self.set_posterior_to_prior()
            return

        return _handler

    def selector_handler(self, prior=False):
        def _handler(change):
            out_widget = self.prior_predictive if prior else self.posterior_predictive
            true_model = None if prior else self.the_true_model
            out_widget.clear_output(wait=True)
            with out_widget:
                fig = self.the_sample.show(change.new, true_model)
                display(fig)
                plt.close(fig)
            return

        return _handler

    def visualise_nominal_model(self):
        return widgets.VBox([
            widgets.HBox([
                widgets.VBox([self.nominal_sliders[v] for v in self.params]),
                self.nominal_revmax]),
            self.nominal_predictive])

    def visualise_nominal_decision(self):
        return widgets.HBox([
            self.nominal_decision_fig,
            self.nominal_decision_report,
        ])

    def visualise_prior(self):
        return widgets.VBox([
            #widgets.VBox([self.sigma_sliders[v] for v in self.params]),
            self.prior_parameters,
            self.visualise_prior_predictive_interactive(),
        ])

    def visualise_prior_predictive_interactive(self):
        return widgets.HBox([
            self.prior_predictive_selector,
            self.prior_predictive,
        ])

    def visualise_posterior(self):
        return widgets.VBox([
            widgets.HBox([
                widgets.VBox([self.sigma_sliders[v] for v in self.params]),
                widgets.VBox([self.upcoming_data, self.view_button]),
            ]),
            self.posterior_parameters,
            self.posterior_revmax,
        ])

    def visualise_posterior_predictive(self):
        return widgets.HBox([
            self.posterior_predictive_selector,
            self.posterior_predictive,
        ])

    def do_data_fig(self):
        _prices = [self.the_data[d][0] for d in self.days]
        _sales = [self.the_data[d][1] for d in self.days]
        fig = plt.figure()
        ax1 = plt.gca()
        ax1.bar(self.days, _sales, color='blue', label='sales')
        ax1.set_ylabel('sales')
        ax2 = ax1.twinx()
        ax2.plot(self.days, _prices, color='orange', label='prices')
        plt.tick_params()
        ax2.set_ylim((0, 100))
        ax2.set_ylabel('prices')
        fig.legend()
        return fig

    def visualise_data(self):
        return widgets.HBox([
            widgets.VBox([self.data_sliders[d] for d in self.days]),
            self.data_fig
        ])

    def do_MLE_fig(self):
        M = np.array([m.parameters() for m in self.the_MLEs])
        T = np.array(self.the_true_model.parameters()).reshape(1, 3)

        fig, axs = plt.subplots(1, 3)
        fig.set_figwidth(15)

        axs[0].plot(self.days, (M) / T, label=['Arrival rate', 'Ref. price', 'Ref. elasticity'])
        axs[0].hlines(1, 0, 6, color='black', linestyles='dotted', label='normalised true values')
        axs[0].legend()
        axs[0].set_title('MLE parameters relative to truth')
        axs[0].set_ylabel('Multiple of true parameter value')

        axs[2].plot(self.days, [m.revmax_price() for m in self.the_MLEs], label='price recommendation', color='red')
        axs[2].hlines(self.the_true_model.revmax_price(), 0, 6, label='optimal price', color='black',
                      linestyles='dotted')
        axs[2].set_title('Price recommendations vs optimality')
        axs[2].legend()
        axs[2].set_ylabel('price')

        true_fits = self.cost_vector_function(self.the_true_model.parameters())
        mle_fits = np.array([self.cost_vector_function(m.parameters()) for m in self.the_MLEs]).diagonal()
        axs[1].plot(self.days, [100 * (1 - (f_mle / f_true)) for f_mle, f_true in zip(mle_fits, true_fits)])
        axs[1].set_title('Improved fits to data from MLE vs true parameters')
        axs[1].set_ylabel('Percentage improvement in cost function')
        return fig

    def visualise_MLEs(self):
        return self.MLE_comparison

    def do_pricing_fig(self):
        M = self.the_true_model
        dist_posterior = stats.gaussian_kde([m.revmax_price() for m in self.the_sample.models])

        p_opt = M.revmax_price()
        p_modal = minimize(lambda P: -dist_posterior.logpdf(P), self.nominal['L_half']).x[0]
        ps_posterior = dist_posterior.resample(256, seed=0)
        p_MLE = self.the_MLEs[self.tomorrow_idx - 1].revmax_price() if self.tomorrow_idx > 0 else float('nan')

        rev_opt = M.expected_revenue(p_opt)
        rev_modal = M.expected_revenue(p_modal)
        rev_posterior = M.expected_revenue(ps_posterior).mean()
        rev_MLE = M.expected_revenue(p_MLE) if p_MLE else float('nan')

        fig = plt.figure()
        plt.bar([f'True: £{p_opt:.2f}', f'MLE: £{p_MLE:.2f}', f'Modal: £{p_modal:.2f}', 'Random posterior'],
                [rev_opt, rev_MLE, rev_modal, rev_posterior],
                alpha=0.3,
                color=['black', 'red', 'purple', 'purple'])
        plt.gca().set_ylabel('expected revenue')
        ttl = f'Prices set on {self.days[self.tomorrow_idx - 1]}' if self.tomorrow_idx > 0 else 'Prices before data'
        plt.title(ttl)
        return fig

    def visualise_price_strategy(self):
        return widgets.HBox([
            self.price_strategy,
            widgets.VBox([self.upcoming_data, self.view_button]),
        ])
