import numpy as np
from scipy.stats import norm


def get_selector(strategy_name, kwargs):
    if strategy_name == "random":
        selector = RandomSelection(**kwargs)
    elif strategy_name == "greedy":
        selector = GreedySelection()
    elif strategy_name == "ei":
        return EISelection(**kwargs)
    elif strategy_name == "pi":
        return PISelection(**kwargs)
    elif strategy_name == "ucb":
        return UCBSelection(**kwargs)
    elif strategy_name == "thompson":
        return ThompsonSamplingSelection(**kwargs)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy_name}")
    return selector


class BaseSelectionStrategy:
    """
    Base class for active learning data selection.
    Subclass this for different selection methods such as random, greedy, etc.
    """
    def select_samples(self, pool_df, preds, selection_size):
        scores = self.acquisition_score(preds)
        sort_idx = np.argsort(-scores)
        return pool_df.index[sort_idx[:selection_size]].tolist()

    def acquisition_score(self, preds):
        raise NotImplementedError("Please implement 'acquisition_score' method.")


class RandomSelection(BaseSelectionStrategy):
    """
    Random selection strategy for active learning.
    """
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def acquisition_score(self, preds):
        return self.rng.rand(len(preds))


class GreedySelection(BaseSelectionStrategy):
    """
    Example of a greedy-based selection using some heuristic scores.
    """
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def acquisition_score(self, preds):
        return preds

class EnsembleSelectionStrategy:
    """
    Selection strategy using Bayesian dropout for uncertainty estimation.
    """
    def select_samples(self, pool_df, mu, sigma, selection_size):
        scores = self.acquisition_score(mu, sigma)
        sort_idx = np.argsort(-scores)
        selected_indices = pool_df.index[sort_idx[:selection_size]].tolist()
        return selected_indices

    def acquisition_score(self, mu, sigma):
        raise NotImplementedError("Please implement 'acquisition_score' method.")


class EISelection(EnsembleSelectionStrategy):
    """
    EI (Expected Improvement) を用いてサンプルを選択するクラスの例。
    """
    def __init__(self, xi=0.01, best_score=None):
        """
        Args:
            xi: float
                Exploration parameter (exploitation とのバランスを調整するため)
            best_score: float or None
                既知のベストスコア。None の場合は、現時点での予測平均の最大を使用
        """
        self.xi = xi
        self.best_score = best_score

    def acquisition_score(self, mu, sigma):
        n_samples = len(mu)

        # ベストスコアが与えられていない場合は、平均の最大値をベストとみなす
        if self.best_score is None:
            f_best = np.max(mu)
        else:
            f_best = self.best_score

        # EI を計算
        # EI = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
        # Z = (mu - f_best - xi) / sigma
        ei_scores = np.zeros(n_samples)
        for i in range(n_samples):
            if sigma[i] < 1e-9:  # sigma が極端に小さい場合の対策
                continue
            Z = (mu[i] - f_best - self.xi) / sigma[i]
            ei_scores[i] = ((mu[i] - f_best - self.xi) * norm.cdf(Z)
                            + sigma[i] * norm.pdf(Z))

        return ei_scores


class PISelection(EnsembleSelectionStrategy):
    """
    PI (Probability of Improvement) を用いてサンプルを選択するクラスの例。
    """
    def __init__(self, best_score=None):
        """
        Args:
            best_score: float or None
                既知のベストスコア。None の場合は、現時点での予測平均の最大を使用
        """
        self.best_score = best_score

    def acquisition_score(self, mu, sigma):
        n_samples = len(mu)

        # ベストスコアが与えられていない場合
        if self.best_score is None:
            f_best = np.max(mu)
        else:
            f_best = self.best_score

        # PI = 1 - Phi((f_best - mu) / sigma)
        pi_scores = np.zeros(n_samples)
        for i in range(n_samples):
            if sigma[i] < 1e-9:
                continue
            Z = (f_best - mu[i]) / sigma[i]
            pi_scores[i] = 1.0 - norm.cdf(Z)

        return pi_scores


class UCBSelection(EnsembleSelectionStrategy):
    """
    UCB (Upper Confidence Bound) を用いてサンプルを選択するクラスの例。
    """
    def __init__(self, beta=1.0):
        """
        Args:
            beta: float
                不確実性 (標準偏差) をどの程度重視するかを調整するパラメータ
        """
        self.beta = beta

    def acquisition_score(self, mu, sigma):
        # UCB = mu + beta * sigma
        ucb_scores = mu + self.beta * sigma
        return ucb_scores


class ThompsonSamplingSelection(EnsembleSelectionStrategy):
    """
    Thompson Sampling を用いてサンプルを選択するクラスの例。
    ここでは、GPRの予測分布(N(μ, σ^2))から乱数を引き、
    その乱数が大きい順にサンプルを選ぶ例を示す。
    """
    def __init__(self, seed=42):
        """
        Args:
            seed: int
                乱数シード
        """
        self.rng = np.random.RandomState(seed)

    def acquisition_score(self, mu, sigma):
        n_samples = len(mu)

        # sigma が 0 のところは乱数を引けないので、わずかに値を持たせる
        safe_sigma = np.where(sigma < 1e-9, 1e-9, sigma)

        # 各サンプルに対して N(mu[i], sigma[i]^2) から1回ずつ乱数を引く
        samples = self.rng.normal(loc=mu, scale=safe_sigma, size=n_samples)
        return samples
