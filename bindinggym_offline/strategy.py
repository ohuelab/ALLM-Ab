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
        """
        Select samples from the pool_df given the current model state.
        The return value is an index or list of indices that are selected.
        """
        raise NotImplementedError("Please implement 'select_samples' method.")


class RandomSelection(BaseSelectionStrategy):
    """
    Random selection strategy for active learning.
    """
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def select_samples(self, pool_df, preds, selection_size):
        all_indices = pool_df.index.to_list()
        self.rng.shuffle(all_indices)
        return all_indices[:selection_size]


class GreedySelection(BaseSelectionStrategy):
    """
    Example of a greedy-based selection using some heuristic scores.
    """
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def select_samples(self, pool_df, preds, selection_size):
        sort_idx = np.argsort(-preds)
        return pool_df.index[sort_idx[:selection_size]].tolist()


class EnsembleUncertaintySelector(BaseSelectionStrategy):
    """
    Selection strategy using ensemble-based uncertainty estimates.
    This example uses variance across multiple models as a metric.
    """
    def select_samples(self, pool_df, preds_list, selection_size):
        variance_array = np.var(preds_list, axis=0)
        sort_idx = np.argsort(-variance_array)
        return pool_df.index[sort_idx[:selection_size]].tolist()


class EnsembleSelectionStrategy:
    """
    Selection strategy using Bayesian dropout for uncertainty estimation.
    """
    def select_samples(self, pool_df, preds_list, selection_size):
        raise NotImplementedError("Please implement 'select_samples' method.")


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

    def select_samples(self, pool_df, preds_list, selection_size):
        n_models, n_samples = len(preds_list), len(preds_list[0])
        # 各サンプルに対してエンサンブルの平均・標準偏差を算出
        mu = np.mean(preds_list, axis=0)
        sigma = np.std(preds_list, axis=0, ddof=1)  # 母集団でなく標本標準偏差を使うなら ddof=1

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

        # EI が大きい順に並べて selection_size 個選ぶ
        sort_idx = np.argsort(-ei_scores)
        selected_indices = pool_df.index[sort_idx[:selection_size]].tolist()
        return selected_indices


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

    def select_samples(self, pool_df, preds_list, selection_size):
        n_models, n_samples = len(preds_list), len(preds_list[0])
        mu = np.mean(preds_list, axis=0)
        sigma = np.std(preds_list, axis=0, ddof=1)

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

        sort_idx = np.argsort(-pi_scores)
        selected_indices = pool_df.index[sort_idx[:selection_size]].tolist()
        return selected_indices


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

    def select_samples(self, pool_df, preds_list, selection_size):
        n_models, n_samples = len(preds_list), len(preds_list[0])
        mu = np.mean(preds_list, axis=0)
        sigma = np.std(preds_list, axis=0, ddof=1)

        # UCB = mu + beta * sigma
        ucb_scores = mu + self.beta * sigma

        sort_idx = np.argsort(-ucb_scores)
        selected_indices = pool_df.index[sort_idx[:selection_size]].tolist()
        return selected_indices


class ThompsonSamplingSelection(EnsembleSelectionStrategy):
    """
    Thompson Sampling を用いてサンプルを選択するクラスの例。
    ここでは、エンサンブル予測から推定される正規分布(N(μ, σ^2))から乱数を引き、
    その乱数が大きい順にサンプルを選ぶ例を示す。
    """
    def __init__(self, seed=42):
        """
        Args:
            seed: int
                乱数シード
        """
        self.rng = np.random.RandomState(seed)

    def select_samples(self, pool_df, preds_list, selection_size):

        n_models, n_samples = len(preds_list), len(preds_list[0])
        mu = np.mean(preds_list, axis=0)
        sigma = np.std(preds_list, axis=0, ddof=1)

        # sigma が 0 のところは乱数を引けないので、わずかに値を持たせる
        safe_sigma = np.where(sigma < 1e-9, 1e-9, sigma)

        # 各サンプルに対して N(mu[i], sigma[i]^2) から1回ずつ乱数を引く
        samples = self.rng.normal(loc=mu, scale=safe_sigma, size=n_samples)

        # サンプル値が大きい順にソートして上位 selection_size 個を選ぶ
        sort_idx = np.argsort(-samples)
        selected_indices = pool_df.index[sort_idx[:selection_size]].tolist()
        return selected_indices
