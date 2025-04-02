"""
Finding topical subsections within text using the TextTiling algorithm.

Notes
-----
- TextTiling was created by Marti A. Hearst in the mid 1990's:
    https://aclanthology.org/J97-1003.pdf
- TextTiling using BERT embeddings was first done June 2021:
    https://arxiv.org/abs/2106.12978
"""
# standard library imports
from collections.abc import Awaitable, Callable
import logging

# current package imports
from .exceptions import TextTilerError

# local package imports
from ai_clips_maker.filesys.manager import FileSystemManager
from ai_clips_maker.utils.config_manager import ConfigManager
from ai_clips_maker.utils.pytorch import (
    max_magnitude_2d,
    get_compute_device,
    assert_compute_device_available,
)
from ai_clips_maker.utils.utils import find_missing_dict_keys

# 3rd party imports
import numpy
import torch
import torch.nn.functional as F

BOUNDARY = 1


class TextTiler:
    """
    Groups sentence embeddings into topical sections using the TextTiling algorithm.
    
    The embeddings are segmented by comparing similarity gaps. Peak gaps are marked
    as boundaries and each resulting group is pooled into a single embedding.
    """

    def __init__(self, device: str = None) -> None:
        self._device = device or get_compute_device()
        assert_compute_device_available(self._device)
        self._config_checker = TextTilerConfigManager()
        self._fs_manager = FileSystemManager()

    def text_tile(
        self,
        embeddings: torch.Tensor,
        k: int = 7,
        window_compare_pool_method: str = "mean",
        embedding_aggregation_pool_method: str = "max",
        smoothing_width: int = 3,
        cutoff_policy: str = "high",
    ) -> tuple[list, torch.Tensor]:
        """
        Segments the input embeddings and returns the detected boundaries and
        pooled segment embeddings.
        """
        config = {
            "k": k,
            "window_compare_pool_method": window_compare_pool_method,
            "embedding_aggregation_pool_method": embedding_aggregation_pool_method,
            "smoothing_width": smoothing_width,
            "cutoff_policy": cutoff_policy,
        }
        self._config_checker.assert_valid_config(config)

        N, E = embeddings.shape

        if k >= N:
            new_k = max(N // 5, 2)
            logging.warning(
                f"{N} embeddings is too few for k={k}. Using k={new_k} instead."
            )
            k = new_k

        if smoothing_width >= N:
            smoothing_width = 2

        unsmoothed_scores = self._calc_gap_scores(embeddings, k, window_compare_pool_method)
        smoothed_scores = self._smooth_scores(unsmoothed_scores, smoothing_width)
        depth_scores = self._calc_depth_scores(smoothed_scores)
        boundaries = self._identify_boundaries(depth_scores, cutoff_policy)

        pooled_embeddings = self._pool_embedding_groups(
            embeddings, boundaries, embedding_aggregation_pool_method
        )

        return list(boundaries), pooled_embeddings

    def _calc_gap_scores(
        self,
        embeddings: torch.Tensor,
        k: int,
        pool_method: str,
    ) -> torch.Tensor:
        pool = self._get_pool_method(pool_method)
        N = embeddings.shape[0]
        gap_scores = torch.empty(N - 1, device=self._device)

        for i in range(N - 1):
            left = embeddings[max(0, i - k + 1): i + 1]
            right = embeddings[i + 1: min(i + 1 + k, N)]
            left_pooled = pool(left, dim=0)
            right_pooled = pool(right, dim=0)
            gap_scores[i] = F.cosine_similarity(left_pooled, right_pooled, dim=0)

        return gap_scores

    def _smooth_scores(self, scores: torch.Tensor, width: int) -> torch.Tensor:
        arr = scores.cpu().detach().numpy()
        return torch.tensor(
            smooth(numpy.array(arr), window_len=width, window="flat"), device=self._device
        )

    def _calc_depth_scores(self, gap_scores: torch.Tensor) -> torch.Tensor:
        N = len(gap_scores)
        depths = torch.zeros(N, device=self._device)

        for i in range(N):
            g = gap_scores[i]
            left_peak = max(gap_scores[max(0, i - j)] for j in range(i + 1))
            right_peak = max(gap_scores[min(N - 1, i + j)] for j in range(N - i))
            depths[i] = (left_peak - g) + (right_peak - g)

        return depths

    def _identify_boundaries(self, depths: torch.Tensor, policy: str) -> torch.Tensor:
        N = len(depths) + 1
        boundaries = torch.zeros(N, device=self._device)

        avg, std = torch.mean(depths), torch.std(depths, unbiased=False)
        cutoff = {"average": avg, "high": avg + std, "low": avg - std}.get(policy)
        if cutoff is None:
            raise TextTilerError(f"Invalid cutoff_policy: {policy}")

        for i in range(len(depths)):
            if (
                depths[i] > cutoff and
                depths[i] > depths[max(0, i - 1)] and
                depths[i] > depths[min(i + 1, len(depths) - 1)]
            ):
                boundaries[i] = 1

        boundaries[N - 1] = BOUNDARY
        return boundaries

    def _pool_embedding_groups(
        self,
        embeddings: torch.Tensor,
        boundaries: list,
        pool_method: str,
    ) -> torch.Tensor:
        pool = self._get_pool_method(pool_method)
        N, E = embeddings.shape
        pooled = []
        group = []

        for i in range(N):
            group.append(embeddings[i].unsqueeze(0))
            if boundaries[i] == BOUNDARY:
                group_tensor = torch.cat(group, dim=0)
                pooled.append(pool(group_tensor, dim=0))
                group = []

        return torch.stack(pooled)

    def _get_pool_method(self, name: str) -> Callable[[torch.Tensor], Awaitable[torch.Tensor]]:
        if name == "mean":
            return torch.mean
        if name == "max":
            return max_magnitude_2d
        raise TextTilerError(f"Unknown pool_method: {name}")


def smooth(x, window_len=3, window="flat"):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1D arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError("Invalid window type.")

    s = numpy.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    w = numpy.ones(window_len) if window == "flat" else eval(f"numpy.{window}({window_len})")
    y = numpy.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1: -window_len + 1]


class TextTilerConfigManager(ConfigManager):
    def __init__(self) -> None:
        super().__init__()

    def check_valid_config(self, config: dict) -> str | None:
        required = [
            "cutoff_policy", "embedding_aggregation_pool_method",
            "k", "smoothing_width", "window_compare_pool_method",
        ]
        missing = find_missing_dict_keys(config, required)
        if missing:
            return f"TextTiler missing config settings: {missing}"

        checkers = {
            "cutoff_policy": self.check_valid_cutoff_policy,
            "embedding_aggregation_pool_method": self.check_valid_pool_method,
            "k": self.check_valid_k,
            "smoothing_width": self.check_valid_smoothing_width,
            "window_compare_pool_method": self.check_valid_pool_method,
        }
        for key, checker in checkers.items():
            err = checker(config[key])
            if err:
                return err

        return None

    def check_valid_k(self, k: int) -> str | None:
        if not isinstance(k, int) or k < 2:
            return f"Invalid k value: {k}. Must be >= 2."
        return None

    def check_valid_pool_method(self, method: str) -> str | None:
        if method not in ["mean", "max"]:
            return f"Invalid pool_method: {method}. Must be 'mean' or 'max'."
        return None

    def check_valid_smoothing_width(self, width: int) -> str | None:
        if not isinstance(width, int) or width < 3:
            return f"Invalid smoothing_width: {width}. Must be >= 3."
        return None

    def check_valid_cutoff_policy(self, policy: str) -> str | None:
        if policy not in ["average", "low", "high"]:
            return f"Invalid cutoff_policy: {policy}. Must be 'average', 'low', or 'high'."
        return None

    def check_valid_clip_times(self, min_dur: float, max_dur: float) -> str | None:
        if not isinstance(min_dur, (int, float)) or min_dur < 0:
            return f"Invalid min_clip_duration: {min_dur}"
        if not isinstance(max_dur, (int, float)) or max_dur <= min_dur:
            return f"max_clip_duration {max_dur} must be > min_clip_duration {min_dur}"
        return None
