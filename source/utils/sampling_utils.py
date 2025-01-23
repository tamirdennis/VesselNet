from torch.utils.data.sampler import Sampler

from typing import Iterator
import random


class ValuesBinsSampler(Sampler[int]):
    r"""
    """

    def __init__(self, num_samples: int, pre_data: dict, bins_size: float, bins=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))

        self.num_samples = num_samples
        self.pre_data = pre_data
        self.bins_size = bins_size
        self.bins = bins

    def __iter__(self) -> Iterator[int]:
        gts, gts_inds = [], []
        for i, d in self.pre_data.items():
            gts_inds.append(i)
            gts.append(d['gt'])
        gts_inds_sorted_by_gts = [f for _, f in sorted(zip(gts, gts_inds))]
        gts_sorted = sorted(gts)

        bins = []
        inds_in_bins = []
        curr_bin = []
        if self.bins is None:
            curr_bin_min, curr_bin_max = gts_sorted[0], gts_sorted[0] + self.bins_size
        else:
            bins_index = 0
            curr_bin_min, curr_bin_max = self.bins[bins_index][0], self.bins[bins_index][1]
        for k, img_gt in enumerate(gts_sorted):
            if curr_bin_min <= img_gt <= curr_bin_max:
                curr_bin.append(gts_inds_sorted_by_gts[k])
                curr_bin_max = max(curr_bin_max, img_gt)
            else:
                bins.append((curr_bin_min, curr_bin_max))
                inds_in_bins.append(curr_bin)
                if self.bins is None:
                    curr_bin_min, curr_bin_max = img_gt, img_gt + self.bins_size
                else:
                    bins_index += 1
                    curr_bin_min, curr_bin_max = self.bins[bins_index][0], self.bins[bins_index][1]
                curr_bin = [gts_inds_sorted_by_gts[k]]
        bins.append((curr_bin_min, curr_bin_max))
        inds_in_bins.append(curr_bin)

        yield_list = []
        for i in range(self.num_samples):
            random_bin_inds = random.choice(inds_in_bins)
            random_ind_from_bin = random.choice(random_bin_inds)
            yield_list.append(random_ind_from_bin)
        yield from yield_list

    def __len__(self) -> int:
        return self.num_samples
