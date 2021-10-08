# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect

import numpy as np
from torch.utils.data.dataloader import default_collate

from . import FairseqDataset


class MultitaskDataset(FairseqDataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            curr_len = len(e)
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1):
        super(MultitaskDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.real_sizes = [len(d) for d in self.datasets]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        sample = self.datasets[dataset_idx][sample_idx]
        sample["dataset_idx"] = dataset_idx
        return sample

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if samples is not None and  len(samples) > 0:
            dataset_idx = samples[0]["dataset_idx"]
        else:
            dataset_idx = 0

        if hasattr(self.datasets[dataset_idx], "collater"):
            return self.datasets[dataset_idx].collater(samples, **extra_args)
        else:
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return getattr(self.datasets[dataset_idx], attr, None)

    @property
    def sizes(self):
        _dataset_sizes = []
        for ds in self.datasets:
            if isinstance(ds.sizes, np.ndarray):
                _dataset_sizes.append(ds.sizes)
            else:
                # Only support underlying dataset with single size array.
                assert isinstance(ds.sizes, list)
                _dataset_sizes.append(ds.sizes[0])
        return np.concatenate(_dataset_sizes)

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets)

    def ordered_indices(self):
        ordered_indices = []
        for i, dataset in enumerate(self.datasets):
            indice = dataset.ordered_indices()
            ordered_indices.append(indice)
        return ordered_indices

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cumulative_sizes, self.datasets):
            real_size = len(ds)
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch([(i - frm) % real_size for i in indices if frm <= i < to])
            frm = to

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        batch_samplers = []
        for i, dataset in enumerate(self.datasets):
            batch_sampler = dataset.batch_by_size(
                indices[i],
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
            if i > 0:
                for batch in batch_sampler:
                    batch += self.cumulative_sizes[i-1]
            if self.sample_ratios[i] != 1.0:
                batch_sampler = np.array(batch_sampler)
                batch_sampler = np.random.choice(batch_sampler, int(len(batch_sampler)*self.sample_ratios[i]))
                batch_sampler = list(batch_sampler)
            batch_samplers.extend(batch_sampler)
        return batch_samplers

    def filter_indices_by_size(self, indices, max_sizes):
        return indices

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(d.can_reuse_epoch_itr_across_epochs for d in self.datasets)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
