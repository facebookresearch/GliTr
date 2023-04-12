import math
import torch
import torch.distributed as dist
from operator import itemgetter
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler

class RandomSubsetSampler(torch.utils.data.Sampler):
    r"""based on RandomSampler
    Samples a subset of elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
        subset_samples (int): total number of elemenets to draw in an epoch; must be less than the dataset size
    """
    def __init__(self, data_source, replacement: bool = False,
                 num_samples = None, generator=None, subset_samples=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.subset_samples = subset_samples if subset_samples is not None else len(self.data_source)

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return self.subset_samples
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            yield from torch.randperm(n, generator=generator).tolist()[:self.subset_samples]

    def __len__(self) -> int:
        return self.num_samples


class DistributedSamplerWrapper(DistributedSampler):
    """
    from: https://github.com/catalyst-team/catalyst/blob/678dc06eda1848242df010b7f34adb572def2598/catalyst/data/sampler.py#L485
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas = None,
        rank = None,
        shuffle = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        self.sampler = sampler

    def __iter__(self):
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    from: https://github.com/catalyst-team/catalyst/blob/678dc06eda1848242df010b7f34adb572def2598/catalyst/data/dataset.py#L6
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)
