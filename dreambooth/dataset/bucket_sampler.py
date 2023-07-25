import random
from typing import Tuple, Dict, List

import numpy as np
from torch import randperm
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.sampler import BatchSampler

from dreambooth.dataset.db_dataset import DbDatasetForResolution


# x10 slower than random.shuffle
def torch_shuffle(xs):
    return [xs[i] for i in randperm(len(xs))]


def build_weighted_sample_indexes(weights, output_length: int):
    shuffeled_index_weights = list(enumerate(weights))
    # shuffle cumsum order
    random.shuffle(shuffeled_index_weights)
    shuffled_indexes, shuffeled_weights = zip(*shuffeled_index_weights)

    cumsum_weights = np.cumsum(np.insert(shuffeled_weights, 0, 0.0))
    scale = output_length / cumsum_weights[-1]
    cumsum_weights = cumsum_weights * scale
    indexes = np.array(range(len(cumsum_weights)))
    weighted_indexes = np.interp(np.array(range(output_length)),
                                 cumsum_weights, indexes).astype(np.int64)
    unshuffeled_weighted_indexes = [shuffled_indexes[i]
                                    for i in weighted_indexes]
    return unshuffeled_weighted_indexes


class InstanceBalancedBatchSampler:
    def __init__(self,
                 batch_size: int,
                 instance_dataset: DbDatasetForResolution,
                 class_dataset: DbDatasetForResolution,
                 interleave_size: int = 1):
        # half of batch are instance and rests are class
        # instance are oversampled
        assert batch_size % 2 == 0
        assert len(class_dataset) >= len(instance_dataset)
        self.batch_size = batch_size
        self._shorter_dataset = instance_dataset
        self._longer_dataset = class_dataset
        self._interleave_size = interleave_size

        self.concated_dataset = ConcatDataset(
            [self._longer_dataset, self._shorter_dataset])

    def _build_indexes(self):
        longer_indexes = list(range(len(self._longer_dataset)))
        random.shuffle(longer_indexes)

        shorter_weights = self._shorter_dataset.get_weights()
        shorter_indexes = build_weighted_sample_indexes(
            shorter_weights, len(longer_indexes))
        random.shuffle(shorter_indexes)

        assert len(longer_indexes) == len(shorter_indexes)
        # add offset for concated dataset
        shorter_indexes = [len(self._longer_dataset) + index
                           for index in shorter_indexes]

        def split_list(xs, n):
            return [xs[i:i + n] for i in range(0, len(xs), n)
                    if i + n < len(xs)]
        return list(sum(map(lambda xy: sum(xy, []),
                    zip(split_list(longer_indexes, self._interleave_size),
                        split_list(shorter_indexes, self._interleave_size))),
                        []))

    def __iter__(self):
        indexes = self._build_indexes()
        for i in range(len(indexes) // self.batch_size):
            start_index = i * self.batch_size
            batched_indexes = indexes[start_index:start_index
                                      + self.batch_size]
            yield batched_indexes

    def __len__(self):
        return 2 * len(self._longer_dataset) // self.batch_size


class ResolutionedInstanceBalancedBatchSampler(BatchSampler):
    def __init__(self,
                 batch_size: int,
                 samplers: List[InstanceBalancedBatchSampler]):
        self.sampler = None
        self._samplers = samplers
        self.batch_size = batch_size
        self._concated_dataset = ConcatDataset(
            [sampler.concated_dataset for sampler in self._samplers])

        self._length = len(self._build_sampler_indexes())

    def _build_sampler_indexes(self):
        sampler_indexes = sum([[sampler_index] * len(sampler)
                               for sampler_index, sampler
                               in enumerate(self._samplers)], [])
        random.shuffle(sampler_indexes)
        return sampler_indexes

    def __iter__(self):
        sampler_indexes = self._build_sampler_indexes()
        offset_indexes = np.cumsum([0] + list(map(len, self._samplers)))
        sampler_iterators = [(start_index, iter(sampler))
                             for start_index, sampler
                             in zip(offset_indexes, self._samplers)]
        for sampler_index in sampler_indexes:
            try:
                offset_index, sampler =\
                    sampler_iterators[sampler_index]
                local_indexes = next(sampler)
                global_indexes = [offset_index + local_index
                                  for local_index in local_indexes]
                yield global_indexes
            except StopIteration:
                continue

    def __len__(self):
        return self._length

    @staticmethod
    def build_dataset_and_sampler(
            batch_size: int,
            res_instance_datasets: Dict[Tuple[int, int],
                                        DbDatasetForResolution],
            res_class_datasets: Dict[Tuple[int, int],
                                     DbDatasetForResolution],
            interleave_size: int = 1
            ) -> Tuple[BatchSampler, Dataset]:
        print(res_instance_datasets.keys()) 
        print(res_class_datasets.keys()) 
        assert set(res_instance_datasets.keys())\
            == set(res_class_datasets.keys())
        samplers = [InstanceBalancedBatchSampler(
                batch_size,
                res_instance_datasets[res],
                res_class_datasets[res],
                interleave_size=interleave_size)
                for res in res_instance_datasets.keys()]
        res_sampler = ResolutionedInstanceBalancedBatchSampler(
            batch_size, samplers)
        return (res_sampler, res_sampler._concated_dataset)

