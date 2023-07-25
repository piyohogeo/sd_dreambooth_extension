import json
import random
import threading
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
    assert len(unshuffeled_weighted_indexes) == output_length
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

        longer_weights = self._longer_dataset.get_weights()
        assert np.isclose(np.sum(longer_weights), 1.0), f'sum(longer_weights): {np.sum(longer_weights)}'
        sampling_magnify = 1.0 / np.min(longer_weights)
        self._normalized_dataset_length = int(len(longer_weights) * sampling_magnify)
        if self._normalized_dataset_length % self._interleave_size != 0:
            self._normalized_dataset_length = ((self._normalized_dataset_length
                                                // self._interleave_size + 1)
                                               * self._interleave_size)

    def _build_indexes(self):
        shorter_weights = self._shorter_dataset.get_weights()
        longer_weights = self._longer_dataset.get_weights()
        shorter_indexes = build_weighted_sample_indexes(
            shorter_weights, self._normalized_dataset_length)
        longer_indexes = build_weighted_sample_indexes(
            longer_weights, self._normalized_dataset_length)
        random.shuffle(shorter_indexes)
        random.shuffle(longer_indexes)

        assert len(longer_indexes) == len(shorter_indexes)
        # add offset for concated dataset
        shorter_indexes = [len(self._longer_dataset) + index
                           for index in shorter_indexes]

        def split_list(xs, n):
            return [xs[i:i + n] for i in range(0, len(xs), n)
                    if i + n <= len(xs)]
        return list(sum(map(lambda xy: xy[0] + xy[1],
                    zip(split_list(longer_indexes, self._interleave_size),
                        split_list(shorter_indexes, self._interleave_size))),
                        []))

    def __iter__(self):
        indexes = self._build_indexes()
        assert len(indexes) // self.batch_size == self.__len__()
        with open('/mnt/d/log/tmp/indexes%08x.json' % threading.get_ident(),
                  'w', encoding='utf-8') as f:
            json.dump(indexes, f)
        for i in range(len(indexes) // self.batch_size):
            start_index = i * self.batch_size
            batched_indexes = indexes[start_index:start_index
                                      + self.batch_size]
            yield batched_indexes

    def __len__(self):
        return 2 * self._normalized_dataset_length // self.batch_size


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
        with open('/mnt/d/log/tmp/sampler_indexes%08x.json' % threading.get_ident(),
                  'w', encoding='utf-8') as f:
            json.dump(sampler_indexes, f)
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


class RepeatedBatchSampler:
    def __init__(self, base_sampler, repeat_length):
        self._base_sampler = base_sampler
        self._repeat_length = repeat_length

    def __iter__(self):
        base_iter = iter(self._base_sampler)
        while True:
            try:
                to_repeat_batches = []
                for _ in range(self._repeat_length):
                    to_repeat_batches.append(next(base_iter))
                # yield twice
                for i in range(2):
                    for batch_indexes in to_repeat_batches:
                        yield batch_indexes
            except StopIteration:
                return

    def __len__(self):
        return ((len(self._base_sampler) // self._repeat_length)
                * self._repeat_length * 2)
