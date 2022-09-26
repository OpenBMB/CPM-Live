import torch.utils.data as data
import linecache
import json
import bmtrain as bmt


class JsonlDataset(data.Dataset):
    def __init__(self, filename, process, **kwargs):
        self._filename = filename
        self._total_data = 0
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
        self.process = process
        self.kwargs = kwargs

    def __len__(self):
        return self._total_data

    def __getitem__(self, i):
        item = json.loads(linecache.getline(self._filename, i + 1))
        processed = self.process(item, **self.kwargs)
        return processed


class DistributedDataLoader:
    def __init__(
        self, dataset, shuffle=False, num_workers=0, batch_size=1, collate_fn=None, **kwargs
    ):
        self.sampler = data.DistributedSampler(
            dataset, shuffle=shuffle, rank=bmt.rank(), num_replicas=bmt.world_size()
        )
        self.loader = data.DataLoader(
            dataset,
            shuffle=False,
            sampler=self.sampler,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **kwargs,
        )
        self.epoch = 0
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.epoch += 1
        self.sampler.set_epoch(self.epoch)
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)
