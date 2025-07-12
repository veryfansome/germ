from torch.utils.data import Sampler
import random

class BucketBatchSampler(Sampler):
    """
    Yields indices such that each batch contains sequences whose lengths lie in the same logarithmic bucket.
    Designed to work with DecoderModel from decoder_v2, to avoid PAD tokens that would cause
    F.scaled_dot_product_attention to take the slower path.
    """
    def __init__(self, lengths, batch_size, n_buckets=50, shuffle=True):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle

        # bucket by log-length so 60 and 64 land together, 10 and 12 together…
        loglen = [int(len_.bit_length()) for len_ in lengths]
        buckets = [[] for _ in range(max(loglen)+1)]
        for idx,ll in enumerate(loglen):
            buckets[ll].append(idx)

        # chunk each bucket into batch-sized chunks
        self.batches = []
        for bucket in buckets:
            if shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                chunk = bucket[i:i+batch_size]
                if len(chunk)==batch_size:          # drop_last
                    self.batches.append(chunk)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)