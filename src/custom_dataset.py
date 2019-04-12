import numpy as np
import torch
from torch.utils import data as data_utils

class DAMICDataset(data_utils.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, targets):
        """Reads source and target sequences from txt files."""
        self.src_seqs = data
        self.trg_seqs = targets
        self.num_total_seqs = len(targets)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        return src_seq, trg_seq

    def __len__(self):
        return self.num_total_seqs


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        sequences = list(sequences)
        lengths = [len(seq) for seq in sequences]
        u_len = len(sequences[0][0])
        for row in sequences:
            diff = max(lengths) - len(row)
            for i in range(diff):
                row.append([0]*u_len)
        return np.array(sequences), np.array(lengths)

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    src_seqs = torch.from_numpy(src_seqs).long()
    trg_seqs = torch.from_numpy(trg_seqs).float()
    src_lengths = torch.from_numpy(src_lengths).long()
    trg_lengths = torch.from_numpy(trg_lengths).long()

    # print(src_seqs.size())

    return src_seqs, src_lengths, trg_seqs, trg_lengths