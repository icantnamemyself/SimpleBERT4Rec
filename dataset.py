import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as Data

# use 0 to padding
DATA = pd.read_csv('data/movielen_lenth_30_cold_10.csv', header=None).values + 1


class TrainDataset(Data.Dataset):
    def __init__(self, mask_prob, max_len):
        self.data = DATA
        self.num_item = DATA.max()
        self.num_user = DATA.shape[0]
        self.mask_token = self.num_item + 1
        self.mask_prob = mask_prob
        self.max_len = max_len

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, :-2]

        tokens = []
        labels = []

        for s in seq:
            if s != 0:
                prob = random.random()
                if prob < self.mask_prob:
                    tokens.append(self.mask_token)
                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)
            else:
                tokens.append(s)
                labels.append(0)

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)


class EvalDataset(Data.Dataset):
    def __init__(self, max_len, sample_size, mode, enable_sample):
        self.data = DATA
        self.num_item = DATA.max()
        self.num_user = DATA.shape[0]
        self.mask_token = self.num_item + 1
        self.max_len = max_len
        self.sample_size = sample_size
        self.mode = mode
        self.enable_sample = enable_sample

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, :-2] if self.mode == 'val' else self.data[index, :-1]
        pos = self.data[index,-2] if self.mode == 'val' else self.data[index, -1]
        negs = []

        if self.enable_sample:
            seen = set(seq)
            seen.update([pos])
            while len(negs) < self.sample_size:
                candidate = np.random.randint(0, self.num_item) + 1
                while candidate in seen or candidate in negs:
                    candidate = np.random.randint(0, self.num_item) + 1
                negs.append(candidate)
        else:
            negs = np.arange(0, self.num_item + 1)
            np.delete(negs, list(set(seq)) + [pos] + [0])
            negs = list(negs)


        answers = [pos] + negs
        labels = [1] + [0] * len(negs)

        seq = list(seq)
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(answers), torch.LongTensor(labels)

