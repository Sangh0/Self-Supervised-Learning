import numpy as np

import torch
import torch.nn as nn


class NTXentLoss(nn.Module):

    def __init__(self, batch_size: int, temperature: float, use_cosine_similarity: bool=True):
        super(NTXentLoss, self).__init__()

        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_function = self._get_similarity_func(use_cosine_similarity)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    def _get_similarity_func(self, use_cosine_similarity):
        if use_cosine_similarity:
            similarity_func = nn.CosineSimilarity(dim=-1)
        else:
            similarity_func = self._dot_similarity

        return similarity_func

    def _dot_similarity(self, x, y):
        v = torch.tensordot(x.unsqueeze(dim=1), y.T.unsqueeze(dim=0), dims=2)
        return v

    def forward(self, i, j):
        representations = torch.cat([i, j], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, self.-batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).long()
        
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)