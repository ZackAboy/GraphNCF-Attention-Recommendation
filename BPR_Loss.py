import torch
import torch.nn as nn

class BPR_Loss(nn.Module):
    def __init__(self, batch_size: int, l2_decay_ratio: float = 1e-5, l1_decay_ratio: float = 1e-6):
        super(BPR_Loss, self).__init__()
        self.batch_size = batch_size
        self.l2_decay = l2_decay_ratio
        self.l1_decay = l1_decay_ratio

    def forward(self, users, pos_items, neg_items):
        pos_scores = torch.mul(users, pos_items).sum(dim=1)
        neg_scores = torch.mul(users, neg_items).sum(dim=1)

        log_prob = nn.LogSigmoid()(pos_scores - neg_scores).sum()
        l2_reg = self.l2_decay * (users.norm(dim=1).pow(2).sum() +
                                pos_items.norm(dim=1).pow(2).sum() +
                                neg_items.norm(dim=1).pow(2).sum())

        l1_reg = self.l1_decay * (users.norm(p=1, dim=1).sum() +
                                pos_items.norm(p=1, dim=1).sum() +
                                neg_items.norm(p=1, dim=1).sum())

        regularization = l2_reg + l1_reg

        loss = regularization - log_prob
        loss /= self.batch_size
        return loss