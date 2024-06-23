# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import torch.nn
import math
from typing import Any, Dict, List
import logging

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics, utils
logger = logging.getLogger(__name__)

@register_criterion('bce_loss')
class BCECriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none') #'sum')

    def forward(self, model, sample, reduce=True):
        assert reduce, RuntimeError(f"BCECriterion: reduce must be True")
        logits, _ = model(**sample['net_input'])
        targets = model.get_targets(sample, logits)
        loss = self.compute_loss(logits, targets)
        if 'target_mask' in sample.keys():
            loss = loss * sample['target_mask'].view(loss.shape)
        loss = loss.sum()

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.item(),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, logits, targets):
        targets = targets.view(-1) 
        loss = self.criterion(logits.view(-1), targets)
        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
