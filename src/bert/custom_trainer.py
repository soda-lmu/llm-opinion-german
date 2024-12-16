import torch.nn as nn
from transformers import Trainer

from src.bert.db_loss import DBloss
from src.bert.focal_loss import focal_loss
from src.bert.resample_loss import ResampleLoss


class CustomTrainer(Trainer):
    def __init__(self, *args, loss_type='default', **kwargs):
        self.class_freq = kwargs.pop('class_freq', None)
        self.train_num = kwargs.pop('train_num', None)
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        print(f'Using {self.loss_type} loss')
        if self.loss_type == 'resample' and (self.class_freq is None or self.train_num is None):
            raise ValueError('class_freq and train_num must be provided for resample loss') 
        if self.loss_type == 'dbloss':
            if self.class_freq is None or self.train_num is None:
                raise ValueError('class_freq and train_num must be provided for DB loss')
            self.dbloss = DBloss(class_freq=self.class_freq, train_num=self.train_num)

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        """
        Custom loss computation supporting default, focal, and DB loss.

        Args:
            model: The model instance.
            inputs: Batch inputs containing 'input_ids', 'attention_mask', 'labels', etc.
            return_outputs: Whether to return the model outputs along with the loss.

        Returns:
            Loss tensor, and optionally the model outputs.
        """
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        if self.loss_type == 'focal':
            loss = focal_loss(logits, labels)
        elif self.loss_type == 'dbloss':
            loss = self.dbloss(logits, labels)
        elif self.loss_type == 'resample':
            resample_loss = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                             class_freq=self.class_freq, train_num=self.train_num)
            loss = resample_loss(logits, labels)
        else:  # default
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.float().view(-1, self.model.config.num_labels))

        return (loss, outputs) if return_outputs else loss