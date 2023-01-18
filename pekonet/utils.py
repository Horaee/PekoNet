import torch.nn as nn


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, task_number):
        super(MultiLabelSoftmaxLoss, self).__init__()
        
        self.criterion = []

        for _ in range(task_number):
            self.criterion.append(nn.CrossEntropyLoss())


    # The size of outputs is [batch_size, task_number, 2].
    def forward(self, preds, labels):
        loss = 0

        # Size of `outputs` = [batch_size, task_number, 2].
        for task_index in range(preds.size(1)):
            # Size of `outputs[:, task_index, :]` = [batch_size, 2].
            one_task_outputs = preds[:, task_index, :]

            # Size of `labels[:, task_index]` = [batch_size].
            one_task_labels = labels[:, task_index]

            loss += self.criterion[task_index](
                one_task_outputs
                , one_task_labels)

        return loss