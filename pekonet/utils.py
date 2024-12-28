import torch.nn as nn


class MultiLabelsLoss(nn.Module):
    def __init__(self, class_number, *args, **kwargs):
        super(MultiLabelsLoss, self).__init__()
        
        self.criterions = []

        for _ in range(class_number):
            self.criterions.append(nn.CrossEntropyLoss())


    # The size of predictions is [batch_size, task_number, 2].
    def forward(self, predictions, labels, *args, **kwargs):
        loss = 0

        # Size of predictions = [batch_size, task_number, 2].
        for task_index in range(predictions.size(1)):
            # Size of predictions[:, task_index, :] = [batch_size, 2].
            one_task_outputs = predictions[:, task_index, :]

            # Size of labels[:, task_index] = [batch_size].
            one_task_labels = labels[:, task_index]

            loss += self.criterions[task_index](
                one_task_outputs
                , one_task_labels)

        return loss