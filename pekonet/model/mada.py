import torch
import torch.nn as nn

from pekonet.model.utils import GRL


class MADA(nn.Module):
    def __init__(self, classes_number, *args, **kwargs):
        super(MADA, self).__init__()

        self.grl = GRL()
        self.domain_clses = [
            nn.Sequential(nn.Linear(
                in_features=self.max_len, out_features=1)
                , nn.Sigmoid())
            for _ in range(classes_number)
        ]

        self.criterion = nn.BCELoss()


    def forward(
            self
            , domain_prediction
            , class_prediction
            , domain_label):
        class_prediction_processed = []

        for one_prediction in class_prediction:
            class_prediction_processed.append(
                one_prediction[1].item() - one_prediction[0].item()
            )

        class_prediction_processed = torch.Tensor(class_prediction_processed)
        class_prediction_processed = nn.functional.softmax(
            class_prediction_processed
            , dim=0)

        grl_result = self.grl.apply(domain_prediction)

        domain_cls_pred = []

        for cls in self.domain_clses:
            domain_cls_pred.append(cls(grl_result))

        loss = self.criterion(
            domain_cls_pred*class_prediction_processed
            , domain_label)

        return loss