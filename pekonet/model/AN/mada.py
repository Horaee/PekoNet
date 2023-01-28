import torch
import torch.nn as nn

from pekonet.model.AN.utils import GRL


class MADA(nn.Module):
    def __init__(self, classes_number, max_len, *args, **kwargs):
        super(MADA, self).__init__()

        self.grl = GRL()
        self.domain_clses = [
            nn.Sequential(nn.Linear(
                in_features=max_len, out_features=1)
                , nn.Sigmoid()).cuda()
            for _ in range(classes_number)
        ]

        self.criterion = nn.CrossEntropyLoss()


    def forward(
            self
            , fact_embedding
            , class_predictions
            , domain_label):
        processed_class_predictions = []

        class_predictions = torch.squeeze(input=class_predictions, dim=0)
        domain_label = torch.squeeze(input=domain_label, dim=0)

        for one_prediction in class_predictions:
            processed_class_predictions.append(
                one_prediction[1].item() - one_prediction[0].item()
            )

        processed_class_predictions = torch.Tensor(
            processed_class_predictions).cuda()
        processed_class_predictions = nn.functional.softmax(
            processed_class_predictions
            , dim=0)

        grl_result = self.grl.apply(fact_embedding)

        domain_cls_predictions = []

        for cls in self.domain_clses:
            domain_cls_predictions.append(cls(grl_result))

        domain_cls_predictions = torch.Tensor(domain_cls_predictions).cuda()
        domain_label = domain_label.to(torch.float32)

        loss = self.criterion(
            torch.mul(domain_cls_predictions, processed_class_predictions)
            , domain_label)

        return loss