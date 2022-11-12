import torch.nn as nn

from pekonet.model.utils import GRL


class DANN(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(DANN, self).__init__()

        self.max_len = config.getint('data', 'max_len')

        self.grl = GRL()
        self.domain_cls = nn.Sequential(
            nn.Linear(in_features=self.max_len, out_features=1)
            , nn.Sigmoid())

        self.criterion = nn.BCELoss()


    def forward(self, data):
        atsm_prediction = data['summary']
        domain_label = data['type']

        atsm_prediction = nn.functional.pad(
            input=atsm_prediction
            , pad=(0, self.max_len-atsm_prediction.size(dim=1)))

        grl_result = self.grl.apply(atsm_prediction)
        domain_cls_pred = self.domain_cls(grl_result)

        loss = self.criterion(domain_cls_pred, domain_label)

        return loss