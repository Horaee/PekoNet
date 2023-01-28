import torch
import torch.nn as nn

from pekonet.model.AN.utils import GRL


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
        # summary_ids = torch.unsqueeze(input=data['summary'], dim=0)
        summary_ids = data['summary']

        # data['type'] == 3 -> CNewSum
        domain_label = torch.Tensor([0])

        # data['type'] == 2 -> TCI Summary
        if data['type'] == 2:
            domain_label = torch.Tensor([1])

        domain_label = domain_label.cuda()

        # summary_ids = nn.functional.pad(
        #     input=summary_ids
        #     , pad=(0, self.max_len-summary_ids.size(dim=1)))

        grl_result = self.grl.apply(summary_ids)
        # grl_result = torch.squeeze(input=grl_result, dim=0)
        grl_result = grl_result.to(torch.float32)

        domain_cls_prediction = self.domain_cls(grl_result)

        loss = self.criterion(domain_cls_prediction, domain_label)

        return loss