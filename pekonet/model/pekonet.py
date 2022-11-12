import torch.nn as nn

from pekonet.model.atsm import ATSM
from pekonet.model.ljpm import LJPM
from pekonet.model.dann import DANN
from pekonet.model.mada import MADA


class PekoNet(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(PekoNet, self).__init__()

        self.atsm = ATSM(config=config)
        self.dann = DANN(config=config)
        self.ljpm = LJPM(config=config)
        self.articles_mada = \
            MADA(classes_number=int(config.get('data', 'articles_number')))
        self.accusations_mada = \
            MADA(classes_number=int(config.get('data', 'accusations_number')))


    def initialize_multiple_gpus(self, gpus, *args, **kwargs):
        self.atsm = nn.DataParallel(module=self.atsm, device_ids=gpus)
        self.dann = nn.DataParallel(module=self.dann, device_ids=gpus)
        self.ljpm = nn.DataParallel(module=self.ljpm, device_ids=gpus)
        self.articles_mada = nn.DataParallel(
            module=self.articles_mada
            , device_ids=gpus)
        self.accusations_mada = nn.DataParallel(
            module=self.accusations_mada
            , device_ids=gpus)


    def forward(self, data, mode, acc_result=None):
        if mode == 'serve':
            tensor = self.atsm(data, mode)
            output = self.ljpm(tensor, mode)

            return output
        # mode == 'train' or 'eval'
        else:
            loss = 0

            # data['type'] == 1 -> TCI
            if data['type'] == 1:
                data = self.atsm(data=data, mode=mode)

                loss += self.dann(data=data)

                outputs = self.ljpm(data=data, mode=mode, acc_result=acc_result)

                loss += outputs['loss']
                acc_result = outputs['acc_result']

                loss += self.articles_mada(
                    domain_prediction=outputs['middle']
                    , class_prediction=outputs['final']
                    , domain_label=data['type'])
                loss += self.accusations_mada(
                    domain_prediction=outputs['middle']
                    , class_prediction=outputs['final']
                    , domain_label=data['type'])

                return {
                    'loss': loss
                    , 'acc_result': acc_result
                    , 'TorS': data['type']
                }
            # data['type'] == 0 -> CNewSum
            else:
                loss += self.atsm(data=data, mode=mode)
                loss += self.dann(data=data)

                return loss