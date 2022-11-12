import torch.nn as nn
import torch

from pekonet.model.ljpm_encoder import LJPMEncoder
from pekonet.model.ljpm_predictor import LJPMPredictor
from legal_judgment_prediction.utils import MultiLabelSoftmaxLoss
from legal_judgment_prediction.evaluation import multi_label_accuracy


class LJPM(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LJPM, self).__init__()

        model_path = config.get('ljpm', 'model_path')
        hidden_size = config.getint('ljpm', 'hidden_size')
        articles_number = config.getint('data', 'articles_number')
        article_sources_number = config.getint('data', 'article_sources_number')
        accusations_number = config.getint('data', 'accusations_number')

        self.bert = LJPMEncoder(model_path=model_path)
        self.fc = LJPMPredictor(
            hidden_size=hidden_size
            , articles_number=articles_number
            , article_sources_number=article_sources_number
            , accusations_number=accusations_number)

        # TODO:
        self.criterion = {
            'article': MultiLabelSoftmaxLoss(task_number=articles_number)
            , 'article_source': MultiLabelSoftmaxLoss(
                task_number=article_sources_number)
            , 'accusation': MultiLabelSoftmaxLoss(
                task_number=accusations_number)
        }
        self.accuracy_function = {
            'article': multi_label_accuracy,
            'article_source': multi_label_accuracy,
            'accusation': multi_label_accuracy
        }


    # TODO:
    def forward(self, data, mode, acc_result=None):
        if mode == 'serve':
            data = torch.unsqueeze(input=data, dim=0)
            output = self.bert(input=data)
            output = self.fc(tensor=output)

            return output
        # mode == 'train' or 'eval'
        else:
            fact = data['text'] if data['TorS'] == 2 else data['summary']
            middle = self.bert(input=fact)
            final = self.fc(tensor=middle)

            loss = 0
            for name in ['article', 'article_source', 'accusation']:
                loss += self.criterion[name](
                    outputs=final[name]
                    , labels=data[name])

            if acc_result == None:
                acc_result = {
                    'article': None
                    , 'article_source': None
                    , 'accusation': None
                }

            for name in ['article', 'article_source', 'accusation']:
                acc_result[name] = self.accuracy_function[name](
                    outputs=final[name]
                    , label=data[name]
                    , result=acc_result[name])

            return {
                'middle': middle
                , 'final': final
                , 'loss': loss
                , 'acc_result': acc_result
            }
