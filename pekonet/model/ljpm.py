import torch.nn as nn

from pekonet.model.ljpm_predictor import LJPMPredictor
from pekonet.utils import MultiLabelsLoss
from pekonet.evaluation import ConfusionMatrix


class LJPM(nn.Module):
    # Checked.
    def __init__(self, config, *args, **kwargs):
        super(LJPM, self).__init__()

        hidden_size = config.getint('model', 'hidden_size')
        articles_number = config.getint('data', 'articles_number')
        accusations_number = config.getint('data', 'accusations_number')

        self.fcs = LJPMPredictor(
            hidden_size=hidden_size
            , articles_number=articles_number
            , accusations_number=accusations_number)

        self.criterions = {
            'article': MultiLabelsLoss(class_number=articles_number)
            , 'accusation': MultiLabelsLoss(class_number=accusations_number)
        }
        self.evaluation = ConfusionMatrix()


    def forward(
            self
            , mode
            , cls_embeddings
            , labels
            , cm_results=None
            , *args
            , **kwargs):
        # Checked.
        # If mode is 'train', 'validate' or 'test'.
        if mode != 'serve':
            aa_results = self.fcs(tensors=cls_embeddings)

            loss = 0
            task_names = ['article', 'accusation']

            for task_name in task_names:
                loss += self.criterions[task_name](
                    predictions=aa_results[task_name]
                    , labels=labels[task_name])

            if cm_results == None:
                cm_results = {
                    'article': None
                    , 'accusation': None
                }

            for task_name in task_names:
                cm_results[task_name] = self.evaluation(
                    predictions=aa_results[task_name]
                    , labels=labels[task_name]
                    , cm_results=cm_results[task_name])

            return {'loss': loss, 'cm_results': cm_results}

        # If mode is 'serve'.
        feature = self.fcs(tensors=cls_embeddings)

        return feature
