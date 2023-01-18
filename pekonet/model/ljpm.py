import torch.nn as nn
import torch

# from pekonet.model.ljpm_encoder import LJPMEncoder
from pekonet.model.ljpm_predictor import LJPMPredictor
from pekonet.utils import MultiLabelSoftmaxLoss
from pekonet.evaluation import ConfusionMatrix
# from legal_judgment_prediction.utils import MultiLabelSoftmaxLoss
# from legal_judgment_prediction.evaluation import multi_label_accuracy


class LJPM(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LJPM, self).__init__()

        # model_path = config.get('model', 'ljpm_model_path')
        hidden_size = config.getint('model', 'hidden_size')
        articles_number = config.getint('data', 'articles_number')
        accusations_number = config.getint('data', 'accusations_number')

        # self.bert = LJPMEncoder(model_path=model_path)
        self.fc = LJPMPredictor(
            hidden_size=hidden_size
            , articles_number=articles_number
            , accusations_number=accusations_number)

        # TODO
        self.criterion = {
            'article': MultiLabelSoftmaxLoss(task_number=articles_number)
            , 'accusation': MultiLabelSoftmaxLoss(
                task_number=accusations_number)
        }
        self.evaluation = ConfusionMatrix()


    # TODO
    def forward(
            self
            , mode
            , cls_embeddings
            , labels
            # , summary_embedding
            # , encodings
            , cm_result=None):
        if mode == 'serve':
            # fact_ids = torch.unsqueeze(input=fact_ids, dim=0)
            # fact_embedding = self.bert(input=fact_ids)
            # feature = self.fc(tensor=fact_embedding)

            # return feature
            print('Hello World')
        # mode == 'train' or 'eval'
        else:
            aa_results = self.fc(tensors=cls_embeddings)

            loss = 0
            for name in ['article', 'accusation']:
                # aa_results[name] = torch.argmax(input=aa_results[name], dim=2)

                loss += self.criterion[name](
                    preds=aa_results[name]
                    , labels=labels[name])

            if cm_result == None:
                cm_result = {
                    'article': None
                    , 'accusation': None
                }

            for name in ['article', 'accusation']:
                cm_result[name] = self.evaluation(
                    preds=aa_results[name]
                    , labels=labels[name]
                    , cm_result=cm_result[name])

            return {'loss': loss, 'cm_result': cm_result}


    # def forward(self, mode, fact_ids, data=None, acc_result=None):
    #     if mode == 'serve':
    #         fact_ids = torch.unsqueeze(input=fact_ids, dim=0)
    #         fact_embedding = self.bert(input=fact_ids)
    #         feature = self.fc(tensor=fact_embedding)

    #         return feature
    #     # mode == 'train' or 'eval'
    #     else:
    #         fact_embedding = self.bert(input=fact_ids)
    #         feature = self.fc(tensor=fact_embedding)

    #         print(feature)
    #         input()

    #         loss = 0
    #         for name in ['article', 'accusation']:
    #             loss += self.criterion[name](
    #                 outputs=feature[name]
    #                 , labels=data[name])

    #         if acc_result == None:
    #             acc_result = {
    #                 'article': None
    #                 , 'accusation': None
    #             }

    #         for name in ['article', 'accusation']:
    #             acc_result[name] = self.accuracy_function[name](
    #                 outputs=feature[name]
    #                 , label=data[name]
    #                 , result=acc_result[name])

    #         return {
    #             'fact_embedding': fact_embedding
    #             , 'feature': feature
    #             , 'loss': loss
    #             , 'acc_result': acc_result
    #         }
