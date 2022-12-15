import torch
import torch.nn as nn

from pekonet.model.bart import BART
from pekonet.model.ljpm import LJPM
# from pekonet.model.dann import DANN
# from pekonet.model.mada import MADA


class PekoNet(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(PekoNet, self).__init__()

        self.bart = BART(config=config)
        self.ljpm = LJPM(config=config)
        # self.dann = DANN(config=config)
        # self.articles_mada = \
        #     MADA(
        #         classes_number=config.getint('data', 'articles_number')
        #         , max_len=config.getint('model', 'hidden_size')
        #     )
        # self.accusations_mada = \
        #     MADA(
        #         classes_number=config.getint('data', 'accusations_number')
        #         , max_len=config.getint('model', 'hidden_size')
        #     )


    def initialize_multiple_gpus(self, gpus, *args, **kwargs):
        self.bart = nn.DataParallel(module=self.bart, device_ids=gpus)
        self.ljpm = nn.DataParallel(module=self.ljpm, device_ids=gpus)
        # self.dann = nn.DataParallel(module=self.dann, device_ids=gpus)
        # self.articles_mada = nn.DataParallel(
        #     module=self.articles_mada
        #     , device_ids=gpus)
        # self.accusations_mada = nn.DataParallel(
        #     module=self.accusations_mada
        #     , device_ids=gpus)


    def forward(self, data, mode, acc_result=None):
        if mode == 'serve':
            print('Hello World')
            # tensor = self.bart(data, mode)
            # output = self.ljpm(tensor, mode)

            # return output
        # mode == 'train' or 'eval'
        else:
            loss = 0

            outputs = self.bart(data=data, mode=mode)

            for index in range(len(data['text'])):
                # No summary -> TCI
                if torch.count_nonzero(data['summary'][index]).item() == 0:
                    summary_embedding = \
                        outputs.decoder_hidden_states[-1][index][0]

                    summary_embedding = summary_embedding.view(1, 1, -1)

                    encodings = {
                        'article': data['article'][index]
                        , 'accusation': data['accusation'][index]
                    }

                    results = self.ljpm(
                        mode=mode
                        , summary_embedding=summary_embedding
                        , encodings=encodings
                        , acc_result=acc_result)

                    loss += results['loss']
                    acc_result = results['acc_result']
                # Have summary -> CNS
                else:
                    loss += outputs.loss

            return {
                'loss': loss
                , 'acc_result': acc_result
            }

            # for one_data in data:
            #     print(one_data)
            #     input()

            #     # one_data['type'] == 1 -> TCI Text
            #     if one_data['type'] == 1:
            #         outputs = self.ljpm(
            #             mode=mode
            #             , fact=one_data['text']
            #             , data=one_data
            #             , acc_result=acc_result)

            #         loss += outputs['loss']
            #         acc_result = outputs['acc_result']

            #         loss += self.articles_mada(
            #             fact_embedding=outputs['fact_embedding']
            #             , class_predictions=outputs['feature']['article']
            #             , domain_label=one_data['article'])
            #         loss += self.accusations_mada(
            #             fact_embedding=outputs['fact_embedding']
            #             , class_predictions=outputs['feature']['accusation']
            #             , domain_label=one_data['accusation'])

            #         print(loss)
            #         input()

            #         return {'loss': loss, 'acc_result': acc_result}
            #     # one_data['type'] == 2 -> TCI Summary
            #     elif one_data['type'] == 2:
            #         one_data = self.bart(data=one_data, mode=mode)
            #         loss += self.dann(data=one_data)
            #         outputs = self.ljpm(
            #             mode=mode
            #             , fact=one_data['summary']
            #             , data=one_data
            #             , acc_result=acc_result)

            #         loss += outputs['loss']
            #         acc_result = outputs['acc_result']

            #         loss += self.articles_mada(
            #             fact_embedding=outputs['fact_embedding']
            #             , class_predictions=outputs['feature']['article']
            #             , domain_label=one_data['article'])
            #         loss += self.accusations_mada(
            #             fact_embedding=outputs['fact_embedding']
            #             , class_predictions=outputs['feature']['accusation']
            #             , domain_label=one_data['accusation'])

            #         print(loss)
            #         input()

            #         return {'loss': loss, 'acc_result': acc_result}
            #     # one_data['type'] == 3 -> CNewSum
            #     elif one_data['type'] == 3:
            #         outputs = self.bart(data=one_data, mode=mode)
                    
            #         loss += outputs['loss']
                    
            #         loss += self.dann(data=one_data)

            #         print(loss)
            #         input()

            #         return loss


            # result = self.atsm(data=data, mode=mode)

            # data['type'] == 1 -> TCI Text
            # data['type'] == 2 -> TCI Summary
            # data['type'] == 3 -> CNewSum
            # if data['type'] != 1:
            #     loss += self.dann(data=data)
            
            # if data['type'] == 3:
            #     loss += result
            #     return loss

            # outputs = self.ljpm(data=result, mode=mode, acc_result=acc_result)

            # loss += outputs['loss']
            # acc_result = outputs['acc_result']

            # loss += self.articles_mada(
            #     fact_embedding=outputs['middle']
            #     , class_predictions=outputs['final']['article']
            #     , domain_label=data['article'])
            # loss += self.accusations_mada(
            #     fact_embedding=outputs['middle']
            #     , class_predictions=outputs['final']['accusation']
            #     , domain_label=data['accusation'])

            # return {'loss': loss, 'acc_result': acc_result}