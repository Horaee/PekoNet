import torch.nn as nn

from pekonet.model.bart import BART
from pekonet.model.ljpm import LJPM


# TODO: Part-checked.
class PekoNet(nn.Module):
    # Checked.
    def __init__(self, config, *args, **kwargs):
        super(PekoNet, self).__init__()

        self.bart = BART(config=config)
        self.ljpm = LJPM(config=config)


    # Checked.
    def initialize_multiple_gpus(self, gpu_ids_int_list, *args, **kwargs):
        self.bart = nn.DataParallel(
            module=self.bart
            , device_ids=gpu_ids_int_list)
        self.ljpm = nn.DataParallel(
            module=self.ljpm
            , device_ids=gpu_ids_int_list)


    def forward(self, data, mode, cm_results=None, *args, **kwargs):
        # Checked.
        if mode == 'train' or mode == 'validate':
            outputs = self.bart(data=data, mode=mode)

            sum_loss, cls_loss = outputs['loss'], None
            cns_tci_data_number = outputs['cns_tci_data_number']

            if outputs['summary_ids'] != None:
                # Size of `outputs['summary_ids']` = \
                #     [batch_size, sequence_length].
                # Size of `cls_embeddings` = [batch_size, hidden_size].
                cls_embeddings = self.bart.module.get_clses_embedding(
                    ids=outputs['summary_ids'])

                labels = {
                    'article': outputs['tci_data']['article']
                    , 'accusation': outputs['tci_data']['accusation']
                }

                outputs = self.ljpm(
                    mode=mode
                    , cls_embeddings=cls_embeddings
                    , labels=labels
                    , cm_results=cm_results)

                cls_loss = outputs['loss']
                cm_results = outputs['cm_results']

            return {
                'cns_tci_data_number': cns_tci_data_number
                , 'sum_loss': sum_loss
                , 'cls_loss': cls_loss
                , 'cm_results': cm_results
            }
        # If mode is 'test' or 'serve'.
        # TODO: Unfinished.
        else:
            cls_embeddings = self.bart.module.get_clses_embedding(
                ids=data['text'])

            labels = {
                'article': data['article']
                , 'accusation': data['accusation']
            }

            outputs = self.ljpm(
                mode=mode
                , cls_embeddings=cls_embeddings
                , labels=labels
                , cm_result=cm_results)

            # cls_embeddings.size(0) is the batch_size.
            cls_data_number = cls_embeddings.size(0)
            cls_loss = outputs['loss']
            cm_results = outputs['cm_results']

            return {
                'cls_data_number': cls_data_number
                , 'cls_loss': cls_loss
                , 'cm_result': cm_results
            }

            # tensor = self.bart(data, mode)
            # output = self.ljpm(tensor, mode)

            # return output
            return 48763;

        # mode == 'evaluate'.
        # Size of `outputs['summary_ids']` = \
        #     [batch_size, sequence_length].
        # Size of `cls_embeddings` = [batch_size, hidden_size].
        # ---
        # cls_embeddings = self.bart.module.get_clses_embedding(
        #     ids=data['text'])

        # labels = {
        #     'article': data['article']
        #     , 'accusation': data['accusation']
        # }

        # outputs = self.ljpm(
        #     mode=mode
        #     , cls_embeddings=cls_embeddings
        #     , labels=labels
        #     , cm_result=cm_results)

        # cls_loss = outputs['loss']
        # cm_results = outputs['cm_results']

        # return {
        #     'cls_loss': cls_loss
        #     , 'cm_result': cm_results
        # }
        # ---