import torch.nn as nn

from transformers import BertModel


class BERT(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(BERT, self).__init__()

        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=config.get(
                'model'
                , 'bert_model_path'))


    def forward(self, ids):
        output = self.bert(input_ids=ids)
        return output.pooler_output