import random
import torch.nn as nn

from transformers import BartForConditionalGeneration, BertTokenizer


class ATSM(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(ATSM, self).__init__()

        self.bart = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.get('atsm', 'model_path'))
        self.tokenizer = BertTokenizer.from_pretrained(
                pretrained_model_name_or_path=config.get('atsm', 'model_path'))
        self.max_len = config.getint('data', 'max_len')


    def forward(self, data, mode, *args, **kwargs):
        if mode == 'serve':
            tensor = self.bart.generate(data, max_length=self.max_len)
            # summary = self.tokenizer.batch_decode(
            #     sequences=output
            #     , skip_special_tokens=True
            #     , clean_up_tokenization_spaces=False)[0]

            return tensor
        # mode == 'train' or 'eval'
        else:
            tensor = self.bart.generate(data['text'], max_length=self.max_len)

            # data['type'] == 1 -> TCI; data['type'] == 0 -> CNewSum
            if data['type'] == 1:
                data['summary'] = tensor

                # 2 -> Use text; 3 -> Use summary
                data['TorS'] == random.randint(a=2, b=3)

                return data

            loss = self.bart(
                input_ids=data['text'], labels=data['summary'])['loss']

            return loss