import torch
import torch.nn as nn

from transformers import BartForConditionalGeneration, BertTokenizer


class BART(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(BART, self).__init__()

        self.bart = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.get(
                'model'
                , 'atsm_model_path'
            ))
        # self.tokenizer = BertTokenizer.from_pretrained(
        #         pretrained_model_name_or_path=config.get(
        #             'model'
        #             , 'atsm_model_path'
        #         ))
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
            outputs = self.bart(
                input_ids=data['text']
                , labels=data['summary'], output_hidden_states=True)

            return outputs
            # data['type'] == 3 -> CNewSum
            # if data['type'] == 3:
            #     data['text'] = torch.unsqueeze(input=data['text'], dim=0)
            #     data['summary'] = torch.unsqueeze(input=data['summary'], dim=0)

            #     loss = self.bart(
            #         input_ids=data['text'], labels=data['summary'])['loss']
            #     summary_embedding = self.bart(
            #         input_ids=data['summary'])['encoder_last_hidden_state']

            #     return {'loss': loss, 'summary_embedding': summary_embedding}

            # # data['type'] == 2 -> TCI Summary
            # data['summary'] = self.bart.generate(
            #     data['text']
            #     , max_length=self.max_len)

            # return data