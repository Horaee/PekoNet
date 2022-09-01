import torch.nn as nn

from transformers import BertTokenizer

from abstractive_text_summarization.model.bart_model import BartModel


class LJSBart(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(LJSBart, self).__init__()

        self.bart = BartModel(config=config)
        self.tokenizer = BertTokenizer.from_pretrained(
                pretrained_model_name_or_path=config.get('model', 'model_path'))


    def initialize_multiple_gpus(self, gpus, *args, **kwargs):
        self.bart = nn.DataParallel(module=self.bart, device_ids=gpus)


    def forward(self, data, mode, *args, **kwargs):
        if mode == 'serve':
            output = self.bart.module.generate(input=data)
            summary = self.tokenizer.batch_decode(
                sequences=output
                , skip_special_tokens=True
                , clean_up_tokenization_spaces=False)[0]

            return summary
        else:
            outputs = self.bart(input=data['text'], label=data['summary'])
            loss, tensor = outputs
            summary = self.tokenizer.batch_decode(
                sequences=tensor
                , skip_special_tokens=True
                , clean_up_tokenization_spaces=False)[0]

            return {'loss': loss, 'summary': summary}