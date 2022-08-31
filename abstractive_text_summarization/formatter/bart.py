import torch

from transformers import BertTokenizer

from abstractive_text_summarization.formatter.utils import set_special_tokens


class BartFormatter:
    def __init__(self, config, *args, **kwargs):
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.get('model', 'bart_path'))
        self.add_tokens_at_beginning = \
            config.getboolean('data', 'add_tokens_at_beginning')
        self.max_len = config.getint('data', 'max_len')


    def process(self, data, *args, **kwargs):
        if isinstance(data, list):
            texts = []
            summaries = []

            for one_data in data:
                text = self.tokenizer.tokenize(one_data['text'])
                text = set_special_tokens(
                    add_tokens_at_beginning=self.add_tokens_at_beginning
                    , max_len=self.max_len
                    , data=text)

                texts.append(self.tokenizer.convert_tokens_to_ids(text))

                summary = self.tokenizer.tokenize(one_data['summary'])
                summary = set_special_tokens(
                    add_tokens_at_beginning=self.add_tokens_at_beginning
                    , max_len=self.max_len
                    , data=summary)

                summaries.append(self.tokenizer.convert_tokens_to_ids(summary))
            
            texts = torch.LongTensor(texts)
            summaries = torch.LongTensor(summaries)

            return {'text': texts, 'summary': summaries}
        elif isinstance(data, str):
            text = self.tokenizer.tokenize(data)
            text = set_special_tokens(
                add_tokens_at_beginning=self.add_tokens_at_beginning
                , max_len=self.max_len
                , data=text)

            text = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(text))

            # The size of data after unsqueeze
            # = [batch_size, seq_len]
            # = [1, 512].
            text = torch.unsqueeze(text, 0)

            return text.cuda()