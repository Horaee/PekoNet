import torch
import numpy as np

from transformers import BertTokenizer

from pekonet.formatter.utils import set_special_tokens


class PekoNetFormatter:
    def __init__(self, config, mode, *args, **kwargs):
        model_path = config.get('model', 'model_path')
        # add_tokens_at_beginning = \
        #     config.getboolean('data', 'add_tokens_at_beginning')
        max_len = config.getint('data', 'max_len')
        articles_path = config.get('data', 'articles_path')
        article_sources_path = config.get('data', 'article_sources_path')
        accusations_path = config.get('data', 'accusations_path')

        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path)
        # self.add_tokens_at_beginning = add_tokens_at_beginning
        self.max_len = max_len

        self.article2id = {}

        articles = open(file=articles_path, mode='r', encoding='UTF-8')

        for article in articles:
            article = article.replace('\r', '').replace('\n', '')
            self.article2id[article] = len(self.article2id)

        articles.close()

        self.article_source2id = {}

        article_sources = open(
            file=article_sources_path
            , mode='r'
            , encoding='UTF-8')

        for article_source in article_sources:
            article_source = article_source.replace('\r', '').replace('\n', '')
            self.article_source2id[article_source] = len(self.article_source2id)

        article_sources.close()

        self.accusation2id = {}

        accusations = open(file=accusations_path, mode='r', encoding='UTF-8')

        for accusation in accusations:
            accusation = accusation.replace('\r', '').replace('\n', '')
            self.accusation2id[accusation] = len(self.accusation2id)

        accusations.close()


    def process(self, data, *args, **kwargs):
        if isinstance(data, list):
            texts = []
            summaries = []
            articles = []
            accusations = []
            types = []

            for one_data in data:
                text = self.string2ids(one_data['text'])
                texts.append(text)

                summary = self.string2ids(one_data['summary'])
                summaries.append(summary)

                article = (
                    one_data['relevant_articles'][0][0]
                    + one_data['relevant_articles'][0][1]
                )

                article_vector = np.zeros(
                    shape=len(self.article2id)
                    , dtype=np.int)

                if self.article2id.get(article):
                    article_vector[self.article2id[article]] = 1
                else:
                    article_vector[self.article2id['others']] = 1

                articles.append(article_vector.tolist())

                accusation = one_data['accusation']

                accusation_vector = np.zeros(
                    shape=len(self.accusation2id)
                    , dtype=np.int)

                if self.accusation2id.get(accusation):
                    accusation_vector[self.accusation2id[accusation]] = 1
                else:
                    accusation_vector[self.accusation2id['others']] = 1

                accusations.append(accusation_vector.tolist())

                types.append(one_data['type'])
            
            texts = torch.LongTensor(texts).cuda()
            summaries = torch.LongTensor(summaries).cuda()
            articles = torch.LongTensor(articles).cuda()
            accusations = torch.LongTensor(accusations).cuda()

            return {
                'text': texts
                , 'summary': summaries
                , 'article': articles
                , 'accusation': accusations
                , 'type': types
            }
        elif isinstance(data, str):
            char_list = self.tokenizer.tokenize(data)
            char_list = set_special_tokens(
                add_tokens_at_beginning=True
                , max_len=self.max_len
                , data=char_list)

            ids = self.tokenizer.convert_tokens_to_ids(char_list)

            result = torch.LongTensor(ids)

            # The size of data after unsqueeze
            # = [batch_size, seq_len]
            # = [1, 512].
            result = torch.unsqueeze(result, 0)

            return result.cuda()


    def string2ids(self, string, type):
        char_list = self.tokenizer.tokenize(string)
        char_list = set_special_tokens(
            # add_tokens_at_beginning=self.add_tokens_at_beginning
            add_tokens_at_beginning=True if type == 0 else False
            , max_len=self.max_len
            , data=char_list)
        ids = self.tokenizer.convert_tokens_to_ids(char_list)

        return ids