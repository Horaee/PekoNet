import torch.nn as nn


class LJPMPredictor(nn.Module):
    # Checked.
    def __init__(
            self
            , hidden_size
            , articles_number
            , accusations_number
            , *args
            , **kwargs):
        super(LJPMPredictor, self).__init__()

        self.article_fc = nn.Linear(
            in_features=hidden_size
            , out_features=articles_number*2)
        self.accusation_fc = nn.Linear(
            in_features=hidden_size
            , out_features=accusations_number*2)


    # Checked.
    def forward(self, tensors, *args, **kwargs):
        articles = self.article_fc(input=tensors)
        accusations = self.accusation_fc(input=tensors)
        
        batch = tensors.size()[0]
        articles = articles.view(batch, -1, 2)
        accusations = accusations.view(batch, -1, 2)

        return {
            'article': articles
            , 'accusation': accusations
        }
