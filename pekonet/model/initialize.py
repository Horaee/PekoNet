import logging

# from pytorch_pretrained_bert import BertAdam
from torch.optim import Adam

# from legal_judgment_prediction.model.bert import LJPBert
from pekonet.model.pekonet import PekoNet


logger = logging.getLogger(__name__)


def initialize_model(config, *args, **kwargs):
    logger.info('Start to initialize model.')

    model_name = config.get('model', 'model_name')

    models = {
        'PekoNet': PekoNet
    }

    if model_name in models.keys():
        model = models[model_name](config=config)

        logger.info('Initialize model successfully.')

        return model
    else:
        logger.error(f'There is no model called {model_name}.')
        raise Exception(f'There is no model called {model_name}.')


def initialize_optimizer(config, model, *args, **kwargs):
    logger.info('Start to initialize optimizer.')

    optimizer_name = config.get('train', 'optimizer')
    learning_rate = config.getfloat('train', 'learning_rate')

    optimizers = {
        'adam': Adam
        # 'bert_adam': BertAdam
    }

    if optimizer_name in optimizers:
        optimizer = optimizers[optimizer_name](
            params=model.parameters()
            , lr=learning_rate)

        logger.info('Initialize optimizer successfully.')

        return optimizer
    else:
        logger.error(f'There is no optimizer called {optimizer_name}.')
        raise Exception(f'There is no optimizer called {optimizer_name}.')