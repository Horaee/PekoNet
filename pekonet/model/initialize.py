import logging

from torch.optim import Adam

from pekonet.model.pekonet import PekoNet


logger = logging.getLogger(__name__)


def initialize_model(config, *args, **kwargs):
    model_name = config.get('model', 'model_name')

    models = {
        'PekoNet': PekoNet
    }

    if model_name not in models.keys():
        logger.error(f'There is no model called {model_name}.')
        raise Exception(f'There is no model called {model_name}.')

    model = models[model_name](config=config)

    logger.info('Initialize model successfully.')

    return model


def initialize_optimizer(config, model, *args, **kwargs):
    optimizer_name = config.get('train', 'optimizer_name')
    learning_rate = config.getfloat('train', 'learning_rate')

    optimizers = {
        'adam': Adam
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