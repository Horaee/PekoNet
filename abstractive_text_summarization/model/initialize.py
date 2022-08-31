import logging
import torch.optim as optim

from abstractive_text_summarization.model.bart import LJSBart


logger = logging.getLogger(__name__)


def initialize_model(config, *args, **kwargs):
    logger.info('Start to initialize model.')

    model_name = config.get('model', 'model_name')
    model_types = {'LJSBart': LJSBart}

    if model_name in model_types.keys():
        model = model_types[model_name](config=config)

        logger.info('Initialize model successfully.')

        return model
    else:
        logger.error(f'There is no model called {model_name}.')
        raise Exception(f'There is no model called {model_name}.')


def initialize_optimizer(config, model, *args, **kwargs):
    logger.info('Start to initialize optimizer.')

    optimizer_name = config.get('train', 'optimizer')
    learning_rate = config.getfloat('train', 'learning_rate')

    optimizer_types = {'adam': optim.Adam}

    if optimizer_name in optimizer_types:
        optimizer = optimizer_types[optimizer_name](
            params=model.parameters()
            , lr=learning_rate)

        logger.info('Initialize optimizer successfully.')

        return optimizer
    else:
        logger.error(f'There is no optimizer called {optimizer_name}.')
        raise Exception(f'There is no optimizer called {optimizer_name}.')