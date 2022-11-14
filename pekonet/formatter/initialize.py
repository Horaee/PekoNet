import logging

from pekonet.formatter.pekonet import PekoNetFormatter


logger = logging.getLogger(__name__)


def initialize_formatter(config, mode, task=None, *args, **kwargs):
    logger.info('Start to initialize formatter.')

    try:
        formatter_name = config.get('data', f'{task}_formatter_type')
    except:
        logger.error('Failed to get the type of formatter.')
        raise Exception('Failed to get the type of formatter.')

    formatters = {'PekoNetFormatter': PekoNetFormatter}

    if formatter_name in formatters:
        formatter = formatters[formatter_name](
            config=config
            , mode=mode)
    else:
        logger.error(f'There is no formatter called {formatter_name}.')
        raise Exception(f'There is no formatter called {formatter_name}.')

    def collate_fn(data):
        return formatter.process(data)

    logger.info('Initialize formatter successfully.')

    return collate_fn