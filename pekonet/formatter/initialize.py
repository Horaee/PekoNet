import logging

from pekonet.formatter.pekonet import PekoNetFormatter


logger = logging.getLogger(__name__)


# Checked.
def initialize_formatter(config, task=None, *args, **kwargs):
    try:
        formatter_name = config.get('data', f'{task}_formatter_name')
    except:
        logger.error('Failed to get the name of formatter.')
        raise Exception('Failed to get the name of formatter.')

    formatters = {'PekoNetFormatter': PekoNetFormatter}

    if formatter_name in formatters:
        formatter = formatters[formatter_name](config=config)
    else:
        logger.error(f'There is no formatter called {formatter_name}.')
        raise Exception(f'There is no formatter called {formatter_name}.')

    def collate_fn(data):
        return formatter.process(data)

    logger.info('Initialize formatter successfully.')

    return collate_fn