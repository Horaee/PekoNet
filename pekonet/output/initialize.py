import logging

from pekonet.output.functions import \
    empty_output_function, aa_output_function


logger = logging.getLogger(__name__)


def initialize_output_function(config, *args, **kwargs):
    logger.info('Start to initialize output function.')

    output_function_types = {
        'empty': empty_output_function
        , 'aa': aa_output_function
    }

    function_name = config.get('output', 'output_function')

    if function_name in output_function_types:
        output_function = output_function_types[function_name]

        logger.info('Initialize output function successfully.')

        return output_function
    else:
        logger.error(f'There is no function called {function_name}.')
        raise Exception(f'There is no function called {function_name}.')
