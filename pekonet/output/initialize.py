import logging

from pekonet.output.functions import empty_output_function, aa_output_function


logger = logging.getLogger(__name__)


def initialize_output_function(config, *args, **kwargs):
    output_function_name = config.get('output', 'output_function')

    output_functions = {
        'empty': empty_output_function
        , 'aa': aa_output_function
    }

    if output_function_name in output_functions:
        output_function = output_functions[output_function_name]

        logger.info('Initialize output function successfully.')

        return output_function
    else:
        logger.error(f'There is no function called {output_function_name}.')
        raise Exception(f'There is no function called {output_function_name}.')
