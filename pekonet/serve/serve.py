import logging
import torch

from pekonet.serve.utils import process_ljpbert_output_text

logger = logging.getLogger(__name__)


def serve(parameters, *args, **kwargs):
    logger.info('Start to serve.')

    model = parameters['model']
    model_name = parameters['model_name']

    if model_name == 'PekoNet':
        articles_table = parameters['articles_table']
        accusations_table = parameters['accusations_table']

        while True:
            fact = input('Enter a fact: ')
            logger.info(f'The input fact: {fact}')

            if fact == 'shutdown':
                logger.info('Stop to serve.')
                break

            fact = parameters['formatter'](data=fact)
            result = model(data=fact, mode='serve', acc_result=None)

            # The size of accusation_result is [number_of_class].
            article_result = torch.max(input=result['article'], dim=2)[1]
            accusation_result = torch.max(input=result['accusation'], dim=2)[1]

            output_text = ''
            output_text = process_ljpbert_output_text(output_text=output_text,
                                                      table=articles_table,
                                                      table_name='article',
                                                      result=article_result)
            output_text = process_ljpbert_output_text(output_text=output_text,
                                                      table=accusations_table,
                                                      table_name='accusation',
                                                      result=accusation_result)

            print(f'The prediction results: {output_text}')
            logger.info(f'The prediction results: {output_text}')
