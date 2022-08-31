import logging
import pickle
import torch
import os

from utils import initialize_gpus
from abstractive_text_summarization.model import initialize_model
from abstractive_text_summarization.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_all(config, device_str, checkpoint_path):
    logger.info('Start to initialize.')

    check_table = {
        'tasks': ['legal_judgment_prediction', 'abstractive_text_summarization']
        , 'summarizations': ['none', 'bart', 'lead_3']
        , 'model_names': ['LJSBart']
        , 'types': ['one_label', 'CNewSum_v2']
    }

    results = {
        'task': config.get('common', 'task')
        , 'type': config.get('common', 'type')
        , 'data_path': config.get('common', 'data_path')
        , 'output_path': config.get('common', 'output_path')
        , 'train_size': config.getfloat('common', 'train_size')
        , 'valid_size': config.getfloat('common', 'valid_size')
        , 'generate_test_data': \
            config.getboolean('common', 'generate_test_data')
        , 'random_seed': config.getint('common', 'random_seed')
    }

    if results['task'] not in check_table['tasks']:
        logger.error(f'There is no task called {results["task"]}.')
        raise Exception(f'There is no task called {results["task"]}.')
    elif results['task'] == 'legal_judgment_prediction':
        results['summarization'] = config.get('common', 'summarization')

        if results['summarization'] not in check_table['summarizations']:
            logger.error(f'There is no summarization method called \
{results["summarization"]}.')
            raise Exception(f'There is no summarization method called \
{results["summarization"]}.')
        elif results['summarization'] == 'none':
            results['article_lowerbound'] = \
                config.getint('common', 'article_lowerbound')
            parameters_file_path = config.get('common', 'parameters')

            with open(
                    file=parameters_file_path
                    , mode='rb') as pkl_file:
                parameters = pickle.load(file=pkl_file)
                pkl_file.close()

            results['articles_times_appeared_of_all_files'] = \
                parameters['articles_times_appeared_of_all_files']
        elif results['summarization'] == 'bart':
            gpus = initialize_gpus(device_str=device_str)

            model_name = config.get('model', 'model_name')

            if model_name not in check_table['model_names']:
                logger.error(
                    f'There is no model name called {model_name}.')
                raise Exception(
                    f'There is no model name called {model_name}.')

            results['model'] = initialize_model(config=config)
            results['model'].cuda()

            try:
                logger.info('Start to initialize multiple gpus.')

                results['model'].initialize_multiple_gpus(gpus=gpus)

                logger.info('Initialize multiple gpus successfully.')
            except Exception:
                logger.warning('Failed to initialize multiple gpus.')

            if checkpoint_path is None:
                logger.error('The checkpoint path is none.')
                raise Exception('The checkpoint path is none.')

            parameters = torch.load(checkpoint_path)
            results['model'].load_state_dict(parameters['model'])
            results['model'].eval()

            results['formatter'] = initialize_formatter(config=config)

            results['output_time'] = config.getint('output', 'output_time')
        elif results['summarization'] == 'lead_3':
            results['output_time'] = config.getint('output', 'output_time')

    if results['type'] not in check_table['types']:
        logger.error(f'There is no type called {results["type"]}.')
        raise Exception(f'There is no type called {results["type"]}.')

    if not os.path.exists(path=results['data_path']):
        logger.error(
            f'The path of data_path {results["data_path"]} does not exist.')
        raise Exception(
            f'The path of data_path {results["data_path"]} does not exist.')

    if not os.path.exists(path=results['output_path']):
        logger.warn(
            f'The path of output {results["output_path"]} does not exist.')
        logger.info('Make the directory automatically.')

        os.makedirs(name=results['output_path'])

    logger.info('Initialize successfully.')

    return results