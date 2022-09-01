import logging
import os
import torch

from torch.optim import lr_scheduler

from utils import initialize_gpus, initialize_batch_size
from abstractive_text_summarization.model import \
    initialize_model, initialize_optimizer
from abstractive_text_summarization.dataset import initialize_dataloader
from abstractive_text_summarization.output import initialize_output_function
from abstractive_text_summarization.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_all(
        config
        , mode
        , device_str
        , batch_size_str
        , checkpoint_path
        , *args
        , **kwargs):
    logger.info('Start to initialize.')

    check_mode(mode=mode)

    gpus = initialize_gpus(device_str=device_str)

    model = initialize_model(config=config)
    model = model.cuda()

    try:
        logger.info('Start to initialize multiple gpus.')

        model.initialize_multiple_gpus(gpus=gpus)

        logger.info('Initialize multiple gpus successfully.')
    except:
        logger.warning('Failed to initialize multiple gpus.')

    results = {}

    if mode == 'train':
        batch_size = initialize_batch_size(batch_size_str=batch_size_str)
        optimizer = initialize_optimizer(config=config, model=model)

        milestones_str_list = config.get(mode, 'milestones').split(',')
        milestones = [int(milestone) for milestone in milestones_str_list]
        exp_lr_scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer
            , milestones=milestones
            , gamma=config.getfloat('train', 'lr_multiplier'))

        trained_epoch = -1

        if checkpoint_path != None:
            if not os.path.exists(path=checkpoint_path):
                logger.error(
                    'The path of checkpoint is not none but it does not exixt.')
                raise Exception(
                    'The path of checkpoint is not none but it does not exixt.')

            parameters = torch.load(checkpoint_path)
            model.load_state_dict(parameters['model'])

            if config.get('train', 'optimizer') == parameters['optimizer_name']:
                optimizer.load_state_dict(parameters['optimizer'])
            else:
                logger.warning('Optimizer has been changed.')
                logger.info('Use the new optimizer to train the model.')


            exp_lr_scheduler.load_state_dict(parameters['exp_lr_scheduler'])

            trained_epoch = parameters['trained_epoch']
        else:
            logger.warn('The path of checkpoint is none.')

        train_dataloader = initialize_dataloader(
            config=config
            , task='train'
            , mode='train'
            , batch_size=batch_size)
        valid_dataloader = initialize_dataloader(
            config=config
            , task='valid'
            , mode='eval'
            , batch_size=batch_size)

        results['model'] = model.train()
        results['optimizer_name'] = config.get(mode, 'optimizer')
        results['optimizer'] = optimizer
        results['exp_lr_scheduler'] = exp_lr_scheduler
        results['trained_epoch'] = trained_epoch
        results['train_dataloader'] = train_dataloader
        results['valid_dataloader'] = valid_dataloader
        results['output_function'] = initialize_output_function(config)
        results['total_epoch'] = config.getint(mode, 'total_epoch')
        results['output_path'] = config.get('output', 'output_path')

        if not os.path.exists(results['output_path']):
            logger.warn(
                f'The path of output {results["output_path"]} does not exist.')
            logger.info('Make the directory automatically.')

            os.makedirs(results['output_path'])

        results['output_time'] = config.getint('output', 'output_time')
        results['test_time'] = config.getint('output', 'test_time')
    elif mode == 'serve':
        if checkpoint_path == None:
            logger.error(
                'The path of checkpoint is none but the mode is serve.')
            raise Exception(
                'The path of checkpoint is none but the mode is serve.')
    
        parameters = torch.load(checkpoint_path)
        model.load_state_dict(parameters['model'])

        results['model'] = model.eval()
        results['formatter'] = initialize_formatter(config=config)

    logger.info('Initialize successfully.')

    return results


def check_mode(mode):
    modes = ['train', 'serve']

    if mode not in modes:
        logger.error(f'There is no mode called {mode}.')
        raise Exception(f'There is no mode called {mode}.')