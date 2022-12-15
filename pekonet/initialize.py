import logging
import random
import numpy
import torch
import os

from torch.optim import lr_scheduler

from utils import initialize_gpus, initialize_batch_size, get_tables
from pekonet.model.initialize import initialize_model, initialize_optimizer
from pekonet.dataset import initialize_dataloader
from pekonet.output import initialize_output_function
from pekonet.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_all(
        config
        , mode
        , device_str
        , checkpoint_path
        , batch_size_str
        # , do_test
        , *args
        , **kwargs):
    logger.info('Start to initialize.')

    check_mode(mode=mode)
    initialize_seeds()

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
        optimizer_name = config.get('train', 'optimizer')
        optimizer = initialize_optimizer(config=config, model=model)

        milestones_str_list = config.get('train', 'milestones').split(',')
        milestones = [int(milestone) for milestone in milestones_str_list]
        gamma = config.getfloat('train', 'lr_multiplier')
        exp_lr_scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer
            , milestones=milestones
            , gamma=gamma)

        trained_epoch = -1

        if checkpoint_path != None:
            if not os.path.exists(path=checkpoint_path):
                logger.error(
                    'The path of checkpoint is not none but it does not exixt.')
                raise Exception(
                    'The path of checkpoint is not none but it does not exixt.')

            parameters = torch.load(f=checkpoint_path)
            model.load_state_dict(parameters['model'])

            if optimizer_name == parameters['optimizer_name']:
                optimizer.load_state_dict(parameters['optimizer'])
            else:
                logger.warning('Optimizer has been changed.')
                logger.info('Use the new optimizer to train the model.')

            exp_lr_scheduler.load_state_dict(parameters['exp_lr_scheduler'])

            trained_epoch = parameters['trained_epoch']
        else:
            logger.warn('The path of checkpoint is none.')

        batch_size = initialize_batch_size(batch_size_str=batch_size_str)

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

        output_function = initialize_output_function(config=config)

        output_path = config.get('output', 'output_path')

        if not os.path.exists(path=output_path):
            logger.warn(
                f'The path of output {output_path} does not exist.')
            logger.info('Make the directory automatically.')

            os.makedirs(name=output_path)

        results['model'] = model.train()
        results['optimizer'] = optimizer
        results['exp_lr_scheduler'] = exp_lr_scheduler
        results['optimizer_name'] = config.get('train', 'optimizer')
        results['trained_epoch'] = trained_epoch
        results['train_dataloader'] = train_dataloader
        results['valid_dataloader'] = valid_dataloader
        results['output_function'] = output_function
        results['output_path'] = output_path
        results['model_name'] = config.get('model', 'model_name')
        results['total_epoch'] = config.getint('train', 'total_epoch')
        results['output_time'] = config.getint('output', 'output_time')
        results['test_time'] = config.getint('output', 'test_time')
    # elif mode == 'eval':
    #     trained_epoch = -1

    #     if checkpoint_path != None:
    #         if not os.path.exists(path=checkpoint_path):
    #             logger.error(
    #                 'The path of checkpoint is not none but it does not exixt.')
    #             raise Exception(
    #                 'The path of checkpoint is not none but it does not exixt.')

    #         parameters = torch.load(f=checkpoint_path)
    #         model.load_state_dict(parameters['model'])
    #         trained_epoch = parameters['trained_epoch']
    #     else:
    #         logger.warn('The path of checkpoint is none.')

    #     batch_size = initialize_batch_size(batch_size_str=batch_size_str)

    #     test_dataloader = initialize_dataloader(
    #         config=config
    #         , task='test'
    #         , mode='eval'
    #         , batch_size=batch_size)

    #     output_function = initialize_output_function(config=config)

    #     results['model'] = model.eval()
    #     results['trained_epoch'] = trained_epoch
    #     results['test_dataloader'] = test_dataloader
    #     results['output_function'] = output_function
    #     results['model_path'] = config.get('model', 'model_path')
    #     results['output_function_name'] = \
    #         config.get('output', 'output_function')
    # elif mode == 'serve':
    #     if checkpoint_path == None:
    #         logger.error('The path of checkpoint is none.')
    #         raise Exception('The path of checkpoint is none.')

    #     parameters = torch.load(f=checkpoint_path)
    #     model.load_state_dict(parameters['model'])

    #     formatter = initialize_formatter(config=config)

    #     model_name = config.get('model', 'model_name')

    #     if model_name == 'PekoNet':
    #         articles_table, accusations_table = \
    #             get_tables(config=config, formatter=formatter)

    #     results['model'] = model.eval()
    #     results['formatter'] = formatter
    #     results['model_name'] = model_name
    #     results['articles_table'] = articles_table
    #     results['accusations_table'] = accusations_table

    logger.info('Initialize successfully.')

    return results


def check_mode(mode):
    modes = ['train', 'eval', 'serve']

    if mode not in modes:
        logger.error(f'There is no mode called {mode}.')
        raise Exception(f'There is no mode called {mode}.')


def initialize_seeds():
    seed = 48763
    random.seed(seed)
    numpy.random.seed(seed)
    # pd.core.common.random_state(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False