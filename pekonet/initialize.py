import logging
import os
import torch
import random
import numpy

from torch.optim import lr_scheduler

from utils import initialize_gpus, initialize_batch_size, get_tables
from pekonet.model import initialize_model, initialize_optimizer
from pekonet.dataset import initialize_dataloader
from pekonet.output import initialize_output_function
from pekonet.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_all(
        config
        , mode
        , gpu_ids_str
        , checkpoint_path
        , batch_size_str
        , do_validation
        , *args
        , **kwargs):
    check_mode(mode=mode)
    initialize_seeds()

    gpu_ids_int_list = initialize_gpus(gpu_ids_str=gpu_ids_str)

    model = initialize_model(config=config)
    model = model.cuda()

    try:
        model.initialize_multiple_gpus(gpu_ids_int_list=gpu_ids_int_list)
        logger.info('Initialize multiple gpus successfully.')
    except:
        logger.warning('Failed to initialize multiple gpus.')

    results = {}

    if mode == 'train':
        optimizer_name = config.get('train', 'optimizer_name')
        optimizer = initialize_optimizer(config=config, model=model)
        milestones_str_list = config.get('train', 'milestones').split(',')
        milestones = [int(milestone) for milestone in milestones_str_list]
        gamma = config.getfloat('train', 'lr_multiplier')
        exp_lr_scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer
            , milestones=milestones
            , gamma=gamma)

        trained_epoch = -1
        learning_rate = config.getfloat('train', 'learning_rate')

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
                logger.error('Optimizer has been changed.')
                raise Exception('Optimizer has been changed.')

            exp_lr_scheduler.load_state_dict(parameters['exp_lr_scheduler'])

            trained_epoch = parameters['trained_epoch']
        else:
            logger.warn('The path of checkpoint is none.')

        results['model'] = model.train()
        results['optimizer_name'] = optimizer_name
        results['optimizer'] = optimizer
        results['exp_lr_scheduler'] = exp_lr_scheduler
        results['trained_epoch'] = trained_epoch
        results['learning_rate'] = learning_rate

        batch_size = initialize_batch_size(batch_size_str=batch_size_str)

        sum_train_dataloader = initialize_dataloader(
            config=config
            , task='sum_train'
            , mode=mode
            , batch_size=batch_size
        )
        results['sum_train_dataloader'] = sum_train_dataloader

        results['use_mix_data'] = config.getboolean('train', 'use_mix_data')
        if results['use_mix_data']:
            mix_train_dataloader = initialize_dataloader(
                config=config
                , task='mix_train'
                , mode=mode
                , batch_size=batch_size
            )
            results['mix_train_dataloader'] = mix_train_dataloader
            results['mix_epoch'] = config.getint('train', 'mix_epoch')
        else:
            ljp_train_dataloader = initialize_dataloader(
                config=config
                , task='ljp_train'
                , mode=mode
                , batch_size=batch_size
            )
            results['ljp_train_dataloader'] = ljp_train_dataloader
            results['ljp_epoch'] = config.getint('train', 'ljp_epoch')

        if do_validation:
            validate_dataloader = initialize_dataloader(
                config=config
                , task='validate'
                , mode='validate'
                , batch_size=batch_size)
            results['validate_dataloader'] = validate_dataloader

        output_function = initialize_output_function(config=config)
        results['output_function'] = output_function

        output_path = config.get('output', 'output_path')
        results['output_path'] = output_path

        if not os.path.exists(path=output_path):
            logger.warn(
                f'The path of output {output_path} does not exist.')
            logger.info('Make the directory automatically.')

            os.makedirs(name=output_path)

        results['sum_epoch'] = config.getint('train', 'sum_epoch')
        results['output_time'] = config.getint('output', 'output_time')
        results['test_time'] = config.getint('output', 'test_time')
    elif mode == 'serve':
        if checkpoint_path == None:
            logger.error('The path of checkpoint is none.')
            raise Exception('The path of checkpoint is none.')

        parameters = torch.load(f=checkpoint_path)
        model.load_state_dict(parameters['model'])

        formatter = initialize_formatter(config=config, task='serve')

        model_name = config.get('model', 'model_name')

        if model_name == 'PekoNet':
            articles_table, accusations_table = \
                get_tables(config=config, formatter=formatter)

        results['model'] = model.eval()
        results['formatter'] = formatter
        results['model_name'] = model_name
        results['articles_table'] = articles_table
        results['accusations_table'] = accusations_table
    # If mode is 'validate' or 'test'.
    else:
        trained_epoch = -1

        if checkpoint_path != None:
            if not os.path.exists(path=checkpoint_path):
                logger.error(
                    'The path of checkpoint is not none but it does not exixt.')
                raise Exception(
                    'The path of checkpoint is not none but it does not exixt.')

            parameters = torch.load(f=checkpoint_path)
            model.load_state_dict(parameters['model'])

            trained_epoch = parameters['trained_epoch']
        else:
            logger.warn('The path of checkpoint is none.')

        results['model'] = model.eval()
        results['trained_epoch'] = trained_epoch

        batch_size = initialize_batch_size(batch_size_str=batch_size_str)

        if mode == 'test':
            logger.info(f'The test file is {config.get("data", "test_file_path")}.')

        dataloader = initialize_dataloader(
            config=config
            , task=mode
            , mode=mode
            , batch_size=batch_size)

        results[f'{mode}_dataloader'] = dataloader

        output_function = initialize_output_function(config=config)
        results['output_function'] = output_function

        results['output_time'] = config.getint('output', 'output_time')

    logger.info('Initialize successfully.')

    return results


def check_mode(mode):
    modes = ['train', 'validate', 'test', 'serve']

    if mode not in modes:
        logger.error(f'There is no mode called {mode}.')
        raise Exception(f'There is no mode called {mode}.')

    logger.info(f'The mode has been confirmed.')


def initialize_seeds(*args, **kwargs):
    seed = 48763

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f'Seed has been set into {seed}.')