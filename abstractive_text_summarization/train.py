import logging
import os
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import get_time_str, log_results
from abstractive_text_summarization.eval import eval_one


logger = logging.getLogger(__name__)


def train(parameters, *args, **kwargs):
    model = parameters['model']
    optimizer_name = parameters['optimizer_name']
    optimizer = parameters['optimizer']
    exp_lr_scheduler = parameters['exp_lr_scheduler']
    trained_epoch = (parameters['trained_epoch'] + 1)
    train_dataloader = parameters['train_dataloader']
    valid_dataloader = parameters['valid_dataloader']
    output_function = parameters['output_function']
    output_path = parameters['output_path']
    total_epoch = parameters['total_epoch']
    output_time = parameters['output_time']
    test_time = parameters['test_time']

    logger.info('Start to train model.')

    total_len = len(train_dataloader)

    for current_epoch in range(trained_epoch, total_epoch):
        start_time = timer()
        current_epoch = current_epoch
        total_loss = 0
        message = ""
        step = -1
        learning_rate = -1

        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        for step, data in enumerate(train_dataloader):
            for key in data.keys():
                data[key] = Variable(data[key].cuda())

            optimizer.zero_grad()

            results = model(data=data, mode='train')

            loss = results['loss']
            total_loss += float(loss)

            loss.backward()
            optimizer.step()

            if step % output_time == 0:
                message = output_function()
                delta_time = (timer() - start_time)

                log_results(
                    epoch=current_epoch
                    , stage='train'
                    , iterations=f'{(step+1)}/{total_len}'
                    , time=f'{get_time_str(delta_time)}/\
{get_time_str(delta_time*(total_len-step-1)/(step+1))}'
                    , loss=f'{(total_loss/(step+1))}'
                    , learning_rate=learning_rate
                    , results=message
                )

        if step == -1:
            logger.error('There is no data in this dataset.')
            raise Exception('There is no data in this dataset.')

        exp_lr_scheduler.step()

        message = output_function()
        delta_time = (timer() - start_time)

        log_results(
            epoch=current_epoch
            , stage='train'
            , iterations=f'{(step+1)}/{total_len}'
            , time=f'{get_time_str(delta_time)}/\
{get_time_str(delta_time*(total_len-step-1)/(step+1))}'
            , loss=f'{(total_loss/(step+1))}'
            , learning_rate=learning_rate
            , results=message
        )

        save_checkpoint(
            model=model
            , optimizer_name=optimizer_name
            , optimizer=optimizer
            , exp_lr_scheduler=exp_lr_scheduler
            , trained_epoch=current_epoch
            , file=os.path.join(output_path, f'checkpoint_{current_epoch}.pkl')
        )

        if current_epoch % test_time == 0:
            with torch.no_grad():
                eval_one(
                    model=model
                    , dataset=valid_dataloader
                    , output_time=output_time
                    , output_function=output_function
                    , current_epoch=current_epoch
                    , task='valid'
                    , from_train=True
                )

        gc.collect()
        torch.cuda.empty_cache()

    logger.info('Train model successfully.')


def save_checkpoint(
        model
        , optimizer_name
        , optimizer
        , exp_lr_scheduler
        , trained_epoch
        , file):
    if hasattr(model, 'module'):
        model = model.module

    save_params = {
        'model': model.state_dict()
        , 'optimizer_name': optimizer_name
        , 'optimizer': optimizer.state_dict()
        , 'exp_lr_scheduler': exp_lr_scheduler.state_dict()
        , 'trained_epoch': trained_epoch
    }

    try:
        torch.save(obj=save_params, f=file)
    except Exception:
        logger.error(f'Failed to save model with error {Exception}.')
        raise Exception