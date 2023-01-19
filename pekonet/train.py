import logging
import torch
import os
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import log_results, get_time_str
from legal_judgment_prediction.eval import eval_one


logger = logging.getLogger(__name__)


def train(parameters, do_test, *args, **kwargs):
    model = parameters['model']
    optimizer = parameters['optimizer']
    exp_lr_scheduler = parameters['exp_lr_scheduler']
    optimizer_name = parameters['optimizer_name']
    trained_epoch = (parameters['trained_epoch'] + 1)
    train_dataloader = parameters['train_dataloader']
    valid_dataloader = parameters['valid_dataloader']
    output_function = parameters['output_function']
    output_path = parameters['output_path']
    total_epoch = parameters['total_epoch']
    output_time = parameters['output_time']
    test_time = parameters['test_time']

    if do_test == True:
        test_dataloader = parameters['test_dataloader']

    logger.info('Start to train model.')

    total_len = len(train_dataloader)

    for current_epoch in range(trained_epoch, total_epoch):
        start_time = timer()
        sum_loss, cls_loss = None, None
        sum_counter, cls_counter = 0, 0
        cm_result = None
        mima_prf_results = ''
        step = -1
        learning_rate = -1

        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        for step, data in enumerate(iterable=train_dataloader):
            for key in data.keys():
                if isinstance(data[key], torch.LongTensor):
                    data[key] = Variable(data[key].cuda())

            optimizer.zero_grad()

            results = model(data=data, mode='train', cm_result=cm_result)

            if results['sum_loss'] != None:
                if sum_loss == None:
                    sum_loss = float(results['sum_loss'])
                else:
                    sum_loss += float(results['sum_loss'])

                sum_counter += results['cns_tci_data_number'][0]
                results['sum_loss'].backward()

            if results['cls_loss'] != None:
                if cls_loss == None:
                    cls_loss = float(results['cls_loss'])
                else:
                    cls_loss += float(results['cls_loss'])

                cls_counter += results['cns_tci_data_number'][1]
                results['cls_loss'].backward()

            cm_result = results['cm_result']

            optimizer.step()

            if step % output_time == 0:
                if cm_result != None:
                    mima_prf_results = output_function(data=cm_result)
                else:
                    mima_prf_results = None

                delta_time = (timer() - start_time)
                sum_loss_log = float(sum_loss / sum_counter) \
                    if sum_loss != None else None
                cls_loss_log = float(cls_loss / cls_counter) \
                    if cls_loss != None else None

                log_results(
                    epoch=current_epoch
                    , stage='train'
                    , iterations=f'{(step+1)}/{total_len}'
                    , time=f'{get_time_str(delta_time)}/\
{get_time_str(delta_time*(total_len-step-1)/(step+1))}'
                    , sum_loss=str(sum_loss_log)
                    , cls_loss=str(cls_loss_log)
                    , learning_rate=learning_rate
                    , results=mima_prf_results
                )

        if step == -1:
            logger.error('There is no data in this dataset.')
            raise Exception('There is no data in this dataset.')

        exp_lr_scheduler.step()

        if cm_result != None:
            mima_prf_results = output_function(data=cm_result)
        else:
            mima_prf_results = None

        delta_time = (timer() - start_time)
        sum_loss_log = float(sum_loss / sum_counter) \
            if sum_loss != None else None
        cls_loss_log = float(cls_loss / cls_counter) \
            if cls_loss != None else None

        log_results(
            epoch=current_epoch
            , stage='train'
            , iterations=f'{(step+1)}/{total_len}'
            , time=f'{get_time_str(delta_time)}/\
{get_time_str(delta_time*(total_len-step-1)/(step+1))}'
            , sum_loss=str(sum_loss_log)
            , cls_loss=str(cls_loss_log)
            , learning_rate=learning_rate
            , results=mima_prf_results
        )

        save_checkpoint(
            model=model
            , optimizer_name=optimizer_name
            , optimizer=optimizer
            , trained_epoch=current_epoch
            , exp_lr_scheduler=exp_lr_scheduler
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

                if do_test:
                    eval_one(
                        model=model
                        , dataset=test_dataloader
                        , output_time=output_time
                        , output_function=output_function
                        , current_epoch=current_epoch
                        , task='test'
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
        , 'optimizer': optimizer.state_dict()
        , 'exp_lr_scheduler': exp_lr_scheduler.state_dict()
        , 'optimizer_name': optimizer_name
        , 'trained_epoch': trained_epoch
    }

    try:
        torch.save(obj=save_params, f=file)
    except Exception:
        logger.error(f'Failed to save model with error {Exception}.')
        raise Exception
