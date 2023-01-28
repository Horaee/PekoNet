import logging
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import log_results, get_time_str


logger = logging.getLogger(__name__)


def eval(parameters, *args, **kwargs):
    model = parameters['model']
    trained_epoch = parameters['trained_epoch']
    test_dataloader = parameters['test_dataloader']
    output_function = parameters['output_function']

    eval_one(
        model=model
        , dataset=test_dataloader
        # "output_time=1000" is for "batch_size=7".
        , output_time=1000
        , output_function=output_function
        , current_epoch=trained_epoch
        , task='test'
    )


def eval_one(
        model
        , dataset
        , output_time
        , output_function
        , current_epoch
        , task
        , from_train=False
        , *args
        , **kwargs):
    model.eval()

    total_len = len(dataset)

    start_time = timer()
    total_loss = 0
    acc_result = None
    mima_prf_results = ''
    step = -1

    for step, data in enumerate(dataset):
        for key in data.keys():
            data[key] = Variable(data[key].cuda())

        results = model(data=data, mode='eval', acc_result=acc_result)

        loss = results['loss']
        total_loss += float(loss)

        acc_result = results['acc_result']

        if step % output_time == 0:
            mima_prf_results = output_function(
                total_loss=total_loss
                , step=step
                , data=acc_result)

            delta_time = (timer() - start_time)

            log_results(
                epoch=current_epoch
                , stage=task
                , iterations=f'{(step+1)}/{total_len}'
                , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
                , loss=f'{(total_loss/(step+1))}'
                , results=mima_prf_results
            )

    if step == -1:
        logger.error('There is no data in this dataset.')
        raise Exception('There is no data in this dataset.')

    mima_prf_results = output_function(
        total_loss=total_loss
        , step=step
        , data=acc_result)

    delta_time = (timer() - start_time)

    log_results(
        epoch=current_epoch
        , stage=task
        , iterations=f'{(step+1)}/{total_len}'
        , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
        , loss=f'{(total_loss/(step+1))}'
        , results=mima_prf_results
    )

    if from_train == True:
        model.train()

    gc.collect()
    torch.cuda.empty_cache()
