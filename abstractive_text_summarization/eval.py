import logging
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import get_time_str, log_results


logger = logging.getLogger(__name__)


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
    message = ''
    step = -1

    for step, data in enumerate(dataset):
        for key in data.keys():
            data[key] = Variable(data[key].cuda())

        results = model(data=data, mode='eval')

        loss = results['loss']
        total_loss += float(loss)

        if step % output_time == 0:
            message = output_function()
            delta_time = (timer() - start_time)

            log_results(
                epoch=current_epoch
                , stage=task
                , iterations=f'{(step+1)}/{total_len}'
                , time=f'{get_time_str(delta_time)}/\
{get_time_str(delta_time*(total_len-step-1)/(step+1))}'
                , loss=f'{(total_loss/(step+1))}'
                , results=message
            )

    if step == -1:
        logger.error('There is no data in this dataset.')
        raise Exception('There is no data in this dataset.')

    message = output_function()
    delta_time = (timer() - start_time)

    log_results(
        epoch=current_epoch
        , stage=task
        , iterations=f'{(step+1)}/{total_len}'
        , time=f'{get_time_str(delta_time)}/\
{get_time_str(delta_time*(total_len-step-1)/(step+1))}'
        , loss=f'{(total_loss/(step+1))}'
        , results=message
    )

    if from_train == True:
        model.train()

    gc.collect()
    torch.cuda.empty_cache()