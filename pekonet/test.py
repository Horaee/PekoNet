import logging
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import log_results, get_time_str


logger = logging.getLogger(__name__)


def test(parameters, mode):
    model = parameters['model']
    trained_epoch = parameters['trained_epoch']
    dataloader = parameters[f'{mode}_dataloader']
    output_function = parameters['output_function']
    output_time = parameters['output_time']

    with torch.no_grad():
        test_one(
            model=model
            , dataloader=dataloader
            , output_time=output_time
            , output_function=output_function
            , current_epoch=trained_epoch
            , task=mode
        )


def test_one(
        model
        , dataset
        , output_time
        , output_function
        , current_epoch
        , task
        , from_train=False):
    model.eval()

    total_len = len(dataset)

    start_time = timer()
    
    cls_loss = None
    cls_counter = 0
    cm_result = None
    mima_prf_results = ''
    step = -1

    for step, data in enumerate(dataset):
        for key in data.keys():
            data[key] = Variable(data[key].cuda())

        results = model(
            data=data
            , mode='train' if from_train else 'test'
            , cm_result=cm_result)

        if cls_loss == None:
            cls_loss = float(results['cls_loss'])
        else:
            cls_loss += float(results['cls_loss'])

        cls_counter += data['text'].size(0)

        cm_result = results['cm_result']

        if step % output_time == 0:
            mima_prf_results = output_function(data=cm_result)

            delta_time = (timer() - start_time)
            cls_loss_log = float(cls_loss / cls_counter)

            log_results(
                epoch=current_epoch
                , stage=task
                , iterations=f'{(step+1)}/{total_len}'
                , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
                , cls_loss=str(cls_loss_log)
                , results=mima_prf_results
            )

    if step == -1:
        logger.error('There is no data in this dataset.')
        raise Exception('There is no data in this dataset.')

    mima_prf_results = output_function(data=cm_result)

    delta_time = (timer() - start_time)
    cls_loss_log = float(cls_loss / cls_counter)

    log_results(
        epoch=current_epoch
        , stage=task
        , iterations=f'{(step+1)}/{total_len}'
        , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
        , cls_loss=str(cls_loss_log)
        , results=mima_prf_results
    )

    if from_train == True:
        model.train()

    gc.collect()
    torch.cuda.empty_cache()