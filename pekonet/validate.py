import logging
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import log_results, get_time_str


logger = logging.getLogger(__name__)


def validate(parameters):
    model = parameters['model']
    trained_epoch = parameters['trained_epoch']
    dataloader = parameters['validate_dataloader']
    output_function = parameters['output_function']
    output_time = parameters['output_time']

    with torch.no_grad():
        validate_one(
            model=model
            , dataloader=dataloader
            , output_time=output_time
            , output_function=output_function
            , current_epoch=trained_epoch
            , task='validate'
        )


def validate_one(
        model
        , dataloader
        , output_time
        , output_function
        , current_epoch
        , task
        , from_train=False):
    model.eval()

    cls_loss, cls_counter = None, 0
    dataloader_len = len(dataloader)
    cm_results, mima_prf_results = None, None

    start_time = timer()

    step = -1
    for step, data in enumerate(dataloader):
        for key in data.keys():
            data[key] = Variable(data[key].cuda())

        results = model(data=data, mode='validate', cm_results=cm_results)

        if cls_loss == None:
            cls_loss = float(results['cls_loss'])
        else:
            cls_loss += float(results['cls_loss'])

        cls_counter += results['cns_tci_data_number'][1]
        cm_results = results['cm_results']

        if step % output_time == 0:
            delta_time = (timer() - start_time)
            loss = float(cls_loss / cls_counter)
            mima_prf_results = output_function(cm_results=cm_results)

            log_results(
                epoch=current_epoch
                , stage=task
                , iterations=f'{(step+1)}/{dataloader_len}'
                , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(dataloader_len-step-1)/(step+1)))}'
                , loss=str(loss)
                , results=mima_prf_results
            )

    if step == -1:
        logger.error('There is no data in this dataset.')
        raise Exception('There is no data in this dataset.')

    delta_time = (timer() - start_time)
    loss = float(cls_loss / cls_counter)
    mima_prf_results = output_function(cm_results=cm_results)

    log_results(
        epoch=current_epoch
        , stage=task
        , iterations=f'{(step+1)}/{dataloader_len}'
        , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(dataloader_len-step-1)/(step+1)))}'
        , loss=str(loss)
        , results=mima_prf_results
    )

    if from_train == True:
        model.train()

    gc.collect()
    torch.cuda.empty_cache()