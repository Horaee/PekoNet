import logging
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import log_results, get_time_str


logger = logging.getLogger(__name__)


def vot(parameters, mode, *args, **kwargs):
    model = parameters['model']
    trained_epoch = parameters['trained_epoch']
    dataloader = parameters[f'{mode}_dataloader']
    output_function = parameters['output_function']
    output_time = parameters['output_time']

    vot_one(
        model=model
        , dataloader=dataloader
        , output_time=output_time
        , output_function=output_function
        , current_epoch=trained_epoch
        , task=mode
    )

    # TODO: Just evaluation the performance of results.
    # if mode == 'validate':
    #     vot_one(
    #         model=model
    #         , dataloader=dataloader
    #         , output_time=output_time
    #         , output_function=output_function
    #         , current_epoch=trained_epoch
    #         , task=mode
    #     )
    # TODO: Generate and write results into file.
    # else:
    #     test_one(
    #         model=model
    #         , dataset=dataloader
    #         , output_time=output_time
    #         , output_function=output_function
    #         , current_epoch=trained_epoch
    #         , task=mode
    #     )


def vot_one(
        model
        , dataloader
        , output_time
        , output_function
        , current_epoch
        , task
        , from_train=False
        , *args
        , **kwargs):
    model.eval()

    cls_loss, cls_counter = None, 0
    cm_result, mima_prf_results = None, None
    dataloader_len = len(dataloader)

    start_time = timer()

    step = -1
    for step, data in enumerate(dataloader):
        for key in data.keys():
            data[key] = Variable(data[key].cuda())

        results = model(data=data, mode='validate', cm_result=cm_result)

        if cls_loss == None:
            cls_loss = float(results['cls_loss'])
        else:
            cls_loss += float(results['cls_loss'])

        cls_counter += results['cns_tci_data_number'][1]
        cm_result = results['cm_result']

        if step % output_time == 0:
            delta_time = (timer() - start_time)
            mima_prf_results = output_function(data=cm_result)

            loss = float(cls_loss / cls_counter)

            log_results(
                epoch=current_epoch
                , stage=task
                , iterations=f'{(step+1)}/{dataloader_len}'
                , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(dataloader_len-step-1)/(step+1)))}'
                , sum_loss=str(sum_loss_log)
                , cls_loss=str(cls_loss_log)
                , results=mima_prf_results
            )

    if step == -1:
        logger.error('There is no data in this dataset.')
        raise Exception('There is no data in this dataset.')

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
        , stage=task
        , iterations=f'{(step+1)}/{dataloader_len}'
        , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(dataloader_len-step-1)/(step+1)))}'
        , sum_loss=str(sum_loss_log)
        , cls_loss=str(cls_loss_log)
        , results=mima_prf_results
    )

    if from_train == True:
        model.train()

    gc.collect()
    torch.cuda.empty_cache()


# def test_one(
#         model
#         , dataset
#         , output_time
#         , output_function
#         , current_epoch
#         , task
#         , from_train=False
#         , *args
#         , **kwargs):
#     model.eval()

#     total_len = len(dataset)

#     start_time = timer()
    
#     cls_loss = None
#     cls_counter = 0
#     cm_result = None
#     mima_prf_results = ''
#     step = -1

#     for step, data in enumerate(dataset):
#         for key in data.keys():
#             data[key] = Variable(data[key].cuda())

#         results = model(
#             data=data
#             , mode='train' if from_train else 'test'
#             , cm_result=cm_result)

#         if cls_loss == None:
#             cls_loss = float(results['cls_loss'])
#         else:
#             cls_loss += float(results['cls_loss'])

#         cls_counter += data['text'].size(0)

#         cm_result = results['cm_result']

#         if step % output_time == 0:
#             mima_prf_results = output_function(data=cm_result)

#             delta_time = (timer() - start_time)
#             cls_loss_log = float(cls_loss / cls_counter)

#             log_results(
#                 epoch=current_epoch
#                 , stage=task
#                 , iterations=f'{(step+1)}/{total_len}'
#                 , time=f'{get_time_str(total_seconds=delta_time)}/\
# {get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
#                 , cls_loss=str(cls_loss_log)
#                 , results=mima_prf_results
#             )

#     if step == -1:
#         logger.error('There is no data in this dataset.')
#         raise Exception('There is no data in this dataset.')

#     mima_prf_results = output_function(data=cm_result)

#     delta_time = (timer() - start_time)
#     cls_loss_log = float(cls_loss / cls_counter)

#     log_results(
#         epoch=current_epoch
#         , stage=task
#         , iterations=f'{(step+1)}/{total_len}'
#         , time=f'{get_time_str(total_seconds=delta_time)}/\
# {get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
#         , cls_loss=str(cls_loss_log)
#         , results=mima_prf_results
#     )

#     if from_train == True:
#         model.train()

#     gc.collect()
#     torch.cuda.empty_cache()