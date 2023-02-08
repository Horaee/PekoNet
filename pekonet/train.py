import logging
import os
import torch
import gc

from timeit import default_timer as timer
from torch.autograd import Variable

from utils import log_results, get_time_str
from pekonet.validate import validate_one


logger = logging.getLogger(__name__)


# Checked.
def train(parameters, do_validation):
    model = parameters['model']
    optimizer = parameters['optimizer']
    exp_lr_scheduler = parameters['exp_lr_scheduler']
    optimizer_name = parameters['optimizer_name']
    trained_epoch = (parameters['trained_epoch'] + 1)
    output_function = parameters['output_function']
    output_path = parameters['output_path']
    output_time = parameters['output_time']
    test_time = parameters['test_time']
    use_mix_data = parameters['use_mix_data']
    sum_epoch = parameters['sum_epoch']

    if use_mix_data:
        total_epoch = sum_epoch + parameters['mix_epoch']
    else:
        total_epoch = sum_epoch + parameters['ljp_epoch']

    if do_validation == True:
        validate_dataloader = parameters['validate_dataloader']

    for current_epoch in range(trained_epoch, total_epoch):
        freeze_model = False
        loss, counter = 0, 0
        sum_loss, sum_counter = 0, 0
        cls_loss, cls_counter = 0, 0

        if current_epoch < sum_epoch:
            dataloader = parameters['sum_train_dataloader']
        else:
            if use_mix_data:
                dataloader = parameters['mix_train_dataloader']
            else:
                dataloader = parameters['ljp_train_dataloader']

            # Re-initialize learning rate for LJP training.
            if current_epoch == sum_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = parameters['learning_rate']

                freeze_model = False

        dataloader_len = len(dataloader)
        cm_results, mima_prf_results = None, None

        learning_rate = -1
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        start_time = timer()

        step = -1
        for step, data in enumerate(iterable=dataloader):
            for key in data.keys():
                data[key] = Variable(data[key].cuda())

            optimizer.zero_grad()

            if freeze_model:
                results = model(
                    data
                    , 'train'
                    , cm_results
                    , freeze_model=freeze_model)
            else:
                results = model(data=data, mode='train', cm_results=cm_results)

            # Sum training
            if current_epoch < sum_epoch:
                loss += float(results['sum_loss'])
                sum_loss += float(results['sum_loss'])
                counter += results['cns_tci_data_number'][0]
                sum_counter += results['cns_tci_data_number'][0]

                results['sum_loss'].backward()
            # Mix training
            else:
                batch_loss = (results['sum_loss'] + results['cls_loss'])
                batch_counter = (
                    results['cns_tci_data_number'][0]
                    + results['cns_tci_data_number'][1])

                loss += float(batch_loss)
                sum_loss += float(results['sum_loss'])
                cls_loss += float(results['cls_loss'])
                counter += batch_counter
                sum_counter += results['cns_tci_data_number'][0]
                cls_counter += results['cns_tci_data_number'][1]

                batch_loss.backward()

                cm_results = results['cm_results']

            optimizer.step()

            if step % output_time == 0:
                delta_time = (timer() - start_time)
                loss /= counter

                if sum_counter != 0:
                    sum_loss /= sum_counter
                
                if cls_counter != 0:
                    cls_loss /= cls_counter

                # Sum training
                if current_epoch < sum_epoch:
                    is_summarization = True
                # Mix training
                else:
                    is_summarization = False

                    if cm_results != None:
                        mima_prf_results = \
                            output_function(cm_results=cm_results)

                log_results(
                    epoch=current_epoch
                    , stage='train'
                    , iterations=f'{(step+1)}/{dataloader_len}'
                    , time=f'{get_time_str(delta_time)}/\
{get_time_str(delta_time*(dataloader_len-step-1)/(step+1))}'
                    , loss=str(round(number=loss, ndigits=7))
                    , sum_loss=sum_loss
                    , cls_loss=cls_loss
                    , is_summarization=is_summarization
                    , learning_rate=learning_rate
                    , results=mima_prf_results
                )

        if step == -1:
            logger.error('There is no data in this dataset.')
            raise Exception('There is no data in this dataset.')

        exp_lr_scheduler.step()

        delta_time = (timer() - start_time)

        # Sum training
        if current_epoch < sum_epoch:
            is_summarization = True
        # Mix training
        else:
            is_summarization = False

            if cm_results != None:
                mima_prf_results = output_function(cm_results=cm_results)

        log_results(
            epoch=current_epoch
            , stage='train'
            , iterations=f'{(step+1)}/{dataloader_len}'
            , time=f'{get_time_str(delta_time)}/\
{get_time_str(delta_time*(dataloader_len-step-1)/(step+1))}'
            , loss=str(round(number=loss, ndigits=7))
            , sum_loss=sum_loss
            , cls_loss=cls_loss
            , is_summarization=is_summarization
            , learning_rate=learning_rate
            , results=mima_prf_results
        )

        save_checkpoint(
            model=model
            , optimizer_name=optimizer_name
            , optimizer=optimizer
            , exp_lr_scheduler=exp_lr_scheduler
            , trained_epoch=current_epoch
            , file=os.path.join(output_path, f'checkpoint_{current_epoch}.pkl')
        )

        if current_epoch >= sum_epoch and \
                ((current_epoch - sum_epoch) % test_time == 0):
            if do_validation:
                with torch.no_grad():
                    validate_one(
                        model=model
                        , dataloader=validate_dataloader
                        , output_time=output_time
                        , output_function=output_function
                        , current_epoch=current_epoch
                        , task='validate'
                        , from_train=True
                    )

        gc.collect()
        torch.cuda.empty_cache()

    logger.info('Train model successfully.')


# Checked.
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
