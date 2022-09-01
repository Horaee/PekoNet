import logging

from torch.utils.data import DataLoader

from abstractive_text_summarization.dataset.JsonFromFiles import JsonFromFiles
from abstractive_text_summarization.formatter import initialize_formatter


logger = logging.getLogger(__name__)


def initialize_dataloader(config, task, mode, batch_size, *args, **kwargs):
    logger.info(f'Start to initialize {task} dataloader.')

    dataset_types = {'JsonFromFiles': JsonFromFiles}
    
    dataset_type = config.get('data', f'{task}_dataset_type')

    if dataset_type in dataset_types:
        dataset = dataset_types[dataset_type](config=config, task=task)
        batch_size = batch_size
        shuffle = config.getboolean(mode, 'shuffle')
        num_workers = config.getint(mode, 'num_workers')
        collate_fn = initialize_formatter(config=config, task=task)
        drop_last = False

        dataloader = DataLoader(
            dataset=dataset
            , batch_size=batch_size
            , shuffle=shuffle
            , num_workers=num_workers
            , collate_fn=collate_fn
            , drop_last=drop_last)

        logger.info(f'Initialize {task} dataloader successfully.')

        return dataloader
    else:
        logger.error(f'There is no dataset_type called {dataset_type}.')
        raise Exception(f'There is no dataset_type called {dataset_type}.')