import sys
import argparse
import configparser

from utils import initialize_logger
from abstractive_text_summarization.initialize import initialize_all
from abstractive_text_summarization.train import train
from abstractive_text_summarization.serve import serve


information = ' '.join(sys.argv)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config'
        , help='The path of config file'
        , required=True
    )
    parser.add_argument(
        '-m', '--mode'
        , help='train or serve'
        , required=True
    )
    parser.add_argument(
        '-g', '--gpu'
        , help='The list of GPU IDs'
        , required=True
    )
    parser.add_argument(
        '-cp', '--checkpoint_path'
        , help='The path of checkpoint (Ignore if you do not use checkpoint)'
    )
    parser.add_argument(
        '-bs', '--batch_size'
        , help='The batch size in train mode'
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    logger = initialize_logger(log_name=config.get('log', 'name'))    
    logger.info(f'\n\n\n\n{information}')

    parameters = initialize_all(
        config=config
        , mode=args.mode
        , device_str=args.gpu
        , batch_size_str=args.batch_size
        , checkpoint_path=args.checkpoint_path)

    if args.mode == 'train':
        train(parameters=parameters)
    elif args.mode == 'serve':
        serve(parameters=parameters)


if __name__ == '__main__':
    main()