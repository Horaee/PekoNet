import sys
import argparse
import configparser

from utils import initialize_logger
from pekonet.initialize import initialize_all
from pekonet.train import train
from pekonet.validate import validate
from pekonet.test import test
from pekonet.serve import serve

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
        , help='train, validate, test or serve'
        , required=True
    )
    parser.add_argument(
        '-g', '--gpus'
        , help='The list of gpu IDs'
        , required=True
    )
    parser.add_argument(
        '-cp', '--checkpoint_path'
        , help='The path of checkpoint (Ignore if you do not use checkpoint)'
    )
    parser.add_argument(
        '-bs', '--batch_size'
        , help='The batch size in train, validate or test mode'
    )
    parser.add_argument(
        '-dv', '--do_validation'
        , help='Validate in train mode (Ignore if you do not validate)'
        , action='store_true'
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    logger = initialize_logger(log_name=config.get('log', 'name'))
    logger.info(f'{information}')

    parameters = initialize_all(
        config=config
        , mode=args.mode
        , gpu_ids_str=args.gpus
        , checkpoint_path=args.checkpoint_path
        , batch_size_str=args.batch_size
        , do_validation=args.do_validation)

    if args.mode == 'train':
        train(parameters=parameters, do_validation=args.do_validation)
    elif args.mode == 'validate':
        validate(parameters=parameters)
    elif args.mode == 'test':
        test(parameters=parameters)
    # If mode is 'serve'.
    else:
        serve(parameters=parameters)


if __name__ == '__main__':
    main()