import sys
import argparse
import configparser

from utils import initialize_logger
from pekonet.initialize import initialize_all
from pekonet.train import train
# from pekonet.eval import eval
# from pekonet.serve import serve


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
        , help='train, eval or serve'
        , required=True
    )
    parser.add_argument(
        '-g', '--gpu'
        , help='The list of gpu IDs'
    )
    parser.add_argument(
        '-cp', '--checkpoint_path'
        , help='The path of checkpoint (Ignore if you do not use checkpoint)'
    )
    parser.add_argument(
        '-bs', '--batch_size'
        , help='The batch size in train or eval mode'
    )
    # parser.add_argument(
    #     '-dt', '--do_test'
    #     , help='Test in train mode (Ignore if you do not test)'
    #     , action='store_true'
    # )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    logger = initialize_logger(log_name=config.get('log', 'name'))
    logger.info(f'\n\n\n\n{information}')

    parameters = initialize_all(
        config=config
        , mode=args.mode
        , device_str=args.gpu
        , checkpoint_path=args.checkpoint_path
        , batch_size_str=args.batch_size)
        # , do_test=args.do_test)

    if args.mode == 'train':
        # train(parameters=parameters, do_test=args.do_test)
        train(parameters=parameters)
    # elif args.mode == 'eval':
    #     eval(parameters=parameters)
    # elif args.mode == 'serve':
    #     serve(parameters=parameters)


if __name__ == '__main__':
    main()