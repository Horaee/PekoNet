import sys
import argparse

from utils import initialize_logger
from text_conversion.initialize import initialize_all
from text_conversion.utils import text_conversion


information = ' '.join(sys.argv)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sdp', '--source_directory_path'
        , help='The path of source file'
        , required=True
    )
    parser.add_argument(
        '-ddp', '--destination_directory_path'
        , help='The path of destination directory'
        , required=True
    )
    parser.add_argument(
        '-c', '--config'
        , help='The config of OpenCC'
    )

    args = parser.parse_args()

    logger = initialize_logger(log_name='text_conversion.log')
    logger.info(f'\n\n\n\n{information}')

    parameters = initialize_all(args=args)

    text_conversion(parameters=parameters)


if __name__ == '__main__':
    main()