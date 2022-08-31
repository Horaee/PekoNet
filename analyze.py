import sys
import argparse
import configparser

from utils import initialize_logger
from data_analysis.initialize import initialize_all
from data_analysis.utils import \
    files_analyze, general_analyze, write_back_results


information = ' '.join(sys.argv)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config'
        , help='The path of config file'
        , required=True
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    logger = initialize_logger(log_name=config.get('log', 'name'))
    logger.info(f'\n\n\n\n{information}')

    parameters = initialize_all(config=config)

    logger.info('Start to analyze data.')

    results = files_analyze(parameters=parameters)
    general_analyze(parameters=parameters, results=results)
    write_back_results(parameters=parameters, results=results)

    logger.info('Analyze data successfully.')


if __name__ == '__main__':
    main()