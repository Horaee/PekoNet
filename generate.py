import sys
import argparse
import configparser

from utils import initialize_logger
from data_generation.initialize import initialize_all
from data_generation.utils import \
    get_common_data \
    , get_bart_summary \
    , get_lead_3_summary \
    , get_ats_model_data


information = ' '.join(sys.argv)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config'
        , help='The path of config file'
        , required=True
    )
    parser.add_argument(
        '-g', '--gpu'
        , help='The list of GPU IDs (Require if summarization method is bart)'
    )
    parser.add_argument(
        '-cp', '--checkpoint_path'
        , help='The path of checkpoint \
(Require if summarization method is bart)'
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    logger = initialize_logger(log_name=config.get('log', 'name'))
    logger.info(f'\n\n\n\n{information}')

    parameters = initialize_all(
        config=config
        , device_str=args.gpu
        , checkpoint_path=args.checkpoint_path)

    logger.info(f'Start to generate data.')

    if parameters['task'] == 'legal_judgment_prediction':
        if parameters['summarization'] == 'none':
            get_common_data(parameters=parameters)
        elif parameters['summarization'] == 'bart':
            get_bart_summary(parameters)
        elif parameters['summarization'] == 'lead_3':
            get_lead_3_summary(parameters)
    elif parameters['task'] == 'abstractive_text_summarization':
        get_ats_model_data(parameters)

    logger.info('Generate data successfully.')


if __name__ == '__main__':
    main()