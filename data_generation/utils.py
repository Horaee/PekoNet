import logging
import os
import json
import shutil
import re
import copy

from timeit import default_timer as timer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import process_string, get_time_str, log_results


logger = logging.getLogger(__name__)


# Get the data without summarizing
# and the tables of articles, article_sources and accusations
def get_common_data(parameters):
    logger.info(f'Start to get common data.')

    articles_list = []

    # Get articles except the times appeared of articles
    # is lower than the lower bound.
    for article in parameters['articles_times_appeared_of_all_files']:
        if int(article[1]) >= parameters['article_lowerbound']:
            articles_list.append(article[0])

    data = []
    article_sources_list = []
    accusations_list = []

    # Get article_sources and accusations according to chosen articles.
    for file_name in os.listdir(path=parameters['data_path']):
        if file_name == 'README.md':
            continue

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for line in lines:
                data.append(line)

                item = json.loads(line)

                for relevant_article in item['meta']['relevant_articles']:
                    article = relevant_article[0] + relevant_article[1]
                    article_source = relevant_article[0]
                    accusation = item['meta']['accusation']

                    if article in articles_list:
                        if article_source not in article_sources_list \
                                and article_source != '':
                            article_sources_list.append(article_source)

                        if accusation not in accusations_list \
                                and accusation != '':
                            accusations_list.append(accusation)

            json_file.close()

    write_back_results(
        parameters=parameters
        , data=data
        , articles_list=articles_list
        , article_sources_list=article_sources_list
        , accusations_list=accusations_list
    )

    logger.info(f'Get common data successfully.')


# Get the summaries based on ATS model.
def get_bart_summary(parameters):
    logger.info('Start to get summaries based on ATS model.')

    formatter = parameters['formatter']
    model = parameters['model']
    output_time = parameters['output_time']
    data = []

    exclusion_file_names = [
        'README.md'
    ]

    copy_file_names = [
        'articles.txt'
        , 'article_sources.txt'
        , 'accusations.txt'
    ]

    for file_name in os.listdir(path=parameters['data_path']):
        if file_name in exclusion_file_names:
            continue

        if file_name in copy_file_names:
            shutil.copy(
                src=os.path.join(parameters['data_path'], file_name)
                , dst=os.path.join(parameters['output_path'], file_name)
            )

            continue

        logger.info(f'Start to process {file_name}.')

        data = []

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            total_len = len(lines)
            start_time = timer()
            step = -1

            for step, line in enumerate(tqdm(lines)):
                item = json.loads(line)

                fact = item['fact']
                fact = process_string(
                    string=fact
                    , adjust_chars=True
                    , process_fact=True)

                fact_tensor = formatter(data=fact)
                result = model(data=fact_tensor, mode='serve')

                item['fact'] = process_string(
                    string=result
                    , adjust_chars=True)

                data.append(json.dumps(item, ensure_ascii=False) + '\n')

                if step % output_time == 0:
                    delta_time = (timer() - start_time)

                    log_results(
                        stage=file_name
                        , iterations=f'{(step+1)}/{total_len}'
                        , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
                    )

            if step == -1:
                logger.error('There is no data in this file.')
                raise Exception('There is no data in this file.')

            delta_time = (timer() - start_time)

            log_results(
                stage=file_name
                , iterations=f'{(step+1)}/{total_len}'
                , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
            )

            json_file.close()

        with open(
                file=os.path.join(parameters['output_path'], file_name)
                , mode='w'
                , encoding='UTF-8') as json_file:
            for one_data in data:
                json_file.write(one_data)

            json_file.close()

        logger.info(f'Process {file_name} successfully.')

    logger.info('Get summaries based on ATS model successfully.')


# Get the summaries based on Lead-3 method.
def get_lead_3_summary(parameters):
    logger.info('Start to use lead-3 method to get summaries.')

    output_time = parameters['output_time']
    data = []

    exclusion_file_names = [
        'README.md'
    ]

    copy_file_names = [
        'articles.txt'
        , 'article_sources.txt'
        , 'accusations.txt'
    ]

    for file_name in os.listdir(path=parameters['data_path']):
        if file_name in exclusion_file_names:
            continue

        if file_name in copy_file_names:
            shutil.copy(
                src=os.path.join(parameters['data_path'], file_name)
                , dst=os.path.join(parameters['output_path'], file_name)
            )

            continue

        logger.info(f'Start to process {file_name}.')

        data = []

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            total_len = len(lines)
            start_time = timer()
            step = -1

            for step, line in enumerate(tqdm(lines)):
                item = json.loads(line)

                fact = item['fact']
                fact = process_string(
                    string=fact
                    , adjust_chars=True)
                fact = re.split(r'([。，；])', fact)

                summary = ''

                # The 5 in this part means 3 sentences and 2 punctuations.
                for counter in range(0, len(fact)):
                    if counter >= 5:
                        break

                    summary += fact[counter]

                summary += '。'
                item['fact'] = summary

                data.append(json.dumps(item, ensure_ascii=False) + '\n')

                if step % output_time == 0:
                    delta_time = (timer() - start_time)

                    log_results(
                        stage=file_name
                        , iterations=f'{(step+1)}/{total_len}'
                        , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
                    )

            if step == -1:
                logger.error('There is no data in this file.')
                raise Exception('There is no data in this file.')

            delta_time = (timer() - start_time)

            log_results(
                stage=file_name
                , iterations=f'{(step+1)}/{total_len}'
                , time=f'{get_time_str(total_seconds=delta_time)}/\
{get_time_str(total_seconds=(delta_time*(total_len-step-1)/(step+1)))}'
            )

            json_file.close()

        with open(
                file=os.path.join(parameters['output_path'], file_name)
                , mode='w'
                , encoding='UTF-8') as json_file:
            for one_data in data:
                json_file.write(one_data)

            json_file.close()

        logger.info(f'Process {file_name} successfully.')

    logger.info('Convert fact to summarization successfully.')


# Get the data that are used to train, eval and test ATS model.
def get_ats_model_data(parameters):
    logger.info('Start to get ATS model data.')

    data = load_data(data_path=parameters['data_path'])
    results = []

    for one_data in data:
        one_data = json.loads(one_data)

        text = ''
        summary = one_data['summary']

        if parameters['type'] == 'CNewSum_v2':
            for item in one_data['article']:
                text += item

        result = {
            'text': process_string(string=text, adjust_chars=True)
            , 'summary': process_string(string=summary, adjust_chars=True)
        }

        results.append(json.dumps(result, ensure_ascii=False) + '\n')

    write_back_results(parameters, data=results)

    logger.info('Get ATS model data successfully.')


def get_integration_data(parameters):
    logger.info('Start to get integration data.')

    results = []
    templete = {
        'text': ''
        , 'summary': ''
        , 'relevant_articles': []
        , 'accusation': ''
        # , 'type': -1
    }

    cns_data = load_data(data_path=parameters['cns_data_path'])

    for one_data in cns_data:
        one_data = json.loads(one_data)

        result = copy.deepcopy(templete)
        result['text'] = process_string(
            string=one_data['text']
            , adjust_chars=True)
        result['summary'] = process_string(
            string=one_data['summary']
            , adjust_chars=True)
        # result['type'] = 3

        results.append(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info('Start to copy 3A files from source to destination.')

    aaa_files = ['articles.txt', 'article_sources.txt', 'accusations.txt']
    for file in aaa_files:
        src = os.path.join(parameters['tci_data_path'], file)
        dst = os.path.join(parameters['output_path'], file)
        shutil.copyfile(src=src, dst=dst)

    logger.info('Copy 3A files from source to destination successfully.')

    except_files = aaa_files
    except_files.append('test.json')

    tci_data = load_data(
        data_path=parameters['tci_data_path']
        , except_files=except_files)

    for one_data in tci_data:
        try:
            one_data = json.loads(one_data)
        except:
            print(one_data)

        result = copy.deepcopy(templete)
        result['text'] = process_string(
            string=one_data['fact']
            , adjust_chars=True
            , process_fact=True)
        result['relevant_articles'] = one_data['meta']['relevant_articles']
        result['accusation'] = one_data['meta']['accusation']
        # result['type'] = 1

        results.append(json.dumps(result, ensure_ascii=False) + '\n')

        # result['type'] = 2
        # results.append(json.dumps(result, ensure_ascii=False) + '\n')

    write_back_results(parameters, data=results)

    logger.info('Get integration data successfully.')


# Load all data from all files in given path.
def load_data(data_path, except_files=[]):
    logger.info('Start to load data.')

    data = []

    for file_name in os.listdir(data_path):
        if file_name == 'README.md' or file_name in except_files:
            continue

        logger.info(f'Start to process {file_name}.')

        with open(
                file=os.path.join(data_path, file_name)
                , mode='r'
                , encoding='UTF-8') as file:
            lines = file.readlines()

            for line in lines:
                data.append(line)

            file.close()

        logger.info(f'Process {file_name} successfully.')

    logger.info('Load data successfully.')

    return data


# Write back the results.
def write_back_results(
        parameters
        , data
        , articles_list=None
        , article_sources_list=None
        , accusations_list=None):
    logger.info('Start to write back results.')

    # This part will run if the method of summarization is none.
    # Write back the tables of articles, article_sources and accusations.
    if articles_list != None or article_sources_list != None or accusations_list != None:
        write_aaa_data(
            parameters=parameters
            , articles_list=articles_list
            , article_sources_list=article_sources_list
            , accusations_list=accusations_list
        )

    write_tvt_data(
        parameters=parameters
        , data=data
    )

    logger.info('Write back results successfully.')


# Write back the tables of articles, article_sources and accusations.
def write_aaa_data(
        parameters
        , articles_list
        , article_sources_list
        , accusations_list):
    name_to_data = {
        'articles.txt': articles_list
        , 'article_sources.txt': article_sources_list
        , 'accusations.txt': accusations_list
    }

    for name in name_to_data:
        logger.info(f'Start to write {name}.')

        with open(
                file=os.path.join(parameters['output_path'], name)
                , mode='w'
                , encoding='UTF-8') as txt_file:
            for data in name_to_data[name]:
                txt_file.write(f'{data}\n')

            txt_file.write('others\n')
            txt_file.close()

        logger.info(f'Write {name} successfully.')


# Split train, valid and test data and write them back.
def write_tvt_data(
        parameters
        , data):
    # unused_data, all_data = train_test_split(
    #     data
    #     , train_size=0.5
    #     , random_state=parameters['random_seed'])
    train_data, valid_data = train_test_split(
        data
        # all_data
        , train_size=parameters['train_size']
        , random_state=parameters['random_seed'])

    file_name_to_data = {'train.json': train_data, 'valid.json': valid_data}

    if parameters['generate_test_data'] is True:
        valid_data, test_data = train_test_split(
            valid_data
            , train_size=parameters['valid_size']
            , random_state=parameters['random_seed'])

        file_name_to_data['valid.json'] = valid_data
        file_name_to_data['test.json'] = test_data

    for file_name in file_name_to_data:
        logger.info(f'Start to write {file_name}.')

        with open(
                file=os.path.join(parameters['output_path'], file_name)
                , mode='w'
                , encoding='UTF-8') as json_file:
            for data in file_name_to_data[file_name]:
                json_file.write(f'{data}')

            json_file.close()

        logger.info(f'Write {file_name} successfully.')