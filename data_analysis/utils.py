import logging
import os
import json
import datetime
import pickle

from utils import process_string


logger = logging.getLogger(__name__)


# Analyze all files of dataset.
def files_analyze(parameters):
    logger.info('Start to analyze all files.')

    all_concepts = []

    concepts = [
        '----- Files Analyzing Task -----\n'
        , f'Dataset name: {parameters["name"]}\n'
        , f'Current time: {str(datetime.datetime.now())}\n'
        , 'The concepts of files:\n'
    ]

    all_concepts.append(concepts)

    if parameters['type'] == 'taiwan_indictments':
        results, all_concepts = taiwan_indictments_files_analyze(
            parameters=parameters
            , all_concepts=all_concepts)
    elif parameters['type'] == 'CNewSum_v2':
        results, all_concepts = cnewsum_v2_files_analyze(
            parameters=parameters
            , all_concepts=all_concepts)

    logger.info('Analyze all files successfully.')
    logger.info('Start to write results.')

    with open(
            file=os.path.join(
                parameters['output_path']
                , 'files_analysis.txt')
            , mode='a'
            , encoding='UTF-8') as txt_file:
        for concepts in all_concepts:
            for concept in concepts:
                txt_file.write(concept)

        txt_file.close()

    logger.info('Write results successfully.')

    return results


# Analyze all files of Taiwan Indictments dataset.
def taiwan_indictments_files_analyze(parameters, all_concepts):
    fact_lengths_of_all_files = []
    articles_times_appeared_of_all_files = {}
    article_sources_times_appeared_of_all_files = {}
    accusations_times_appeared_of_all_files = {}

    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        fact_lengths = []
        source_indictment_files_times_cited = {}
        articles_times_appeared = {}
        article_sources_times_appeared = {}
        accusations_times_appeared = {}
        articles_numbers_of_every_data = []
        criminals_times_appeared = {}
        criminals_numbers_of_every_data = []
        indexes_of_death_penalty_not_null = []
        indexes_of_imprisonment_not_null = []
        indexes_of_life_imprisonment_not_null = []
        data_number = 0
        concepts = []

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for index, line in enumerate(lines):
                data = json.loads(line)

                fact = data['fact']
                fact = process_string(string=fact, adjust_chars=True)

                # Process the length of fact.
                fact_lengths.append(len(fact))
                fact_lengths_of_all_files.append(len(fact))

                # Process the times cited of source indictment file.
                if data['file'] not in source_indictment_files_times_cited:
                    source_indictment_files_times_cited[data['file']] = 0

                source_indictment_files_times_cited[data['file']] += 1

                # Process the times appeared of article and article_source.
                for relevant_article in data['meta']['relevant_articles']:
                    article = relevant_article[0] + relevant_article[1]
                    article_source = relevant_article[0]

                    article = process_string(string=article, adjust_chars=True)

                    article_source = process_string(
                        string=article_source
                        , adjust_chars=True)

                    if article not in articles_times_appeared:
                        articles_times_appeared[article] = 0

                    if article not in articles_times_appeared_of_all_files:
                        articles_times_appeared_of_all_files[article] = 0

                    if article_source not in article_sources_times_appeared:
                        article_sources_times_appeared[article_source] = 0

                    if article_source not in \
                            article_sources_times_appeared_of_all_files:
                        article_sources_times_appeared_of_all_files[
                            article_source] = 0

                    articles_times_appeared[article] += 1
                    articles_times_appeared_of_all_files[article] += 1
                    article_sources_times_appeared[article_source] += 1
                    article_sources_times_appeared_of_all_files[
                        article_source] += 1

                # Process the times appeared of accusation.
                accusation = data['meta']['accusation']

                accusation = process_string(
                    string=accusation
                    , adjust_chars=True)

                if accusation not in accusations_times_appeared:
                    accusations_times_appeared[accusation] = 0

                if accusation not in accusations_times_appeared_of_all_files:
                    accusations_times_appeared_of_all_files[accusation] = 0

                accusations_times_appeared[accusation] += 1
                accusations_times_appeared_of_all_files[accusation] += 1

                # Process the number of articles in each data.
                articles_numbers_of_every_data.append(
                    data['meta']['#_relevant_articles'])

                # Process the times appeared of criminals.
                for criminal in data['meta']['criminals']:
                    if criminal not in criminals_times_appeared:
                        criminals_times_appeared[criminal] = 0

                    criminals_times_appeared[criminal] += 1

                # Process the number of criminals in each data.
                criminals_numbers_of_every_data.append(
                    data['meta']['#_criminals'])

                # Process the data indexes which death penalty is not null.
                if data['meta']['term_of_imprisonment']['death_penalty'] \
                        != None:
                    indexes_of_death_penalty_not_null.append(index)

                # Process the data indexes which imprisonment is not null.
                if data['meta']['term_of_imprisonment']['imprisonment'] != None:
                    indexes_of_imprisonment_not_null.append(index)

                # Process the data indexes which life imprisonment is not null.
                if data['meta']['term_of_imprisonment'] \
                        ['life_imprisonment'] != None:
                    indexes_of_life_imprisonment_not_null.append(index)

                # Count the number of data.
                data_number += 1

            json_file.close()

        # Sort the items by value from big to small.
        source_indictment_files_times_cited = sorted(
            source_indictment_files_times_cited.items()
            , key=lambda item:item[1]
            , reverse=True)
        articles_times_appeared = sorted(
            articles_times_appeared.items()
            , key=lambda item:item[1]
            , reverse=True)
        article_sources_times_appeared = sorted(
            article_sources_times_appeared.items()
            , key=lambda item:item[1]
            , reverse=True)
        accusations_times_appeared = sorted(
            accusations_times_appeared.items()
            , key=lambda item:item[1]
            , reverse=True)
        criminals_times_appeared = sorted(
            criminals_times_appeared.items()
            , key=lambda item:item[1]
            , reverse=True)

        logger.info('Start to generate the concepts')

        # Add the name of this file.
        concepts.append(f'\t- {file_name}\n')

        # Add the average length of facts in this file.
        facts_average_length = float(sum(fact_lengths) / len(fact_lengths))

        concepts.append(f'\t\t- The average length of facts: \
{facts_average_length}\n')

        # Add the times cited of source indictment files in this file.
        # Commented out, these are unused information.
        # concepts.append('\t\t- The times cited of files:\n')

        # for item in source_indictment_files_times_cited:
        #     # If the value of item is 1, all values after item are all 1.
        #     if item[1] == 1:
        #         concepts.append('\t\t\t- All times cited of other files: 1\n')
        #         break

        #     concepts.append(f'\t\t\t- {str(item[0])}: {str(item[1])}\n')

        # Add the times appeared of articles in this file.
        concepts.append('\t\t- The times appeared of relevant articles:\n')

        for item in articles_times_appeared:
            # If the value of item is 1, all values after item are all 1.
            if item[1] == 1:
                concepts.append('\t\t\t- All times appeared of \
other relevant_articles: 1\n')
                break

            concepts.append(f'\t\t\t- {str(item[0])}: {str(item[1])}\n')

        # Add the times appeared of article_sources in this file.
        concepts.append('\t\t- The times appeared of \
relevant article_sources:\n')

        for item in article_sources_times_appeared:
            # If the value of item is 1, all values after item are all 1.
            if item[1] == 1:
                concepts.append('\t\t\t- All times appeared of \
other relevant_article_sources: 1\n')
                break

            concepts.append(f'\t\t\t- {str(item[0])}: {str(item[1])}\n')

        # Add the times appeared of accusations in this file.
        concepts.append('\t\t- The times appeared of accusations:\n')

        for item in accusations_times_appeared:
            # If the value of item is 1, all values after item are all 1.
            if item[1] == 1:
                concepts.append('\t\t\t- All times appeared of \
other accusations: 1\n')
                break

            concepts.append(f'\t\t\t- {str(item[0])}: {str(item[1])}\n')

        # Add the average number of articles in this file.
        articles_average_number = float(
            sum(articles_numbers_of_every_data) 
            / len(articles_numbers_of_every_data))

        concepts.append(f'\t\t- The average number of relevant \
articles: {articles_average_number}\n')

        # Add the times appeared of criminals in this file.
        # Commented out, these are unused information.
#         concepts.append('\t\t- The times appeared of criminals:\n')

#         for item in criminals_times_appeared:
#             # If the value of item is 1, all values after item are all 1.
#             if item[1] == 1:
#                 concepts.append('\t\t\t- All times appeared of \
# other criminals: 1\n')
#                 break

#             concepts.append(f'\t\t\t- {str(item[0])}: {str(item[1])}\n')

        # Add the average number of criminals in this file.
        # Commented out, this is unused information.
#         criminals_average_number = float(
#             sum(criminals_numbers_of_every_data) 
#             / len(criminals_numbers_of_every_data))

#         concepts.append(f'\t\t- The average number of criminals: \
# {criminals_average_number}\n')

        # Add the indexes of data which 'death_penalty' is not null
        # in this file.
        # Commented out, all data's 'death_penalty' are null.
#         concepts.append('\t\t- The indexes of data \
# which \'death_penalty\' is not null:\n')

#         for item in indexes_of_death_penalty_not_null:
#             concepts.append(f'\t\t\t- {str(item)}\n')

        # Add the indexes of data which 'imprisonment' is not null
        # in this file.
        # Commented out, all data's 'imprisonment' are null.
#         concepts.append('\t\t- The indexes of data \
# which \'imprisonment\' is not null:\n')

#         for item in indexes_of_imprisonment_not_null:
#             concepts.append(f'\t\t\t- {str(item)}\n')

        # Add the indexes of data which 'life_imprisonment' is not null
        # in this file.
        # Commented out, all data's 'life_imprisonment' are null.
#         concepts.append('\t\t- The indexes of data \
# which \'life_imprisonment\' is not null:\n')

#         for item in indexes_of_life_imprisonment_not_null:
#             concepts.append(f'\t\t\t- {str(item)}\n')

        # Add the number of data in this file.
        concepts.append(f'\t\t- The number of data in this file: \
{data_number}\n')

        logger.info('Generate the concepts successfully.')

        all_concepts.append(concepts)

    # Sort the items by value from big to small.
    articles_times_appeared_of_all_files = sorted(
        articles_times_appeared_of_all_files.items()
        , key=lambda item:item[1]
        , reverse=True)
    article_sources_times_appeared_of_all_files = sorted(
        article_sources_times_appeared_of_all_files.items()
        , key=lambda item:item[1]
        , reverse=True)
    accusations_times_appeared_of_all_files = sorted(
        accusations_times_appeared_of_all_files.items()
        , key=lambda item:item[1]
        , reverse=True)

    results = {
        'fact_lengths_of_all_files': fact_lengths_of_all_files,
        'articles_times_appeared_of_all_files': \
            articles_times_appeared_of_all_files,
        'article_sources_times_appeared_of_all_files': \
            article_sources_times_appeared_of_all_files,
        'accusations_times_appeared_of_all_files': \
            accusations_times_appeared_of_all_files
    }

    return results, all_concepts


# Analyze all files of CNewSum dataset.
def cnewsum_v2_files_analyze(parameters, all_concepts):
    article_lengths_of_all_files = []
    summary_lengths_of_all_files = []

    for file_name in os.listdir(parameters['data_path']):
        if file_name == 'README.md':
            continue

        article_lengths = []
        summary_lengths = []
        data_number = 0
        max_article_length = float('-inf')
        min_article_length = float('inf')
        max_summary_length = float('-inf')
        min_summary_length = float('inf')

        with open(
                file=os.path.join(parameters['data_path'], file_name)
                , mode='r'
                , encoding='UTF-8') as json_file:
            lines = json_file.readlines()

            for line in lines:
                data = json.loads(line)

                article = ''

                for string in data['article']:
                    article += string

                article = process_string(string=article, adjust_chars=True)

                # Update the max and min length of article.
                if len(article) > max_article_length:
                    max_article_length = len(article)

                if len(article) < min_article_length:
                    min_article_length = len(article)

                article_lengths.append(len(article))
                article_lengths_of_all_files.append(len(article))

                summary = data['summary']
                summary = process_string(string=summary, adjust_chars=True)

                # Update the max and min length of summary.
                if len(summary) > max_summary_length:
                    max_summary_length = len(summary)

                if len(summary) < min_summary_length:
                    min_summary_length = len(summary)

                summary_lengths.append(len(summary))
                summary_lengths_of_all_files.append(len(summary))

                # Count the number of data.
                data_number += 1

            json_file.close()

        logger.info('Start to generate the concepts.')

        articles_average_length = float(
            sum(article_lengths) / len(article_lengths))
        summaries_average_length = float(
            sum(summary_lengths) / len(summary_lengths))

        concepts = [
            f'\t- {file_name}\n'
            , f'\t\t- The average length of articles: \
{articles_average_length}\n'
            , f'\t\t- The maximum length of articles: {max_article_length}\n'
            , f'\t\t- The minimum length of articles: {min_article_length}\n'
            , f'\t\t- The average length of summaries: \
{summaries_average_length}\n'
            , f'\t\t- The maximum length of summaries: {max_summary_length}\n'
            , f'\t\t- The minimum length of summaries: {min_summary_length}\n'
            , f'\t\t- The number of data in this file: {data_number}\n'
        ]

        logger.info('Generate the concepts successfully.')

        all_concepts.append(concepts)

    results = {
        'article_lengths_of_all_files': article_lengths_of_all_files,
        'summary_lengths_of_all_files': summary_lengths_of_all_files
    }

    return results, all_concepts


# Analyze whole dataset.
def general_analyze(parameters, results):
    logger.info('Start to analyze whole dataset.')

    concepts = [
        '----- General Analysis Task -----\n'
        , f'Dataset name: {parameters["name"]}\n'
        , f'Current time: {str(datetime.datetime.now())}\n'
        , 'File list:\n'
    ]

    if parameters['type'] == 'taiwan_indictments':
        concepts = taiwan_indictments_general_analyze(
            parameters=parameters
            , results=results
            , concepts=concepts)
    elif parameters['type'] == 'CNewSum_v2':
        concepts = cnewsum_v2_general_analyze(
            parameters=parameters
            , results=results
            , concepts=concepts)

    logger.info('Start to write results.')

    output_file_path = \
        os.path.join(parameters['output_path'], 'general_analysis.txt')

    with open(
            file=output_file_path
            , mode='a'
            , encoding='UTF-8') as txt_file:
        for concept in concepts:
            txt_file.write(concept)

        txt_file.close()

    logger.info('Write results successfully.')
    logger.info('Analyze whole dataset successfully.')


def taiwan_indictments_general_analyze(parameters, results, concepts):
    fact_lengths_of_all_files = results['fact_lengths_of_all_files']
    articles_times_appeared_of_all_files = \
        results['articles_times_appeared_of_all_files']
    article_sources_times_appeared_of_all_files = \
        results['article_sources_times_appeared_of_all_files']
    accusations_times_appeared_of_all_files = \
        results['accusations_times_appeared_of_all_files']

    file_names_strings, number_of_files = get_file_names_strings(
        data_path=parameters['data_path'])

    # Process the names of files in this dataset.
    for file_name_string in file_names_strings:
        concepts.append(file_name_string)

    concepts.append(
        f'Totol number of this dataset files: {str(number_of_files)}\n')

    # Process the architecture of data in this dataset.
    concepts.append('The architecture of data:\n')

    # Choose anyone file to analyze.
    any_file_name = ''

    for file_name in os.listdir(parameters['data_path']):
        if file_name != 'README.md':
            any_file_name = file_name
            break

    if any_file_name == '':
        logger.error('There is no any file in this data path.')
        raise Exception('There is no any file in this data path.')

    # Travel all nodes of a data in this file.
    nodes_strings = []

    with open(
            file=os.path.join(parameters['data_path'], any_file_name)
            , mode='r'
            , encoding='UTF-8') as json_file:
        line = json_file.readline()
        data = json.loads(line)

        logger.info(
            'Start to traversal and save the architecture of this data.')

        nodes_strings = traversal_all_nodes(
            nodes_strings=nodes_strings
            , data=data
            , tab_num=1)

        for node_string in nodes_strings:
            concepts.append(node_string)

        logger.info('Traversal the architecture of this data successfully.')

        json_file.close()

    logger.info('Start to generate the concepts.')

    # Process the average length of all facts.
    facts_average_length = float(
        sum(fact_lengths_of_all_files) / len(fact_lengths_of_all_files))

    concepts.append(f'The average length of facts: {facts_average_length}\n')

    # Process the times appeared of all articles.
    concepts.append('The times appeared of relevant articles:\n')

    for item in articles_times_appeared_of_all_files:
        # If the value of item is 1, all values after item are all 1.
        if item[1] == 1:
            concepts.append(
                '\t- All times appeared of other relevant_articles: 1\n')
            break

        concepts.append(f'\t- {str(item[0])}: {str(item[1])}\n')

    # Process the times appeared of all article_sources.
    concepts.append('The times appeared of relevant article_sources:\n')

    for item in article_sources_times_appeared_of_all_files:
        # If the value of item is 1, all values after item are all 1.
        if item[1] == 1:
            concepts.append('\t- All times appeared of other \
relevant_article_sources: 1\n')
            break

        concepts.append(f'\t- {str(item[0])}: {str(item[1])}\n')

    # Process the times appeared of all accusations.
    concepts.append('The times appeared of accusations:\n')

    for item in accusations_times_appeared_of_all_files:
        # If the value of item is 1, all values after item are all 1.
        if item[1] == 1:
            concepts.append(
                '\t- All times appeared of other accusations: 1\n')
            break
        
        concepts.append(f'\t- {str(item[0])}: {str(item[1])}\n')

    logger.info('Generate the concepts successfully.')

    return concepts


def cnewsum_v2_general_analyze(parameters, results, concepts):
    # Process the names of files in this dataset.
    file_names_strings, number_of_files = get_file_names_strings(
        data_path=parameters['data_path'])

    for file_name_string in file_names_strings:
        concepts.append(file_name_string)

    # Add the number of files in this dataset.
    concepts.append(
        f'Totol number of this dataset files: {str(number_of_files)}\n')

    # Process the architecture of data in this dataset.
    concepts.append('The architecture of data:\n')

    any_file_name = ''

    for file_name in os.listdir(parameters['data_path']):
        if file_name != 'README.md':
            any_file_name = file_name
            break

    if any_file_name == '':
        logger.error('There is no any file in this data path.')
        raise Exception('There is no any file in this data path.')

    nodes_strings = []

    with open(
            file=os.path.join(parameters['data_path'], any_file_name)
            , mode='r'
            , encoding='UTF-8') as json_file:
        line = json_file.readline()
        data = json.loads(line)

        logger.info(
            'Start to traversal and save the architecture of this data.')

        nodes_strings = traversal_all_nodes(
            nodes_strings=nodes_strings
            , data=data
            , tab_num=1)

        for node_string in nodes_strings:
            concepts.append(node_string)

        logger.info('Traversal the architecture of this data successfully.')

        json_file.close()

    article_lengths_of_all_files = results['article_lengths_of_all_files']
    summary_lengths_of_all_files = results['summary_lengths_of_all_files']

    logger.info('Start to generate the concepts.')

    # Add the average length of articles.
    articles_average_length = float(
        sum(article_lengths_of_all_files) / len(article_lengths_of_all_files))

    concepts.append(
        f'The average length of articles: {articles_average_length}\n')

    # Add the average length of summaries.
    summaries_average_length = float(
        sum(summary_lengths_of_all_files) / len(summary_lengths_of_all_files))

    concepts.append(
        f'The average length of summaries: {summaries_average_length}\n')

    logger.info('Generate the concepts.')

    return concepts


# Get file names strings from the input data path.
def get_file_names_strings(data_path):
    file_names_strings = []
    number_of_files = 0

    for file_name in os.listdir(data_path):
        if file_name == 'README.md':
            continue

        file_names_strings.append(f'\t- {file_name}\n')
        number_of_files += 1

    return file_names_strings, number_of_files


# Traversal all node in this input data.
def traversal_all_nodes(nodes_strings, data, tab_num):
    if type(data) == dict:
        for item in data:
            tab_string = ('\t' * tab_num)
            nodes_strings.append(f'{tab_string}- {item}\n')
            nodes_strings = traversal_all_nodes(
                nodes_strings=nodes_strings
                , data=data[item]
                , tab_num=tab_num+1)

    return nodes_strings


# Save the information that is used in LJP task into .pkl file.
def write_back_results(parameters, results):
    with open(
            file=os.path.join(parameters['output_path'], 'parameters.pkl')
            , mode='wb') as pkl_file:
        pickle.dump(results, pkl_file)

        pkl_file.close()