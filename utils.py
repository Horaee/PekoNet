import logging
import torch
import re

from tabulate import tabulate


logger = logging.getLogger(__name__)


# Initialize logger.
def initialize_logger(log_name):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(
        filename=f'logs/{log_name}'
        , mode='a'
        , encoding='UTF-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


# Initialize GPUs.
def initialize_gpus(device_str):
    if device_str == None:
        logger.error('There is no any given gpu.')
        raise Exception('There is no any given gpu.')

    if not torch.cuda.is_available():
        logger.error('CUDA is not available.')
        raise Exception('CUDA is not available.')
        
    devices = device_str.replace(' ', '').split(',')
    gpus = []

    for device in devices:
        gpus.append(int(device))

    logger.info(f'CUDA is available.')
    logger.info('Initialize GPUs successfully.')

    return gpus


def initialize_batch_size(batch_size_str):
    if batch_size_str == None:
        logger.warn(f'There is no batch_size.')
        logger.info('Set the batch_size to 1 to continue.')
        batch_size_str = '1'

    batch_size = int(batch_size_str)

    return batch_size


# This function is used to preprocess input string (fact).
def process_string(
        string
        , adjust_chars=False
        , process_fact=False
        , translator=None):

    # Remove special chars and unify punctuations.
    if adjust_chars == True:
        string = string.replace(' ', '')
        string = string.replace('\\', '')
        string = string.replace('`', '')
        string = string.replace('#', '')
        string = string.replace(',', '，')
        string = string.replace('：', ':')
        string = string.replace('；', ';')
        string = string.replace('？', '?')
        string = string.replace('！', '!')
        string = string.replace('（', '(')
        string = string.replace('）', ')')

    # Remove unused part which is in string (fact).
    if process_fact == True:
        string = get_fact(string=string)

    # Translate string (fact) to target language.
    if translator != None:
        string = translator.convert(string)

    return string


# Remove unused part which is in string (fact).
def get_fact(string):
    chinese_numbers = \
        ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    paragraphs = []
    record_index = 0

    # According titles ('一、', '二、', ...) to separate paragraphs.
    for index in range(len(string)):
        if (index + 1) < len(string):
            if string[index] == '。' and string[index+1] in chinese_numbers:
                for temp_index in range((index+1), len(string)):
                    if string[temp_index] == '、':
                        paragraphs.append(string[record_index:(index+1)])
                        record_index = (index + 1)

                    if string[temp_index] not in chinese_numbers:
                        break
        else:
            paragraphs.append(string[record_index:])

    # Remove the last paragraph (because last paragraph is unused content).
    paragraphs = paragraphs[:-1]

    # Remove the title ('一、', '二、', ...) of paragraphs.
    for index in range(len(paragraphs)):
        for temp_index in range(len(paragraphs[index])):
            if paragraphs[index][temp_index] == '、':
                paragraphs[index] = paragraphs[index][(temp_index+1):]
                break

    # Get all sentences that are in paragraphs.
    all_sentences = []
    for paragraph in paragraphs:
        part_sentences = re.split(pattern=r'([，。])', string=paragraph)

        for sentence in part_sentences:
            all_sentences.append(sentence)

    # Use 'Regular Expression' to remove the contents that are not fact.
    for index, sentence in enumerate(iterable=all_sentences):
        if re.match(
                pattern=r'讵[^，:;?!()]*[不未][^，:;?!()]*[悔惕戒]'
                , string=sentence) != None:
            if (index + 1) >= len(all_sentences):
                continue

            if len(all_sentences[index+1]) > 1:
                all_sentences = all_sentences[index+1:]
            else:
                all_sentences = all_sentences[index+2:]

    # Save the result.
    fact = ''
    for sentence in all_sentences:
        fact += sentence

    return fact


# Get article and accusations table.
def get_tables(config, formatter, *args, **kwargs):
    names = ['article', 'accusation']
    paths = {
        'article': config.get('data', 'articles_path')
        , 'accusation': config.get('data', 'accusations_path')
    }
    tables = {'article': {}, 'accusation': {}}

    for name in names:
        items = []

        with open(file=paths[name], mode='r', encoding='UTF-8') as file:
            lines = file.readlines()

            # Save article, article_source, accusation without '\n'
            for index in range(len(lines)):
                if lines[index][-1] == '\n':
                    items.append(lines[index][0:-1])
                else:
                    items.append(lines[index])

            file.close()

        # Convert article, article_source, accusation to one-hot encoding form.
        for item in items:
            tables[name][item] = formatter({name: item})
    
    return tables['article'], tables['accusation']


# Get the time string with 'minute:second' format.
def get_time_str(total_seconds):
    total_seconds = int(total_seconds)
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    return ('%2d:%02d' % (minutes, seconds))


# Generate and print log.
def log_results(
        epoch=None
        , stage=None
        , iterations=None
        , time=None
        , sum_loss = None
        , cls_loss = None
        # , loss=None
        , learning_rate=None
        , results=None):

    header2item = {
        'epoch': epoch
        , 'stage': stage
        , 'iterations': iterations
        , 'time': time
        , 'sum_loss': sum_loss
        , 'cls_loss': cls_loss
        # , 'loss': loss
        , 'learning_rate': learning_rate
    }

    information_headers = []
    information_table = [[]]

    for header in header2item:
        if header2item[header] != None:
            information_headers.append(header)
            information_table[0].append(header2item[header])

    if isinstance(results, dict):
        results_headers = [
            'Type'
            , 'MiP', 'MiR', 'MiF'
            , 'MaP', 'MaR', 'MaF'
        ]
        results_table = [
            [
                'article'
                , results['article']['mip']
                , results['article']['mir']
                , results['article']['mif']
                , results['article']['map']
                , results['article']['mar']
                , results['article']['maf']
            ]
            , [
                'accusation'
                , results['accusation']['mip']
                , results['accusation']['mir']
                , results['accusation']['mif']
                , results['accusation']['map']
                , results['accusation']['mar']
                , results['accusation']['maf']
            ]
        ]
    elif isinstance(results, str):
        results_headers = ['Message']
        results_table = [[results]]

    information = (
        '\n'
        + tabulate(
            tabular_data=information_table
            , headers=information_headers
            , tablefmt='pretty')
    )

    if results != '' and results != None:
        information += (
            '\n'
            + tabulate(
                tabular_data=results_table
                , headers=results_headers
                , tablefmt='pretty')
        )

    logger.info(f'{information}\n')