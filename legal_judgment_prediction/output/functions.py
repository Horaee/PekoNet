import json

from legal_judgment_prediction.evaluation import get_micro_macro_prf


def empty_output_function(*args, **kwargs):
    return ''


def bert_output_function(*args, **kwargs):
    data = kwargs['data']

    results = {}
    results['article'] = get_micro_macro_prf(
        data=data['article'])
    results['article_source'] = get_micro_macro_prf(
        data=data['article_source'])
    results['accusation'] = get_micro_macro_prf(
        data=data['accusation'])

    return results