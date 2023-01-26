from pekonet.evaluation import get_micro_macro_prf


# Checked.
def empty_output_function(*args, **kwargs):
    return ''


# Checked.
def aa_output_function(cm_results):
    return {
        'article': get_micro_macro_prf(cm_results=cm_results['article'])
        , 'accusation': get_micro_macro_prf(cm_results=cm_results['accusation'])
    }