def set_special_tokens(add_tokens_at_beginning, data_max_len, data):
    if add_tokens_at_beginning == True:
        data.insert(0, '[CLS]')
        data.append('[SEP]')

    if len(data) > data_max_len:
        data = data[0:data_max_len-1]
        data.append('[SEP]')

    while len(data) < data_max_len:
        data.append('[PAD]')

    return data