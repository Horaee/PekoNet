import torch
import torch.nn as nn

from transformers import BartForConditionalGeneration, BertTokenizer


class BART(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(BART, self).__init__()

        self.bart = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.get(
                'model'
                , 'atsm_model_path'))

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.get(
                'model'
                , 'atsm_model_path'))

        self.pooler = BertPooler(
            hidden_size=config.getint('model', 'hidden_size'))

        self.min_len = config.getint('data', 'min_len')
        self.max_len = config.getint('data', 'max_len')


    def forward(self, data, mode, *args, **kwargs):
        if mode == 'serve':
            summary_id = self.bart.generate(
                inputs=data
                , min_length=self.min_len
                , max_length=self.max_len)

            # summary = self.tokenizer.batch_decode(
            #     sequences=summary_id
            #     , skip_special_tokens=True
            #     , clean_up_tokenization_spaces=False)[0]

            return summary_id
        # mode == 'train' or 'eval'
        else:
            cns_data, tci_data, cns_tci_data_number = \
                self.get_cns_and_tci_data(data)

            loss = None

            if cns_data:
                loss = self.bart(
                    input_ids=cns_data['text']
                    , labels=cns_data['summary'])['loss']

            summary_ids = None

            if tci_data:
                summary_ids = self.bart.generate(
                    inputs=tci_data['text']
                    , min_length=10
                    , max_length=50)

                # summaries = self.tokenizer.batch_decode(
                #     sequences=summary_ids
                #     , skip_special_tokens=True)

                summary_ids = self.get_summary_ids(summary_ids=summary_ids)

            return {
                'loss': loss
                , 'summary_ids': summary_ids
                , 'tci_data': tci_data
                , 'cns_tci_data_number': cns_tci_data_number
            }


    def get_cns_and_tci_data(self, data):
        cns_data, tci_data = {}, {}
        cns_data_number, tci_data_number = 0, 0

        for index in range(len(data['summary'])):
            for item in data:
                temp_data = torch.unsqueeze(input=data[item][index], dim=0)

                # summary exists -> CNS
                if torch.count_nonzero(data['summary'][index]) != 0:
                    if not item in cns_data:
                        cns_data[item] = temp_data
                    else:
                        cns_data[item] = torch.cat(
                            tensors=(cns_data[item], temp_data)
                            , dim=0)

                    cns_data_number += 1
                else:
                    if not item in tci_data:
                        tci_data[item] = temp_data
                    else:
                        tci_data[item] = torch.cat(
                            tensors=(tci_data[item], temp_data)
                            , dim=0)

                    tci_data_number += 1

        cns_data_number /= 4
        tci_data_number /= 4

        return cns_data, tci_data, (cns_data_number, tci_data_number)


    def get_summary_ids(self, summary_ids):
        summary_ids = summary_ids.tolist()

        for index in range(len(summary_ids)):
            temp_summary_id = []

            for small_index in range(len(summary_ids[index])):
                # Remove [CLS] and [SEP] tokens
                if summary_ids[index][small_index] != 101 and \
                        summary_ids[index][small_index] != 102:
                    temp_summary_id.append(summary_ids[index][small_index])

            # Add [CLS] token
            temp_summary_id.insert(0, 101)

            # Add [SEP] token
            temp_summary_id.append(102)

            current_id_length = len(temp_summary_id)

            for _ in range(self.max_len - current_id_length):
                temp_summary_id.append(0)

            summary_ids[index] = temp_summary_id

        summary_ids = torch.Tensor(summary_ids).to(torch.long).cuda()

        return summary_ids


    # Size of `summary_ids` = [batch_size, sequence_length]
    def get_clses_embedding(self, ids):
        # Size of `hidden_states` = [batch_size, sequence_length, hidden_size]
        hidden_states = \
            self.bart(input_ids=ids)['encoder_last_hidden_state']

        # Size of returning value = [batch_size, hidden_size]
        return self.pooler(hidden_states)


# This pooler is the same as Hugging Face's BERT design.
class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()

        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

        self.pooler = nn.Sequential(
            self.fc
            , self.activation
        )


    # Size of `hidden_states` = [batch_size, sequence_length, hidden_size].
    def forward(self, hidden_states):
        # `hidden_states[:, 0]` is [CLS] token.
        # Size of `hidden_states[:, 0]` = [batch_size, hidden_size].
        # Size of returning value is the same as `hidden_states[:, 0]`.
        return self.pooler(hidden_states[:, 0])