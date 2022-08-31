import json
import random

from torch.utils.data import Dataset


class JsonFromFiles(Dataset):
    def __init__(self, config, task, encoding='UTF-8', *args, **kwargs):
        self.file = config.get('data', f'{task}_file_path')
        self.data = []

        with open(file=self.file, mode='r', encoding=encoding) as json_file:
            for line in json_file:
                one_data = json.loads(line)
                self.data.append(one_data)

        if task == 'train':
            random.shuffle(self.data)


    def __getitem__(self, item, *args, **kwargs):
        return self.data[item]


    def __len__(self, *args, **kwargs):
        return len(self.data)
