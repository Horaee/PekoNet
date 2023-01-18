import torch.nn as nn


def get_prf(data, *args, **kwargs):
    if data['TP'] == 0:
        if data['FP'] == 0 and data['FN'] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = (1.0 * data['TP'] / (data['TP'] + data['FP']))
        recall = (1.0 * data['TP'] / (data['TP'] + data['FN']))
        f1 = (2 * precision * recall / (precision + recall))

    return precision, recall, f1


def get_micro_macro_prf(data, *args, **kwargs):
    precision = []
    recall = []
    f1 = []

    total = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    for one_data in range(len(data)):
        total['TP'] += data[one_data]['TP']
        total['FP'] += data[one_data]['FP']
        total['FN'] += data[one_data]['FN']
        total['TN'] += data[one_data]['TN']

        p, r, f = get_prf(data=data[one_data])

        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(data=total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0

    for index in range(len(f1)):
        macro_precision += precision[index]
        macro_recall += recall[index]
        macro_f1 += f1[index]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return {
        'mip': round(micro_precision, 3)
        , 'mir': round(micro_recall, 3)
        , 'mif': round(micro_f1, 3)
        , 'map': round(macro_precision, 3)
        , 'mar': round(macro_recall, 3)
        , 'maf': round(macro_f1, 3)
    }


class ConfusionMatrix(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConfusionMatrix, self).__init__()

        self.softmax = nn.Softmax(dim=2)


    # Size of `preds` = [batch_size, task_number, 2].
    # Size of `labels` = [batch_size, task_number].
    def forward(self, preds, labels, cm_result, *args, **kwargs):
        preds = self.softmax(preds)

        # Get the value in index 1 of dim 2 in `preds`.
        # Size of `preds[:, :, 1]` = [batch_size, task_number].
        preds = preds[:, :, 1]

        # According to \
        # https://blog.csdn.net/DreamHome_S/article/details/85259533
        # , using `.detach()` to instead of `.data`.
        preds = preds.detach()
        labels = labels.detach()

        if cm_result == None:
            cm_result = []

        task_number = preds.size(1)

        while len(cm_result) < task_number:
            cm_result.append({'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0})

        for index in range(task_number):
            # If value >= 0.5, set value = 1 else 0.
            part_outputs = (preds[:, index] >= 0.5).long()
            part_labels = labels[:, index]

            # `tensor1 * tensor2` equals to `torch.mul(tensor1, tensor2)`.
            # `tensor.sum()` equals to `torch.sum(tensor)`.
            # `1 - tensor` will change 0 to 1 or 1 to 0.
            cm_result[index]['TP'] += int((part_labels * part_outputs).sum())
            cm_result[index]['FN'] += int(
                (part_labels * (1 - part_outputs)).sum())
            cm_result[index]['FP'] += int(
                ((1 - part_labels) * part_outputs).sum())
            cm_result[index]['TN'] += int(
                ((1 - part_labels) * (1 - part_outputs)).sum())

        return cm_result