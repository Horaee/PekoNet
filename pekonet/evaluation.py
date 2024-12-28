import torch.nn as nn


def get_prf(cm_result):
    if cm_result['TP'] == 0:
        if cm_result['FP'] == 0 and cm_result['FN'] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = \
            (1.0 * cm_result['TP'] / (cm_result['TP'] + cm_result['FP']))
        recall = (1.0 * cm_result['TP'] / (cm_result['TP'] + cm_result['FN']))
        f1 = (2 * precision * recall / (precision + recall))

    return precision, recall, f1


def get_micro_macro_prf(cm_results):
    precisions = []
    recalls = []
    f1s = []

    total = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    for index in range(len(cm_results)):
        total['TP'] += cm_results[index]['TP']
        total['FP'] += cm_results[index]['FP']
        total['FN'] += cm_results[index]['FN']
        total['TN'] += cm_results[index]['TN']

        precision, recall, f1 = get_prf(cm_result=cm_results[index])

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    micro_precision, micro_recall, micro_f1 = get_prf(cm_result=total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0

    for index in range(len(cm_results)):
        macro_precision += precisions[index]
        macro_recall += recalls[index]
        macro_f1 += f1s[index]

    macro_precision /= len(precisions)
    macro_recall /= len(recalls)
    macro_f1 /= len(f1s)

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


    # Size of predictions = [batch_size, class_number, 2].
    # Size of labels = [batch_size, class_number].
    def forward(self, predictions, labels, cm_results, *args, **kwargs):
        predictions = self.softmax(predictions)

        # Get the value in index 1 of dim 2 in predictions.
        # Size of predictions[:, :, 1] = [batch_size, class_number].
        predictions = predictions[:, :, 1]

        # According to 
        # https://blog.csdn.net/DreamHome_S/article/details/85259533
        # , using '.detach()' to instead of '.data'.
        predictions = predictions.detach()
        labels = labels.detach()

        if cm_results == None:
            cm_results = []

        class_number = predictions.size(1)

        while len(cm_results) < class_number:
            cm_results.append({'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0})

        for index in range(class_number):
            # If value >= 0.5, set value = 1 else 0.
            part_predictions = (predictions[:, index] >= 0.5).long()
            part_labels = labels[:, index]

            # 'tensor1 * tensor2' equals to 'torch.mul(tensor1, tensor2)'.
            # 'tensor.sum()' equals to 'torch.sum(tensor)'.
            # '1 - tensor' will change 0 to 1 or 1 to 0.
            cm_results[index]['TP'] += int(
                (part_labels * part_predictions).sum())
            cm_results[index]['FN'] += int(
                (part_labels * (1 - part_predictions)).sum())
            cm_results[index]['FP'] += int(
                ((1 - part_labels) * part_predictions).sum())
            cm_results[index]['TN'] += int(
                ((1 - part_labels) * (1 - part_predictions)).sum())

        return cm_results