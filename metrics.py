from collections import defaultdict
class Metrics:

    def __init__(self, targets):
        self.tags_name = targets
        tags = set()
        for t in targets:
            begin, mid = 'B-' + t, 'I-' + t
            tags.add((begin, mid))
        self.tags = list(tags)

    def _find_tag(self, labels, B_label=None, I_label=None):
        '''
        :param labels: list of string such as ['O', 'O', 'B-SUR']
        :param B_label: the begin label of the tags, such as B-PAR, B-SUR
        :param I_label: the medium label of the tags
        :return: a list of tuple whose first ele is position of the begin lable of the tags,
                    and the second one is the length of the tags
        '''
        if not labels or not B_label or not I_label:
            print('Check your labels')
            raise ValueError
        result = []
        tags_begin_pos, length = 0, 0
        for num in range(len(labels)):
            if labels[num] == B_label:
                tags_begin_pos = num
            if labels[num] == I_label and labels[num - 1] == B_label:
                length = 2
                for num2 in range(num, len(labels)):
                    if labels[num2] == I_label and labels[num2 - 1] == I_label:
                        length += 1
                    if labels[num2] != I_label:
                        result.append((tags_begin_pos, length))
                        break
        return result

    def find_all_tag(self, labels):
        result = {}
        for tag in self.tags:
            res = self._find_tag(labels, B_label=tag[0], I_label=tag[1])
            result[tag[0].split("-")[1]] = res
        return result

    def precision(self, pre_labels, true_labels):
        '''
        :param pre_tags: list
        :param true_tags: list
        :return:
        '''
        if isinstance(pre_labels, str):
            pre_labels = pre_labels.strip().split()
            pre_labels = [label for label in pre_labels]
        if isinstance(true_labels, str):
            true_labels = true_labels.strip().split()
            true_labels = [label for label in true_labels]

        pre_result = self.find_all_tag(pre_labels)
        pre_dic = {}
        for name in pre_result:
            pre = []
            for x in pre_result[name]:
                if x:
                    if pre_labels[x[0]:x[0] + x[1]] == true_labels[x[0]:x[0] + x[1]]:
                        pre.append(1)
                    else:
                        pre.append(0)
            # print(f'{name}: {pre}')
            pre_dic[name] = sum(pre) / len(pre) if len(pre) else 0
            # print(pre_dic)

        return pre_dic

    def recall(self, pre_labels, true_labels):
        '''
        :param pre_tags: list
        :param true_tags: list
        :return:
        '''
        if isinstance(pre_labels, str):
            pre_labels = pre_labels.strip().split()
            pre_labels = [label for label in pre_labels]
        if isinstance(true_labels, str):
            true_labels = true_labels.strip().split()
            true_labels = [label for label in true_labels]
        true_result = self.find_all_tag(true_labels)
        recall_dic = {}
        for name in true_result:
            recall = []
            for x in true_result[name]:
                if x:
                    if pre_labels[x[0]:x[0] + x[1]] == true_labels[x[0]:x[0] + x[1]]:
                        recall.append(1)
                    else:

                        recall.append(0)
            recall_dic[name] = sum(recall) / len(recall) if len(recall) else 0
        return recall_dic

    def f1_score(self, precision, recall):
        return (2 * precision * recall) / (precision + recall) if (precision+recall) != 0else 0

    def classification_report(self, pre_labels, true_labels):
        pre_dic = self.precision(pre_labels, true_labels)
        recall_dic = self.recall(pre_labels, true_labels)
        report = defaultdict(dict)
        for name in self.tags_name:
            report[name]['precision'] = pre_dic[name]
            report[name]['recall'] = recall_dic[name]
            f1 = self.f1_score(pre_dic[name], recall_dic[name])
            report[name]['f1_score'] = f1
        return report