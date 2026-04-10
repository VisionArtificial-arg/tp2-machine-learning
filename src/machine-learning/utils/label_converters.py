def label_to_int(string_label):
    if string_label == 'diamond': return 1
    if string_label == 'trebol': return 2
    if string_label == 'heart':
        return 3

    else:
        raise Exception('unknown class_label')


def int_to_label(string_label):
    if string_label == 1: return 'diamond'
    if string_label == 2: return 'trebol'
    if string_label == 3:
        return 'heart'
    else:
        raise Exception('unknown class_label')
