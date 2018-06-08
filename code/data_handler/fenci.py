import jieba


def fenci_jieba(sentence):
    seg_list = jieba.lcut(sentence)
    return seg_list


def fenci_single_word(sentence):
    seg_list = sorted(set("".join(sentence)))
    return seg_list


print(fenci_single_word("寒随穷律变，春逐鸟声开。"))
