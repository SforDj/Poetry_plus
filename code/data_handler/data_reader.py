def poetry_sentence_reader(file):
    poetry_sentence_list = []
    with open(file, 'r', encoding='UTF') as r:
        while True:
            sentence = r.readline()
            if sentence == "":
                break
            poetry_sentence_list.append(sentence)
    return poetry_sentence_list

