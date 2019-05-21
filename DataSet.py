from io import open
import re
import torch
import os


dir_path = os.path.dirname(os.path.dirname(__file__))
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1="en", lang2="cn", reverse=False):
    print("Read lines...")
    # Read the file and split into lines
    with open(os.path.join(dir_path, "data", "%s-%s.txt" % (lang1, lang2)), encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    # Split every line into pairs and normalize
    line = [l.split('\t') for l in lines]
    en_data = [normalizeString(word[0]) for word in line]
    cn_data = [" ".join(s for s in word[1]) for word in line]
    pairs = [[en_sentence, cn_sentence]for en_sentence, cn_sentence in zip(en_data, cn_data)]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def compute_max_length(pairs):
    length_list = []
    for p in pairs:
        length_zero = len(p[0].split(' '))
        length_one = len(p[1].split(' '))
        length_list.append(length_zero)
        length_list.append(length_one)
    max_length = max(length_list)
    return max_length


def prepareData(lang1="en", lang2="cn", reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    max_length = compute_max_length(pairs)
    print("Read {} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        if not reverse:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs, max_length


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence, device):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensor_from_pair(input_lang, output_lang, pair, device):
    input_tensor = tensor_from_sentence(input_lang, pair[0], device)
    target_tensor = tensor_from_sentence(output_lang, pair[1], device)
    return input_tensor, target_tensor

