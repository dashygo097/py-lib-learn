import collections
import re

text_path = "./data/timemachine.txt"
def read_text(path):

    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_text(text_path)

def tokenizer(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('error!' + token)

tokens = tokenizer(lines)

