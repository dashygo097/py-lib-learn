import os
from datasets import DatasetDict, Dataset


def load_txt(path , flatten = False,pre =False, token="word"):
    with open(path, "r") as f:
        lines = f.readlines()

    data = []
    vocab_dir = {}
    index = 0
    for sentence_id, line in enumerate(lines):
        data.append(line)
        word_list = line.split(" ")
        for word in word_list:
            if word not in vocab_dir.keys():
                vocab_dir[word] = index
                index += 1
    if token == "word":
        if flatten :
            if pre:
                data = [vocab_dir[word] for line in data for word in line.split(" ")]
            else :
                data = [word for line in data for word in line.split(" ")]

    elif token == "char":
        vocab_dir_char = {}
        index = 0
        for word in vocab_dir.keys():
            for char in word :
                if char not in vocab_dir_char.keys():
                    vocab_dir_char[char] = index
                    index += 1
        if flatten:
            if pre :
                data = [vocab_dir_char[letter] for line in data for letter in line]
            else :
                data =[letter for line in data for letter in line]

        vocab_dir = vocab_dir_char

    return data, vocab_dir

def convert_from_txt(data_dir , to_path = "./data"):
    files = os.listdir(data_dir)
    datadir = {}
    for file_name in files :
        data_path = data_dir + "/" + file_name
        with open(data_path , "r") as f :
            lines = f.readlines()
            sentence_id = []
            sentence = []
            for id,line in enumerate(lines):
                sentence_id.append(id)
                sentence.append(line)
            dataset = Dataset.from_dict({"sentence_id":sentence_id , "sentence":sentence})
        if "train" in data_path:
            datadir["train"] = dataset
        if "test" in data_path:
            datadir["test"] = dataset
        if "valid" in data_path:
            datadir["validation"] = dataset
    datadir = DatasetDict(datadir)
    datadir.save_to_disk(to_path)

# Examples:
#
# load_from_txt("./ptb")
#
#


