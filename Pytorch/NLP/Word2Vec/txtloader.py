import os
from datasets import DatasetDict,Dataset,load_dataset


def load_txt(path):

    with open(path, "r") as f:
        lines = f.readlines()

    data = {}
    vocab_dir = {}
    for sentence_id,line in enumerate(lines) :
        data[sentence_id] = line

    return data

def load_from_txt(data_dir , to_path = "./data"):
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

