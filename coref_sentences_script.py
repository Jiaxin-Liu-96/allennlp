import sys
from allennlp.data.dataset_readers.dataset_utils.ontonotes import Ontonotes


def _normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def write_sentences(output_path, file_path):
    ontonotes_reader = Ontonotes()

    with open(output_path, "w+") as out_file:
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            flattened_sentences = [_normalize_word(word)
                                for sentence in sentences
                                for word in sentence]
            tokens = " ".join(flattened_sentences)
            out_file.write(tokens + "\n")

if __name__ == "__main__":
    write_sentences(sys.argv[1], sys.argv[2])