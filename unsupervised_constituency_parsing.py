

from typing import List
import os
from functools import lru_cache

from nltk import Tree
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
import torch
import numpy


from allennlp.training.metrics import EvalbBracketingScorer
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.common.util import lazy_groups_of

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

def strip_functional_tags(tree: Tree) -> None:
    """
    Removes all functional tags from constituency labels in an NLTK tree.
    We also strip off anything after a =, - or | character, because these
    are functional tags which we don't want to use.
    """
    clean_label = tree.label().split("=")[0].split("-")[0].split("|")[0]
    tree.set_label(clean_label)
    for child in tree:
        if not isinstance(child[0], str):
            strip_functional_tags(child)

def get_embedded_sentence(sentences: List[List[str]], elmo: Elmo, layer="mean"):
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    if layer == "mean":
        return embeddings["elmo_representations"][0].squeeze(0)
    else:
        return embeddings["layer_activations"][layer].squeeze(0)

@lru_cache(maxsize=128)
def dot_product(x: "Fragment", y: "Fragment"):
    return float(sum(x.vector * y.vector))


class Fragment:
    def __init__(self, rep: Tree, vector):
        self.rep = rep
        self.vector = vector

    def merge(self, other: "Fragment", merge_index, lift=True):
        if lift:
            rep = Tree(f"{merge_index}", [self.rep, other.rep])
        else:
            for subtree in other.rep:
                num_children = len(self.rep)
                self.rep.insert(num_children, subtree)
            rep = self.rep
        return Fragment(rep, (self.vector + other.vector) / 2)

    def __hash__(self):
        return id(str(self.rep))

    def __str__(self):
        return str(self.rep)


class ParseConstructor:
    def __init__(self, words: List[Fragment]):
        self.words = words

        self.similarities = []
        for i in range(len(words) -1):
            value = dot_product(self.words[i], self.words[i + 1])
            self.similarities.append(((i, i + 1), value))

    def merge(self, i: int, j:int, merge_index:int, lift=True):
        word1 = self.words[i]
        word2 = self.words[j]

        words_before = self.words[:i]
        words_after = self.words[j+1:]
        pre_length = len(self.words)
        self.words = words_before + [word1.merge(word2, merge_index, lift)] + words_after
        post_length = len(self.words)

        assert post_length == pre_length - 1
        self.update_similarities()

    def update_similarities(self):
        new_similarities = []
        for i in range(len(self.words) -1):
            value = dot_product(self.words[i], self.words[i + 1])
            new_similarities.append(((i, i + 1), value))

        self.similarities = new_similarities

def data_generator(file_path: str, layer="mean"):
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    directory, filename = os.path.split(file_path)
    parse_reader = BracketParseCorpusReader(root=directory,
                                            fileids=[filename]).parsed_sents()
    for parses in lazy_groups_of(parse_reader, 20):

        batch_sentences = []
        batch_pos = []
        gold_parses = []
        for parse in parses:
            strip_functional_tags(parse)
            # All the trees also contain a root S node.
            if parse.label() == "VROOT":
                parse = parse[0]

            gold_parses.append(parse)
            tokens = parse.leaves()
            pos = [x[1] for x in parse.pos()]
            batch_pos.append(pos)
            batch_sentences.append(tokens)

        embedded_sentences = get_embedded_sentence(batch_sentences, elmo, layer=layer)

        for i, (sentence, pos, gold_parse) in enumerate(zip(batch_sentences, batch_pos, gold_parses)):
            yield sentence, pos, embedded_sentences[i], gold_parse


def greedy_similarity_parse(words: List[str],
                            pos: List[str],
                            embedded_sentence: torch.Tensor):

    embedded_sentence = embedded_sentence / torch.norm(embedded_sentence, dim=1).unsqueeze(-1)
    sentence = [Fragment(Tree(tag, [word]), vector) 
                for word, vector, tag in zip(words, embedded_sentence, pos)]

    sentence = ParseConstructor(sentence)
    merge_index = 0
    while len(sentence.words) != 1:
        
        tuple_index = numpy.argmax([x[1] for x in sentence.similarities])
        tuple_to_merge = sentence.similarities[tuple_index][0]
        sentence.merge(*tuple_to_merge, merge_index)
        merge_index+=1

    predicted_tree = sentence.words[0].rep

    return predicted_tree



if __name__ == "__main__":

    gen = data_generator("tiny.trees")

    scorer = EvalbBracketingScorer("scripts/EVALB")

    batch_size = 3
    batch_predicted = []
    batch_gold = []
    index = 0
    for sentence, pos, embedding, gold_parse in gen:
        predicted = greedy_similarity_parse(sentence, pos, embedding)

        batch_predicted.append(predicted)
        batch_gold.append(gold_parse)
        index +=1

        if index % batch_size == 0:
            scorer(batch_predicted, batch_gold)
            print(scorer.get_metric())
            batch_predicted = []
            batch_gold = []
    
        print("===================================")
        print(predicted)
        print("===================================")
        print(gold_parse)

    print("Finished parsing - final score:")
    print(scorer.get_metric())