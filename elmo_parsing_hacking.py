
from typing import List, Dict, Tuple
import os
from collections import defaultdict

from nltk import Tree
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
import torch
import numpy
from matplotlib import pyplot
import seaborn

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.dataset_readers.penn_tree_bank import PennTreeBankConstituencySpanDatasetReader
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans

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


def get_pre_leaf_spans(tree: Tree, index: int, typed_spans: Dict[Tuple[int, int], str]) -> int:
    """
    Recursively construct the gold spans from an nltk ``Tree``.
    Labels are the constituents, and in the case of nested constituents
    with the same spans, labels are concatenated in parent-child order.
    For example, ``(S (NP (D the) (N man)))`` would have an ``S-NP`` label
    for the outer span, as it has both ``S`` and ``NP`` label.
    Spans are inclusive.

    Parameters
    ----------
    tree : ``Tree``, required.
        An NLTK parse tree to extract spans from.
    index : ``int``, required.
        The index of the current span in the sentence being considered.
    typed_spans : ``Dict[Tuple[int, int], str]``, required.
        A dictionary mapping spans to span labels.

    Returns
    -------
    typed_spans : ``Dict[Tuple[int, int], str]``.
        A dictionary mapping all subtree spans in the parse tree
        to their constituency labels. POS tags are ignored.
    """
    # NLTK leaves are strings.
    if isinstance(tree[0], str):
        # The "length" of a tree is defined by
        # NLTK as the number of children.
        # We don't actually want the spans for leaves, because
        # their labels are POS tags. Instead, we just add the length
        # of the word to the end index as we iterate through.
        end = index + len(tree)
    else:
        # otherwise, the tree has children.
        child_start = index
        for child in tree:
            # typed_spans is being updated inplace.
            end = get_pre_leaf_spans(child, child_start, typed_spans)
            child_start = end
        # Set the end index of the current span to
        # the last appended index - 1, as the span is inclusive.
        span = (index, end - 1)
        current_span_label = typed_spans.get(span)

        # We only want spans for which all of their children are leaves.
        if all([isinstance(child[0], str) for child in tree]):
            if current_span_label is None:
                # This span doesn't have nested labels, just
                # use the current node's label.
                typed_spans[span] = tree.label()
            else:
                # This span has already been added, so prepend
                # this label (as we are traversing the tree from
                # the bottom up).
                typed_spans[span] = tree.label() + "-" + current_span_label

    return end


def get_gold_spans(tree: Tree, index: int, typed_spans: Dict[Tuple[int, int], str]) -> int:
    """
    Recursively construct the gold spans from an nltk ``Tree``.
    Labels are the constituents, and in the case of nested constituents
    with the same spans, labels are concatenated in parent-child order.
    For example, ``(S (NP (D the) (N man)))`` would have an ``S-NP`` label
    for the outer span, as it has both ``S`` and ``NP`` label.
    Spans are inclusive.

    Parameters
    ----------
    tree : ``Tree``, required.
        An NLTK parse tree to extract spans from.
    index : ``int``, required.
        The index of the current span in the sentence being considered.
    typed_spans : ``Dict[Tuple[int, int], str]``, required.
        A dictionary mapping spans to span labels.

    Returns
    -------
    typed_spans : ``Dict[Tuple[int, int], str]``.
        A dictionary mapping all subtree spans in the parse tree
        to their constituency labels. POS tags are ignored.
    """
    # NLTK leaves are strings.
    if isinstance(tree[0], str):
        # The "length" of a tree is defined by
        # NLTK as the number of children.
        # We don't actually want the spans for leaves, because
        # their labels are POS tags. Instead, we just add the length
        # of the word to the end index as we iterate through.
        end = index + len(tree)
    else:
        # otherwise, the tree has children.
        child_start = index
        for child in tree:
            # typed_spans is being updated inplace.
            end = get_gold_spans(child, child_start, typed_spans)
            child_start = end
        # Set the end index of the current span to
        # the last appended index - 1, as the span is inclusive.
        span = (index, end - 1)
        current_span_label = typed_spans.get(span)
        if current_span_label is None:
            # This span doesn't have nested labels, just
            # use the current node's label.
            typed_spans[span] = tree.label()
        else:
            # This span has already been added, so prepend
            # this label (as we are traversing the tree from
            # the bottom up).
            typed_spans[span] = tree.label() + "-" + current_span_label

    return end

def get_embedded_sentence(sentences: List[List[str]], elmo, layer="mean", gpu=False):
    character_ids = batch_to_ids(sentences)
    if gpu:
        character_ids = character_ids.cuda()
    embeddings = elmo(character_ids)
    if layer == "mean":
        return embeddings["elmo_representations"][0].squeeze(0)
    else:
        return embeddings["layer_activations"][layer].squeeze(0)

def get_self_similarity(tensor):
    tensor = tensor / torch.norm(tensor, dim=1).unsqueeze(-1)
    similarity = torch.matmul(tensor, tensor.transpose(0,1))
    return similarity


def generate_word_similarity_heatmap(embedded_sentence):
    word_similarity = get_self_similarity(embedded_sentence).data.numpy()

    fig, ax = pyplot.subplots()
    # the size of A4 paper
    #fig.set_size_inches(18.7, 14.27)

    seaborn.heatmap(word_similarity,
                    linewidths=0.5,
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cbar=True,
                    vmin=0.0,
                    vmax=1.0)
    pyplot.show()

# TODO use average?
def get_outside_similarity(embedded_sentence, span):
    if span[1] - span[0] == embedded_sentence.size(0):
        return 0.0

    embedded_sentence = embedded_sentence / torch.norm(embedded_sentence, dim=1).unsqueeze(-1)
    embedded_span = embedded_sentence[slice(*span), :]

    if span[0] == 0:
        before = []
    else:
        before = [embedded_sentence[0: span[0], :]]
    if span[1] == embedded_sentence.size(0):
        after = []
    else:
        after = [embedded_sentence[span[1]:, :]]
    outside_span = torch.cat(before + after, 0)
    similarity = torch.matmul(outside_span, embedded_span.transpose(0, 1)).tril()
    return float(similarity.sum() / float((similarity != 0).sum()))

def get_inside_similarity(embedded_sentence, span):
    if span[1] - span[0] == 1:
        return 1.0

    embedded_sentence = embedded_sentence / torch.norm(embedded_sentence, dim=1).unsqueeze(-1)
    embedded_span = embedded_sentence[slice(*span), :]
    similarity = torch.matmul(embedded_span, embedded_span.transpose(0, 1))

    for i in range(similarity.size(0)):
        similarity[i, i] = 0.0
    similarity = similarity.tril()

    return float(similarity.sum() / float((similarity !=0).sum())) 


def get_elmo(elmo_type, options, weights, gpu=False):
    from calypso.train import load_encoder
    from calypso.token_embedders import ELMoWrapper
    if elmo_type == "lstm":
        elmo = Elmo(options, weights, num_output_representations=1, dropout=0.0)
    else:
        num_elmo_layers = 17 if elmo_type == "cnn" else 6
        module = ELMoWrapper(load_encoder(options, weights, -1), num_elmo_layers)
        elmo = Elmo(None, None, num_output_representations=1, dropout=0.0, module=module)

    if gpu:
        return elmo.cuda()
    return elmo

def data_generator(file_path, use_pre_leaf_spans=False, layer="mean"):
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    directory, filename = os.path.split(file_path)
    for parses in lazy_groups_of(BracketParseCorpusReader(root=directory, fileids=[filename]).parsed_sents(), 20):

        batch_sentences = []
        batch_constituents = []
        for parse in parses:
            strip_functional_tags(parse)
            # All the trees also contain a root S node.
            if parse.label() == "VROOT":
                parse = parse[0]

            tokens = parse.leaves()
            batch_sentences.append(tokens)
            constituents = {}
            if use_pre_leaf_spans:
                get_pre_leaf_spans(parse, 0, constituents)
            else:
                get_gold_spans(parse, 0, constituents)
            # Offset as allennlp uses internal indices
            constituents = {(start, end + 1): label for ((start, end), label) in constituents.items() if (end - start != 0 and end-start !=len(tokens) -1)}
            batch_constituents.append(constituents)

        embedded_sentences = get_embedded_sentence(batch_sentences, elmo, layer=layer)

        for i, (sentence, constituents) in enumerate(zip(batch_sentences, batch_constituents)):
            yield sentence, embedded_sentences[i], constituents

        print("Processed 20 parses")


def compute_similarity_statistics(embedded_sentence,
                                  constituents, 
                                  similarity_stats = None):

    similarity_stats = similarity_stats or defaultdict(list)
    for span, label in constituents.items():
        inside = get_inside_similarity(embedded_sentence, span)
        outside = get_outside_similarity(embedded_sentence, span)

        similarity_stats["constituents_inside"].append(inside)
        similarity_stats["constituents_outside"].append(outside)

    return similarity_stats


def graph_constituent_distributions(layer, num_sentences=10000000, show=False):

    gen = data_generator("/Users/markn/allen_ai/data/ptb-wsj/wsj.dev.notrace.trees", layer=layer)
    stats = defaultdict(list)
    count = 0
    for (tokens, embedded_sentence, constituents) in gen:
        stats = compute_similarity_statistics(embedded_sentence, constituents, stats)

        count +=1
        if count == num_sentences:
            break
    fig, ax = pyplot.subplots()

    bins = [0.02 * i for i in range(50)]
    print(stats["constituents_inside"])
    seaborn.distplot(stats["constituents_inside"], bins=bins, kde=False, axlabel="similarity", label="Avg Inside Similarity")
    seaborn.distplot(stats["constituents_outside"], bins=bins, kde=False, axlabel="similarity", label="Avg Outside Similarity")

    ax.set(xlim=(0, 1.0))

    pyplot.legend()
    fig.savefig(f"distribution_layer_{layer}_full_dev.png")

    if show:
        pyplot.show()


if __name__ == "__main__":

    # elmo = Elmo(options_file, weight_file, 1, dropout=0)
    # tree = "(VROOT(S(NP-SBJ(DT That))(VP(MD could)(VP(VB cost)(NP(PRP him))(NP(DT the)(NN chance)(S(VP(TO to)(VP(VP(VB influence)(NP(DT the)(NN outcome)))(CC and)(VP(ADVP(RB perhaps))(VB join)(NP(DT the)(VBG winning)(NN bidder)))))))))(. .)))"
    # tree = Tree.fromstring(tree
    # print(tree)
    # constituents = {}
    # get_gold_spans(tree, 0, constituents)
    # print(constituents)
    # constituent_leaves = {}
    # get_pre_leaf_spans(tree, 0, constituent_leaves)
    # print(constituent_leaves)

    # tokens = tree.leaves()
    # embedded_sentence = get_embedded_sentence([tokens], elmo)

    graph_constituent_distributions("mean", 40)
    #graph_constituent_distributions(0 , 40)
    #graph_constituent_distributions(1, 200)
    #graph_constituent_distributions(2, 200)
