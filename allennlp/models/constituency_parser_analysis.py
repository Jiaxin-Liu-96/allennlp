from typing import Dict, Tuple, List, Optional, NamedTuple, Any
from overrides import overrides

import torch
from torch.nn.modules.linear import Linear
from nltk import Tree

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import TimeDistributed, TextFieldEmbedder, Elmo, ScalarMix
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import last_dim_softmax, get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.metrics import EvalbBracketingScorer



class SpanInformation(NamedTuple):
    """
    A helper namedtuple for handling decoding information.

    Parameters
    ----------
    start : ``int``
        The start index of the span.
    end : ``int``
        The exclusive end index of the span.
    no_label_prob : ``float``
        The probability of this span being assigned the ``NO-LABEL`` label.
    label_prob : ``float``
        The probability of the most likely label.
    """
    start: int
    end: int
    label_prob: float
    no_label_prob: float
    label_index: int


@Model.register("constituency_parser_analysis")
class SpanConstituencyParserAnalysis(Model):
    """
    This ``SpanConstituencyParser`` simply encodes a sequence of text
    with a stacked ``Seq2SeqEncoder``, extracts span representations using a
    ``SpanExtractor``, and then predicts a label for each span in the sequence.
    These labels are non-terminal nodes in a constituency parse tree, which we then
    greedily reconstruct.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    span_extractor : ``SpanExtractor``, required.
        The method used to extract the spans from the encoded sequence.
    encoder : ``Seq2SeqEncoder``, required.
        The encoder that we will use in between embedding tokens and
        generating span representations.
    feedforward_layer : ``FeedForward``, required.
        The FeedForward layer that we will use in between the encoder and the linear
        projection to a distribution over span labels.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 span_extractor: SpanExtractor,
                 num_elmo_layers: int,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 evalb_directory_path: str = None) -> None:
        super(SpanConstituencyParserAnalysis, self).__init__(vocab, regularizer)

        self.span_extractor = span_extractor
        self.num_classes = self.vocab.get_vocab_size("labels")
        output_dim = span_extractor.get_output_dim()

        self._num_elmo_layers = num_elmo_layers
        self.tag_projection_layers = []
        for i in range(self._num_elmo_layers + 1):
            projection = TimeDistributed(Linear(output_dim, self.num_classes))
            self.add_module(f"linear_{i}", projection)
            self.tag_projection_layers.append(projection)

        representation_dim = 1024
        check_dimensions_match(representation_dim,
                               span_extractor.get_input_dim(),
                               "encoder input dim",
                               "span extractor input dim")


        self.scalar_mix = ScalarMix(num_elmo_layers)
        id_to_labels = {index: label for index, label in
                        self.vocab.get_index_to_token_vocabulary("labels").items() if "-" not in label}

        self.label_f1 = {}
        for i in range(num_elmo_layers):
            metrics = {label: F1Measure(index) for index, label
                       in id_to_labels.items()}
            self.label_f1[f"layer_{i}"] = metrics

        self.label_f1["mixed"] = {label: F1Measure(index) for index, label in id_to_labels.items()}

        self.tag_accuracies = {f"layer_{i}": CategoricalAccuracy() for i in range(num_elmo_layers)}
        self.tag_accuracies["mixed"] = CategoricalAccuracy()

        if evalb_directory_path is not None:
            self._evalb_scorers = {f"layer_{i}":EvalbBracketingScorer(evalb_directory_path)
                                   for i in range(num_elmo_layers)}
            self._evalb_scorers["mixed"] = EvalbBracketingScorer(evalb_directory_path)
        else:
            self._evalb_scorers = None
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.FloatTensor],
                lm_embeddings: torch.FloatTensor,
                spans: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                span_labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        spans : ``torch.LongTensor``, required.
            A tensor of shape ``(batch_size, num_spans, 2)`` representing the
            inclusive start and end indices of all possible spans in the sentence.
        metadata : List[Dict[str, Any]], required.
            A dictionary of metadata for each batch element which has keys:
                tokens : ``List[str]``, required.
                    The original string tokens in the sentence.
                gold_tree : ``nltk.Tree``, optional (default = None)
                    Gold NLTK trees for use in evaluation.
                pos_tags : ``List[str]``, optional.
                    The POS tags for the sentence. These can be used in the
                    model as embedded features, but they are passed here
                    in addition for use in constructing the tree.
        pos_tags : ``torch.LongTensor``, optional (default = None)
            The output of a ``SequenceLabelField`` containing POS tags.
        span_labels : ``torch.LongTensor``, optional (default = None)
            A torch tensor representing the integer gold class labels for all possible
            spans, of shape ``(batch_size, num_spans)``.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing a distribution over the label classes per span.
        spans : ``torch.LongTensor``
            The original spans tensor.
        tokens : ``List[List[str]]``, required.
            A list of tokens in the sentence for each element in the batch.
        pos_tags : ``List[List[str]]``, required.
            A list of POS tags in the sentence for each element in the batch.
        num_spans : ``torch.LongTensor``, required.
            A tensor of shape (batch_size), representing the lengths of non-padded spans
            in ``enumerated_spans``.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        all_elmo_layers = [x.squeeze(1) for x in lm_embeddings.split(1, dim=1)]
        # This is the mixed layer.
        all_elmo_layers.append(self.scalar_mix(all_elmo_layers))

        mask = get_text_field_mask(tokens)
        # Looking at the span start index is enough to know if
        # this is padding or not. Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).long()
        if span_mask.dim() == 1:
            # This happens if you use batch_size 1 and encounter
            # a length 1 sentence in PTB, which do exist. -.-
            span_mask = span_mask.unsqueeze(-1)
        if span_labels is not None and span_labels.dim() == 1:
            span_labels = span_labels.unsqueeze(-1)

        num_spans = get_lengths_from_binary_sequence_mask(span_mask)

        span_representations = [self.span_extractor(text, spans, mask, span_mask)
                                for text in all_elmo_layers]
        per_layer_logits = []
        per_layer_softmax = []
        for representations, linear in zip(span_representations, self.tag_projection_layers):
            logits = linear(representations)
            per_layer_logits.append(logits)
            per_layer_softmax.append(last_dim_softmax(logits, span_mask.unsqueeze(-1)))

        output_dict = {
                "class_probabilities": per_layer_softmax[-1],
                "spans": spans,
                "tokens": [meta["tokens"] for meta in metadata],
                "pos_tags": [meta.get("pos_tags") for meta in metadata],
                "num_spans": num_spans
        }
        if span_labels is not None:
            loss = 0.0
            for logits, probs, metric in zip(per_layer_logits, per_layer_softmax, self.tag_accuracies.values()):
                this_layer_loss = sequence_cross_entropy_with_logits(logits, span_labels, span_mask)
                loss = loss + this_layer_loss
                metric(probs, span_labels, span_mask)
            output_dict["loss"] = loss

        # The evalb score is expensive to compute, so we only compute
        # it for the validation and test sets.
        batch_gold_trees = [meta.get("gold_tree") for meta in metadata]
        if all(batch_gold_trees) and self._evalb_scorers is not None and not self.training:
            gold_pos_tags: List[List[str]] = [list(zip(*tree.pos()))[1]
                                              for tree in batch_gold_trees]

            for probs, evalb, f1_metrics in zip(per_layer_softmax, self._evalb_scorers.values(), self.label_f1.values()):
                predicted_trees = self.construct_trees(probs.cpu().data,
                                                       spans.cpu().data,
                                                       num_spans.data,
                                                       output_dict["tokens"],
                                                       gold_pos_tags)
                evalb(predicted_trees, batch_gold_trees)

                for f1_metric in f1_metrics.values():
                    f1_metric(probs, span_labels, span_mask)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Constructs an NLTK ``Tree`` given the scored spans. We also switch to exclusive
        span ends when constructing the tree representation, because it makes indexing
        into lists cleaner for ranges of text, rather than individual indices.

        Finally, for batch prediction, we will have padded spans and class probabilities.
        In order to make this less confusing, we remove all the padded spans and
        distributions from ``spans`` and ``class_probabilities`` respectively.
        """
        all_predictions = output_dict['class_probabilities'].cpu().data
        all_spans = output_dict["spans"].cpu().data

        all_sentences = output_dict["tokens"]
        all_pos_tags = output_dict["pos_tags"] if all(output_dict["pos_tags"]) else None
        num_spans = output_dict["num_spans"].data
        trees = self.construct_trees(all_predictions, all_spans, num_spans, all_sentences, all_pos_tags)

        batch_size = all_predictions.size(0)
        output_dict["spans"] = [all_spans[i, :num_spans[i]] for i in range(batch_size)]
        output_dict["class_probabilities"] = [all_predictions[i, :num_spans[i], :] for i in range(batch_size)]

        output_dict["trees"] = trees
        return output_dict

    def construct_trees(self,
                        predictions: torch.FloatTensor,
                        all_spans: torch.LongTensor,
                        num_spans: torch.LongTensor,
                        sentences: List[List[str]],
                        pos_tags: List[List[str]] = None) -> List[Tree]:
        """
        Construct ``nltk.Tree``'s for each batch element by greedily nesting spans.
        The trees use exclusive end indices, which contrasts with how spans are
        represented in the rest of the model.

        Parameters
        ----------
        predictions : ``torch.FloatTensor``, required.
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing a distribution over the label classes per span.
        all_spans : ``torch.LongTensor``, required.
            A tensor of shape (batch_size, num_spans, 2), representing the span
            indices we scored.
        num_spans : ``torch.LongTensor``, required.
            A tensor of shape (batch_size), representing the lengths of non-padded spans
            in ``enumerated_spans``.
        sentences : ``List[List[str]]``, required.
            A list of tokens in the sentence for each element in the batch.
        pos_tags : ``List[List[str]]``, optional (default = None).
            A list of POS tags for each word in the sentence for each element
            in the batch.

        Returns
        -------
        A ``List[Tree]`` containing the decoded trees for each element in the batch.
        """
        # Switch to using exclusive end spans.
        exclusive_end_spans = all_spans.clone()
        exclusive_end_spans[:, :, -1] += 1
        no_label_id = self.vocab.get_token_index("NO-LABEL", "labels")

        trees: List[Tree] = []
        for batch_index, (scored_spans, spans, sentence) in enumerate(zip(predictions,
                                                                          exclusive_end_spans,
                                                                          sentences)):
            selected_spans = []
            for prediction, span in zip(scored_spans[:num_spans[batch_index]],
                                        spans[:num_spans[batch_index]]):
                start, end = span
                no_label_prob = prediction[no_label_id]
                label_prob, label_index = torch.max(prediction, -1)

                # Does the span have a label != NO-LABEL or is it the root node?
                # If so, include it in the spans that we consider.
                if int(label_index) != no_label_id or (start == 0 and end == len(sentence)):
                    # TODO(Mark): Remove this once pylint sorts out named tuples.
                    # https://github.com/PyCQA/pylint/issues/1418
                    selected_spans.append(SpanInformation(start=int(start), # pylint: disable=no-value-for-parameter
                                                          end=int(end),
                                                          label_prob=float(label_prob),
                                                          no_label_prob=float(no_label_prob),
                                                          label_index=int(label_index)))

            # The spans we've selected might overlap, which causes problems when we try
            # to construct the tree as they won't nest properly.
            consistent_spans = self.resolve_overlap_conflicts_greedily(selected_spans)

            spans_to_labels = {(span.start, span.end):
                               self.vocab.get_token_from_index(span.label_index, "labels")
                               for span in consistent_spans}
            sentence_pos = pos_tags[batch_index] if pos_tags is not None else None
            trees.append(self.construct_tree_from_spans(spans_to_labels, sentence, sentence_pos))

        return trees

    @staticmethod
    def resolve_overlap_conflicts_greedily(spans: List[SpanInformation]) -> List[SpanInformation]:
        """
        Given a set of spans, removes spans which overlap by evaluating the difference
        in probability between one being labeled and the other explicitly having no label
        and vice-versa. The worst case time complexity of this method is ``O(k * n^4)`` where ``n``
        is the length of the sentence that the spans were enumerated from (and therefore
        ``k * m^2`` complexity with respect to the number of spans ``m``) and ``k`` is the
        number of conflicts. However, in practice, there are very few conflicts. Hopefully.

        This function modifies ``spans`` to remove overlapping spans.

        Parameters
        ----------
        spans: ``List[SpanInformation]``, required.
            A list of spans, where each span is a ``namedtuple`` containing the
            following attributes:

        start : ``int``
            The start index of the span.
        end : ``int``
            The exclusive end index of the span.
        no_label_prob : ``float``
            The probability of this span being assigned the ``NO-LABEL`` label.
        label_prob : ``float``
            The probability of the most likely label.

        Returns
        -------
        A modified list of ``spans``, with the conflicts resolved by considering local
        differences between pairs of spans and removing one of the two spans.
        """
        conflicts_exist = True
        while conflicts_exist:
            conflicts_exist = False
            for span1_index, span1 in enumerate(spans):
                for span2_index, span2 in list(enumerate(spans))[span1_index + 1:]:
                    if (span1.start < span2.start < span1.end < span2.end or
                                span2.start < span1.start < span2.end < span1.end):
                        # The spans overlap.
                        conflicts_exist = True
                        # What's the more likely situation: that span2 was labeled
                        # and span1 was unlabled, or that span1 was labeled and span2
                        # was unlabled? In the first case, we delete span2 from the
                        # set of spans to form the tree - in the second case, we delete
                        # span1.
                        if (span1.no_label_prob + span2.label_prob <
                                    span2.no_label_prob + span1.label_prob):
                            spans.pop(span2_index)
                        else:
                            spans.pop(span1_index)
                        break
        return spans

    @staticmethod
    def construct_tree_from_spans(spans_to_labels: Dict[Tuple[int, int], str],
                                  sentence: List[str],
                                  pos_tags: List[str] = None) -> Tree:
        """
        Parameters
        ----------
        spans_to_labels : ``Dict[Tuple[int, int], str]``, required.
            A mapping from spans to constituency labels.
        sentence : ``List[str]``, required.
            A list of tokens forming the sentence to be parsed.
        pos_tags : ``List[str]``, optional (default = None)
            A list of the pos tags for the words in the sentence, if they
            were either predicted or taken as input to the model.

        Returns
        -------
        An ``nltk.Tree`` constructed from the labelled spans.
        """
        def assemble_subtree(start: int, end: int):
            if (start, end) in spans_to_labels:
                # Some labels contain nested spans, e.g S-VP.
                # We actually want to create (S (VP ...)) nodes
                # for these labels, so we split them up here.
                labels: List[str] = spans_to_labels[(start, end)].split("-")
            else:
                labels = None

            # This node is a leaf.
            if end - start == 1:
                word = sentence[start]
                pos_tag = pos_tags[start] if pos_tags is not None else "XX"
                tree = Tree(pos_tag, [word])
                if labels is not None and pos_tags is not None:
                    # If POS tags were passed explicitly,
                    # they are added as pre-terminal nodes.
                    while labels:
                        tree = Tree(labels.pop(), [tree])
                elif labels is not None:
                    # Otherwise, we didn't want POS tags
                    # at all.
                    tree = Tree(labels.pop(), [word])
                    while labels:
                        tree = Tree(labels.pop(), [tree])
                return [tree]

            argmax_split = start + 1
            # Find the next largest subspan such that
            # the left hand side is a constituent.
            for split in range(end - 1, start, -1):
                if (start, split) in spans_to_labels:
                    argmax_split = split
                    break

            left_trees = assemble_subtree(start, argmax_split)
            right_trees = assemble_subtree(argmax_split, end)
            children = left_trees + right_trees
            if labels is not None:
                while labels:
                    children = [Tree(labels.pop(), children)]
            return children

        tree = assemble_subtree(0, len(sentence))
        return tree[0]

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}

        for name, metric in self.tag_accuracies.items():
            all_metrics[name + "_accuracy"] = metric.get_metric(reset=reset)

        for layer, metric_dict in self.label_f1.items():
            for label, metric in metric_dict.items():
                base_name = f"{layer}_{label}"
                f1, precision, recall = metric.get_metric(reset)
                all_metrics[base_name + "_" + "f1"] = f1
                all_metrics[base_name + "_" + "recall"] = recall
                all_metrics[base_name + "_" + "precision"] = precision

        if self._evalb_scorers is not None:
            for name, metric in self._evalb_scorers.items():
                evalb_metrics = metric.get_metric(reset=reset)
                for metric_name, evalb_metric in evalb_metrics.items():
                    all_metrics[name + "_" + metric_name] = evalb_metric

        return all_metrics

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SpanConstituencyParserAnalysis':
        span_extractor = SpanExtractor.from_params(params.pop("span_extractor"))

        num_elmo_layers = params.pop_int("num_elmo_layers")
        elmo_weights = params.pop("elmo_weights")
        elmo_options = params.pop("elmo_options")
        elmo_type = params.pop("elmo_type")
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        evalb_directory_path = params.pop("evalb_directory_path", None)
        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   span_extractor=span_extractor,
                   num_elmo_layers=num_elmo_layers,
                   elmo_options=elmo_options,
                   elmo_weights=elmo_weights,
                   elmo_type=elmo_type,
                   initializer=initializer,
                   regularizer=regularizer,
                   evalb_directory_path=evalb_directory_path)
