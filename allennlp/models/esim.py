from typing import Dict, Optional, List, Any

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("esim")
class ESIM(Model):
    """
    This ``Model`` implements the ESIM sequence model described in `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
    by Chen et al., 2017.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    encoder : ``Seq2SeqEncoder``
        Used to encode the premise and hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between encoded
        words in the premise and words in the hypothesis.
    projection_feedforward : ``FeedForward``
        The feedforward network used to project down the encoded and enhanced premise and hypothesis.
    inference_encoder : ``Seq2SeqEncoder``
        Used to encode the projected premise and hypothesis for prediction.
    output_feedforward : ``FeedForward``
        Used to prepare the concatenated premise and hypothesis for prediction.
    output_logit : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 dropout: float = 0.5,
                 embed_dropout: float = None,
                 compute_f1: bool = False,
                 attend_text_field: bool = False,
                 is_symmetric: bool = False,
                 is_regression: bool = False,
                 cached_cls: TokenEmbedder = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder

        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward

        self._inference_encoder = inference_encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
            if embed_dropout is not None:
                self.embed_dropout = InputVariationalDropout(embed_dropout)
            else:
                self.embed_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None

        self._attend_text_field = attend_text_field
        self._is_symmetric = is_symmetric
        self._is_regression = is_regression

        self._output_feedforward = output_feedforward
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        if not self._attend_text_field:
            check_dimensions_match(encoder.get_output_dim() * 4, projection_feedforward.get_input_dim(),
                               "encoder output dim", "projection feedforward input")
        check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
                               "proj feedforward output dim", "inference lstm input dim")

        self._compute_f1 = compute_f1

        if self._is_regression:
            from jiant.allennlp_mods.correlation import Correlation
            self._metrics = {'pearson': Correlation("pearson"),
                             "spearman": Correlation("spearman")}
            self._loss = torch.nn.MSELoss()
        else:
            self._accuracy = CategoricalAccuracy()
            if compute_f1:
                # DANGER :make sure 1 is the first label in the dataset
                # so it has index 0...
                self._f1 = F1Measure(0)
            self._loss = torch.nn.CrossEntropyLoss()

        self.cached_cls = cached_cls

        initializer(self)

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        # apply dropout for LSTM
        if self.dropout:
            embedded_premise = self.embed_dropout(embedded_premise)
            embedded_hypothesis = self.embed_dropout(embedded_hypothesis)

        # encode premise and hypothesis
        encoded_premise = self._encoder(embedded_premise, premise_mask)
        encoded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(encoded_premise, encoded_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(encoded_premise, h2p_attention)

        # the "enhancement" layer
        premise_enhanced = torch.cat(
                [encoded_premise, attended_hypothesis,
                 encoded_premise - attended_hypothesis,
                 encoded_premise * attended_hypothesis],
                dim=-1
        )
        hypothesis_enhanced = torch.cat(
                [encoded_hypothesis, attended_premise,
                 encoded_hypothesis - attended_premise,
                 encoded_hypothesis * attended_premise],
                dim=-1
        )

        # add in the attention of the embedded text field if needed
        if self._attend_text_field:
            similarity_matrix_embed = self._matrix_attention(embedded_premise, embedded_hypothesis)
            p2h_attention_embed = masked_softmax(similarity_matrix_embed, hypothesis_mask)
            attended_hypothesis_embed = weighted_sum(embedded_hypothesis, p2h_attention_embed)
            h2p_attention_embed = masked_softmax(similarity_matrix_embed.transpose(1, 2).contiguous(), premise_mask)
            attended_premise_embed = weighted_sum(embedded_premise, h2p_attention_embed)
            premise_enhanced_embed = torch.cat(
                    [embedded_premise, attended_hypothesis_embed,
                     embedded_premise - attended_hypothesis_embed,
                     embedded_premise * attended_hypothesis_embed],
                    dim=-1
            )
            hypothesis_enhanced_embed = torch.cat(
                    [embedded_hypothesis, attended_premise_embed,
                     embedded_hypothesis - attended_premise_embed,
                     embedded_hypothesis * attended_premise_embed],
                    dim=-1
            )
            premise_enhanced = torch.cat([premise_enhanced, premise_enhanced_embed], dim=-1)
            hypothesis_enhanced = torch.cat([hypothesis_enhanced, hypothesis_enhanced_embed], dim=-1)

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_enhanced_premise = self._projection_feedforward(premise_enhanced)
        projected_enhanced_hypothesis = self._projection_feedforward(hypothesis_enhanced)

        # Run the inference layer
        if self.dropout:
            projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
            projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)
        v_ai = self._inference_encoder(projected_enhanced_premise, premise_mask)
        v_bi = self._inference_encoder(projected_enhanced_hypothesis, hypothesis_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_a_max, _ = replace_masked_values(
                v_ai, premise_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        v_b_max, _ = replace_masked_values(
                v_bi, hypothesis_mask.unsqueeze(-1), -1e7
        ).max(dim=1)

        v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1) / torch.sum(
                premise_mask, 1, keepdim=True
        )
        v_b_avg = torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1) / torch.sum(
                hypothesis_mask, 1, keepdim=True
        )

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        output_hidden = self._output_feedforward(v_all)

        # add in the cls if needed
        if self.cached_cls:
            # (batch_size, embed_dim)
            cls_embeddings = self.cached_cls(premise['bert'])
            output_hidden = torch.cat([output_hidden, cls_embeddings], dim=1)

        label_logits = self._output_logit(output_hidden)

        if self._is_symmetric:
            # swap a and b and re-compute
            v_all = torch.cat([v_b_avg, v_b_max, v_a_avg, v_a_max], dim=1)
            if self.dropout:
                v_all = self.dropout(v_all)
                output_hidden = self._output_feedforward(v_all)
                label_logits = 0.5 * (label_logits + self._output_logit(output_hidden))

        output_dict = {"label_logits": label_logits}

        if not self._is_regression:
            label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
            output_dict["label_probs"] = label_probs
            if label is not None:
                loss = self._loss(label_logits, label.long().view(-1))
                self._accuracy(label_logits, label)
                if self._compute_f1:
                    self._f1(label_logits, label)
                output_dict["loss"] = loss
        else:
            # label_logits is the output variable for regression
            if label is not None:
                loss = self._loss(label_logits.view(-1), label.view(-1))
                detached_logits = label_logits.view(-1).detach()
                detached_label = label.view(-1).detach()
                for m in self._metrics.values():
                    m(detached_logits, detached_label)
                output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if not self._is_regression:
            metrics = {'accuracy': self._accuracy.get_metric(reset)}
            if self._compute_f1:
                metrics['f1'] = self._f1.get_metric(reset)[2]
                metrics['accuracy_and_f1'] = 0.5 * (metrics['accuracy'] + metrics['f1'])
        else:
            metrics = {k: float(m.get_metric(reset)) for k, m in self._metrics.items()}
            metrics['avg_correlation'] = 0.5 * (metrics['spearman'] + metrics['pearson'])
        return metrics
