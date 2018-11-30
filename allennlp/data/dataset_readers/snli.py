from typing import Dict, Union
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers import Token

import numpy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("snli")
class SnliReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_str: str = "gold_label",
                 label_type: str = "str",
                 max_length: int = None,
                 cached_pair: bool = False,
                 special_tokens: Dict[str, str] = None,
                 flatten_sequence: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._label_str = label_str
        self._label_type = label_type
        if max_length is None:
            max_length = int(1e9)
        self._max_length = max_length
        self._cached_pair = cached_pair
        assert label_type in ['str', 'float']

        self._special_tokens = special_tokens
        self._flatten_sequence = flatten_sequence

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as snli_file:
            logger.info("Reading SNLI instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)

                label = example[self._label_str]
                if label == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the training data.
                    continue
                if self._label_type == 'str':
                    label = str(label)
                elif self._label_type == 'float':
                    label = float(label)

                premise = example["sentence1"]
                hypothesis = example["sentence2"]
                premise_tokens = self._tokenizer.tokenize(premise)[:self._max_length]
                hypothesis_tokens = self._tokenizer.tokenize(hypothesis)[:self._max_length]


                if not self._flatten_sequence:
                    yield self.text_to_instance(premise_tokens, hypothesis_tokens, label)
                else:
                    # concate the premise, hypothesis with special tokens
                    all_tokens = [self._special_tokens['start']] + \
                                 [x.text for x in premise_tokens] + \
                                 [self._special_tokens['delimiter']] + \
                                 [x.text for x in hypothesis_tokens] + \
                                 [self._special_tokens['predict']]
                    yield self.text_to_instance(all_tokens, None, label)


    @overrides
    def text_to_instance(self,  # type: ignore
                         premise_tokens,
                         hypothesis_tokens,
                         label: Union[str, float] = None) -> Instance:
        # pylint: disable=arguments-differ
        if hypothesis_tokens is None:
            return self._text_to_instance_flat(premise_tokens, label)

        fields: Dict[str, Field] = {}
        if not self._cached_pair:
            fields['premise'] = TextField(premise_tokens, self._token_indexers)
            fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        else:
            # need to concatenate for indexing
            all_tokens = premise_tokens + [Token('|||')] + hypothesis_tokens
            fields['premise'] = TextField(
                    [Token('0')] + all_tokens, self._token_indexers
            )
            fields['hypothesis'] = TextField(
                    [Token('1')] + all_tokens, self._token_indexers
            )

        if label is not None and label != '':
            if self._label_type == 'str':
                fields['label'] = LabelField(label)
            elif self._label_type == 'float':
                fields['label'] = ArrayField(numpy.array([label]))

        metadata = {"premise_tokens": [x.text for x in premise_tokens],
                    "hypothesis_tokens": [x.text for x in hypothesis_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    def _text_to_instance_flat(self,  # type: ignore
                         all_tokens,
                         label: Union[str, float] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields['tokens'] = TextField(
                [Token(token) for token in all_tokens], self._token_indexers
        )
        if label is not None and label != '':
            if self._label_type == 'str':
                fields['label'] = LabelField(label)
            elif self._label_type == 'float':
                fields['label'] = ArrayField(numpy.array([label]))

        return Instance(fields)



