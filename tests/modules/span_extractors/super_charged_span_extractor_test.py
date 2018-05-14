# pylint: disable=no-self-use,invalid-name,protected-access
import numpy
import torch
from torch.autograd import Variable

from allennlp.modules.span_extractors import SpanExtractor, SuperChargedSpanExtractor
from allennlp.common.params import Params
from allennlp.nn.util import batched_index_select

class TestEndpointSpanExtractor:
    def test_endpoint_span_extractor_can_build_from_params(self):
        params = Params({
                "type": "super_charged",
                "input_dim": 7,
                })
        extractor = SpanExtractor.from_params(params)
        assert isinstance(extractor, SuperChargedSpanExtractor)
        assert extractor.get_output_dim() == 4 * 7  

    def test_correct_sequence_elements_are_embedded(self):
        sequence_tensor = Variable(torch.randn([2, 5, 7]))
        # Concatentate start and end points together to form our representation.
        extractor = SuperChargedSpanExtractor(7)

        indices = Variable(torch.LongTensor([[[1, 3],
                                              [2, 4]],
                                             [[0, 2],
                                              [3, 4]]]))
        span_representations = extractor(sequence_tensor, indices)

        assert list(span_representations.size()) == [2, 2, 4 * 7]
