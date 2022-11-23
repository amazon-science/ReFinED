from collections import defaultdict
from typing import List, Dict

import torch
from torch.nn import DataParallel
from torch.nn.parallel._functions import Gather

from refined.data_types.base_types import Span
from refined.data_types.modelling_types import ModelReturn


# DataParallelReFinED is same as torch `DataParallel` (in torch==1.13.0),
# except we made a slight modification to allow Span return value from model forward pass.

def _is_namedtuple(obj):
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def gather(outputs, target_device, dim=0):
    r"""
    Taken from torch code with slight modification to allow Span return value from model forward pass.
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        # do not attempt to gather `Span` objects
        if out is None or isinstance(out, Span) or isinstance(out, defaultdict):
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs]))
                             for k in out)
        if _is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))

        return type(out)(map(gather_map, zip(*outputs)))

    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


class DataParallelReFinED(DataParallel):
    def gather(self, outputs, output_device):
        # Do not gather `Span`. Instead, just return the `Span`.
        # This allows the return type of RefinedModel to return a list of Span.
        all_entity_spans: List[Span] = []
        all_other_spans: Dict[str, List[Span]] = defaultdict(list)
        for output in outputs:
            output: ModelReturn
            if output.entity_spans is not None:
                all_entity_spans.extend(output.entity_spans)
            if output.other_spans is not None:
                for other_span_ner_type, other_spans in output.other_spans.items():
                    all_other_spans[other_span_ner_type].extend(other_spans)
        result: ModelReturn = gather(outputs, output_device, dim=self.dim)
        # note that these span lists may not be sorted
        return result._replace(entity_spans=all_entity_spans, other_spans=dict(all_other_spans))

    def forward(self, *inputs, **kwargs):
        batch = inputs[0] if len(inputs) > 0 else kwargs["batch"]
        # batch_elements_included keeps track of which GPU gets which batch element
        kwargs["batch_elements_included"] = torch.arange(batch.token_id_values.size(0)).unsqueeze(-1)
        return super().forward(*inputs, **kwargs)
