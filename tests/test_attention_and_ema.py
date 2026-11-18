from types import SimpleNamespace

import pytest
import torch

from yolo.core.modules.blocks import Attention
from yolo.engine.utils import copy_attr


@pytest.mark.parametrize("include", [(), ("labels", "unused", "skip", "_private")])
def test_copy_attr_filters_and_shares_metadata(include):
    source = SimpleNamespace(labels=["cat"], extra=7, skip=True, _private=True)
    target = SimpleNamespace(skip=False)
    copy_attr(target, source, include=include, exclude=("skip",))
    expected = {"skip": False, "labels": ["cat"]}
    if not include:
        expected["extra"] = 7
    assert vars(target) == expected
    assert target.labels is source.labels


def test_attention_respects_checkpoint_head_layout():
    torch.manual_seed(7)
    layer = Attention(12, num_heads=3).double().eval()
    image = torch.randn(2, 12, 3, 5, dtype=torch.double)
    # Compute each head independently from its contiguous checkpoint channels.
    heads = layer.qkv(image).flatten(2).chunk(3, dim=1)
    contexts, values = [], []
    for head in heads:
        query, key, value = head.split([2, 2, 4], dim=1)
        logits = torch.bmm(query.mT, key) / 2**0.5
        contexts.append(torch.bmm(logits.softmax(-1), value.mT).mT)
        values.append(value)
    context = torch.cat(contexts, dim=1).reshape_as(image)
    position = layer.pe(torch.cat(values, dim=1).reshape_as(image))
    torch.testing.assert_close(layer(image), layer.proj(context + position))
