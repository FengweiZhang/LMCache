# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
)


@pytest.mark.parametrize("request_count", [1, 2])
def test_completed_prefill_load_is_not_advanced_during_decode(
    request_count: int,
) -> None:
    """A decode forward after a full layerwise load must not re-read its iterator."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.num_layers = 3
    connector.current_layer = 0
    connector.layerwise_retrievers = [
        iter([None, None, torch.ones(4, dtype=torch.bool)])
        for _ in range(request_count)
    ]
    for layer in range(3):
        connector.wait_for_layer_load(f"layer{layer}")
    for _ in range(2):
        for layer in range(3):
            connector.wait_for_layer_load(f"layer{layer}")


def test_incomplete_prefill_load_still_raises() -> None:
    """Retiring completed loads must not mask an unexpectedly short iterator."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.num_layers = 3
    connector.current_layer = 0
    connector.layerwise_retrievers = [iter([None])]
    connector.wait_for_layer_load("layer0")
    with pytest.raises(StopIteration):
        connector.wait_for_layer_load("layer1")
