# Layerwise retrieval pin ownership

`LMCacheEngine.retrieve_layer()` separates read references from lookup pins.

| Result source | Read ownership | Retrieval cleanup | Lookup cleanup |
| --- | --- | --- | --- |
| `LocalCPUBackend` cache hit | One reference; no additional pin | Release read reference only | `lookup_unpin(id)` releases that request's recorded pins |
| Disk-loaded staging buffer | Temporary reference and staging pin | Release read reference; release staging pin after the consumer's sync point | Releases the backend entry's separate lookup pin |

A positive `MemoryObj.is_pinned` does not identify the pin owner. For CPU
hits it may belong to the current lookup, another request, or a controller.
Unpinning on return from retrieval would steal that owner's pin and cause a
second unpin during request cleanup. Masked chunks retain their lookup pins
until the same request-scoped cleanup as retrieved chunks.

The vLLM layerwise adapter already calls `lookup_unpin(request.req_id)` from
`wait_for_save()`. Retrieval must not duplicate that responsibility. Cleanup
of temporary buffers from non-CPU backends remains unchanged.

Regression coverage lives in
`tests/inprocess/integration/vllm/test_retrieve_pin_ownership.py`: single and
shared lookups, independent controller pins, masked prefixes, repeated
request cleanup, and independent disk-read staging buffers. GPU copying is
mocked in these CPU tests; the engine, cache objects, lookup pin bookkeeping
and backend pin/ref operations are real.
