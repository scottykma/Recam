# DataManager

Splits "encoding" and "training" into two phases and caches the encoded
results.

## What it does

- Calls `encode_batch()` to pre-encode a batch
- Caches encoded batches in a queue
- Moves all but the last cached batch to CPU
- Moves batches back to the target device on demand during training
- With CP enabled, only one rank encodes a batch and then broadcasts it to the
  rest of the CP group

## When it refills

- `get_batch()` refills the cache when it runs dry
- With CP enabled, one refill is reused by the whole CP group before the next
  refill

## The two phase callbacks

- `enter_encode()`
  - Moves the VAE / text encoder back to the GPU
- `enter_train()`
  - Moves the encoder modules to CPU and frees GPU memory

## How to tune it

- `pre_encode_batches`
  - Cache depth only; it is not the CP size
  - Larger: fewer phase switches and steadier throughput
  - Smaller: lower memory footprint, more frequent switches
- `pre_encode_log_every`
  - Controls refill progress logging

## What belongs in encode

- VAE encoding
- Text encoding
- Condition assembly
- Preprocessing that does not participate in backpropagation
