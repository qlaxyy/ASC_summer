# Steering Direction Protocol

## Evidence from reference implementations

The three reference papers and two implementations agree on a basic contract:

1. Define an oriented behavioral direction.
2. Read and modify the same residual-stream location.
3. Apply the desired direction by addition with a positive coefficient.

ActAdd (`2308.10248v5.pdf`) describes a target-minus-source vector such as
`Love - Hate`, then adds that vector. Its Hugging Face implementation reads the
input residual stream with a transformer-block pre-hook and injects at the same
pre-hook location.

The IBM CAST implementation (`2409.05907v3.pdf`) obtains a PCA direction from
positive/negative contrast pairs. Because PCA has arbitrary sign, it explicitly
checks the projections of the labeled examples and flips the component when
positive examples do not lie on the positive side. Its default control operator
is `current + control`, applied to block-input hidden states.

The released ASC code differs in two ways:

- it computes `short - long` but subtracts that vector during generation;
- it extracts `hidden_states[layer_index]` but injects after
  `model.layers[layer_index]`, which usually refers to adjacent residual-stream
  locations because `hidden_states[0]` is the embedding output.

Our measured vector similarities and signed-gamma generation results show that
this is not a random file inversion: the newly extracted vector is strongly
aligned with both the previous self-extracted vector and the released author
vector, while its negative causal direction compresses generation.

## Two different claims must not be conflated

The representational contrast direction is

```text
v_repr = mean_i(h_pre(short_i) - h_pre(long_i)).
```

This definition guarantees that the paired short examples have a larger average
projection than the long examples on the calibration representations. It does
not mathematically guarantee that adding the direction during autoregressive
generation shortens the trajectory. A separating/readout direction is not
automatically a causal intervention direction.

The deployed compression direction is therefore defined as

```text
v_compress = s * v_repr,  s in {-1, +1},
h_pre <- h_pre + gamma * v_compress,  gamma > 0.
```

The sign `s` is selected once on a held-out calibration split using symmetric
positive/negative gamma trials. A sign may be accepted only if it reduces token
count while remaining within a predeclared accuracy tolerance. If neither sign
passes, the pipeline refuses to label the vector a compression vector.

This mirrors the explicit orientation step in CAST, but orients against the
causal deployment objective rather than only the representation labels. Final
accuracy/token reporting must use a different test split from sign calibration.

## Residual-stream location

New vectors use `block_input` for both extraction and injection. This is the
pre-residual location used by the reference implementations. A vector sidecar
records its activation site and recommended sign. Evaluation rejects a mismatch
unless an explicit diagnostic override is supplied.

`block_output` remains available only for reproducing legacy ASC artifacts.

## Required experimental sequence

1. Extract `short_minus_long` at `block_input` from the checked MATH-train pairs.
2. On a held-out calibration split (for example GSM8K train), run symmetric
   signed gamma values using addition at `block_input`.
3. Save the causally oriented vector with its calibration provenance.
4. On GSM8K test, use only positive gamma and addition with the oriented vector.
5. Report the raw representational direction and the causal orientation sign
   separately. Do not rewrite a negative result as if the original paper sign
   had succeeded.
