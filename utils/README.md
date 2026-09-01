# Model Export Utilities

These scripts convert the RobotsMali models used by this functional Android
demo to ONNX. They are reference integrations, not general-purpose exporters.
The custom paths have only been tested with the model configurations deployed
in this repository.

## Exporters

- `export-nemo-asr.py` loads a NeMo CTC or RNNT-BPE checkpoint and uses NeMo's
  native `.export()` implementation. The application uses
  [`RobotsMali/quartznum-v0`](https://huggingface.co/RobotsMali/quartznum-v0)
  and [`RobotsMali/soloni-be-kalan-v0`](https://huggingface.co/RobotsMali/soloni-be-kalan-v0).
- `export-nemo-slu.py` manually exports the encoder, classifier, inference
  embedding, and cached decoder of
  [`RobotsMali/soloni-ic-slot-fintech-v0`](https://huggingface.co/RobotsMali/soloni-ic-slot-fintech-v0).
  `SLUIntentSlotBPEModel` does not provide a complete `.export()` path for this
  deployment, so the script also compares PyTorch and ONNX Runtime outputs.
- `export-vits-onnx.py` manually adapts
  [`RobotsMali/bam-vits-fintech`](https://huggingface.co/RobotsMali/bam-vits-fintech)
  to the input names and metadata expected by sherpa-onnx. Hugging Face
  `VitsModel` does not expose a directly usable `.export()` method.
- `wrappers.py` contains the PyTorch adapters shared by the custom exporters.

Run scripts from the repository root. Examples:

```bash
python utils/export-nemo-asr.py RobotsMali/quartznum-v0 assets/asr/quartznum.onnx
python utils/export-nemo-slu.py
python utils/export-vits-onnx.py
```

The VITS requirements are listed in the repository-level `requirements.txt`.
The NeMo scripts additionally require a NeMo/PyTorch environment compatible
with the source checkpoints (that can be verified on the model cards, it's usually nemo 2.5.0).
I recommend using separate virtual envs for NeMo and VITS as the version of transformers and some other dependencies may clash.
Exported files must be declared in `pubspec.yaml` before Flutter can package them.

## Compatibility Limits

The manual SLU and VITS graphs reflect the deployed checkpoints' interfaces and
may require adaptation for other architectures or configurations. In
particular, the SLU embedding deliberately omits `token_type_ids`; passing a
zero tensor is not equivalent for the tested model because token type zero has
a learned embedding. Models that require `token_type_ids`, use a different
decoder-memory layout, change encoder outputs, or expose different speaker and
tokenizer settings need corresponding wrapper, dynamic-axis, and verification
updates.

RobotsMali models within the same ASR, SLU, or VITS family generally share the
relevant configuration, so these scripts are expected to cover that family.
Always inspect the source model configuration, run the built-in numerical
checks where available, test dynamic input lengths, and validate the resulting
audio or predictions on the target ARM64 device.
