# Mobile Bam Speech

Mobile Bam Speech is a functional Flutter/Android demonstration of running
RobotsMali speech models locally on an ARM64 phone. It shows how to integrate
automatic speech recognition (ASR), speech intent and slot understanding (SLU),
and VITS text-to-speech (TTS). Most deployed models are based on NVIDIA NeMo.

This repository is an integration testbed, not a production application or an
optimized inference stack. In particular, VITS synthesis can be slow on mobile
hardware, and the deployed TTS checkpoint is undertrained. Consult each model's
Hugging Face page for architecture, training, limitations, and intended-use
details.

## Deployed Models

- ASR: [`RobotsMali/quartznum-v0`](https://huggingface.co/RobotsMali/quartznum-v0)
  and [`RobotsMali/soloni-be-kalan-v0`](https://huggingface.co/RobotsMali/soloni-be-kalan-v0)
- Intent and slot understanding:
  [`RobotsMali/soloni-ic-slot-fintech-v0`](https://huggingface.co/RobotsMali/soloni-ic-slot-fintech-v0)
- TTS: [`RobotsMali/bam-vits-fintech`](https://huggingface.co/RobotsMali/bam-vits-fintech)

The exact assets selected by the application are declared in `pubspec.yaml`. However, we do not share the onnx files in this repository, 
if we wish to run this application, please recreate the required onnx assets with the scripts inside `utils/` folder

## Mobile Inference Architecture

ASR and SLU use shared libraries compiled with
[NeMoOnnxSharp](https://github.com/diarray-hub/NeMoOnnxSharp). TTS uses the
Flutter `sherpa_onnx` package. The required ARM64 libraries are included under
`android/app/src/main/jniLibs/arm64-v8a/`.

NeMoOnnxSharp and sherpa-onnx can require different ONNX Runtime versions. The
NeMo copy is therefore renamed to `libonnxruntime_nemo.so`, while sherpa-onnx
loads `libonnxruntime.so`. Keep the explicit
`ffi.DynamicLibrary.open('libonnxruntime.so')` call in the TTS initialization
path; changing this loading arrangement can break either inference stack.

## Model Export

The scripts under [`utils/`](utils/) create the ONNX assets used by the app.
See [`utils/README.md`](utils/README.md) for commands, dependencies, tested
models, custom wrapper behavior, and compatibility limits. The ASR exporter can
use NeMo's native `.export()` support. The SLU and Hugging Face VITS models need
custom export paths that are currently tuned and validated against the
RobotsMali model families deployed here.

## Android Configuration

The Android application must extract its native libraries. Keep these
attributes on the `<application>` element in `AndroidManifest.xml`:

```xml
android:extractNativeLibs="true"
tools:replace="android:extractNativeLibs"
```

The manifest root must also declare
`xmlns:tools="http://schemas.android.com/tools"`. When replacing model or media
files, declare every shipped asset in `pubspec.yaml`.

## Run the App

Use a physical Android ARM64 device because the bundled native libraries target
that ABI:

```bash
flutter pub get
flutter run
```

Before contributing, run:

```bash
dart format lib test
flutter analyze
flutter test
```

Performance and output quality vary by device and checkpoint. Validate exported
models on target hardware and treat this application as a working integration
example rather than an optimized deployment baseline.
