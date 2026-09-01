# Repository Guidelines

## Project Structure & Module Organization

This repository is a Flutter testbed for on-device Bambara ASR, speech intent/slot understanding, and VITS text-to-speech. Application code lives in `lib/`: `main.dart` starts the app, `presentation/` contains screens and feature pages, and `core/` contains asset and native-FFI helpers. Tests belong in `test/`. Android and iOS runner projects are under `android/` and `ios/`; Android native libraries are stored in `android/app/src/main/jniLibs/arm64-v8a/`. Audio, model, and vocabulary files are referenced from `assets/` through `pubspec.yaml`. Python conversion tools live in `utils/`.

## Build, Test, and Development Commands

- `flutter pub get` installs the locked Dart and Flutter dependencies.
- `flutter run` launches the app; use a physical Android ARM64 device for native speech inference.
- `flutter analyze` applies the analyzer rules from `analysis_options.yaml`.
- `dart format lib test` formats Dart source and tests.
- `flutter test` runs all widget and unit tests.
- `flutter build apk --debug` produces a debug APK for an integration sanity check.

Model export scripts require a separate Python environment with their ML dependencies. For example, run `python utils/export-nemo-asr.py <model-id> <output.onnx>`; do not assume these dependencies are installed by Flutter tooling.

## Coding Style & Naming Conventions

Follow `flutter_lints` and Dart's standard two-space indentation. Format before committing. Use `snake_case.dart` for files, `UpperCamelCase` for classes/widgets, and `lowerCamelCase` for functions, fields, and variables. Keep UI composition in `presentation/` and reusable platform/model logic in `core/`. Dispose controllers, recorders, players, native sessions, and allocated FFI memory explicitly.

## Testing Guidelines

Use `flutter_test`; name files `*_test.dart` and group tests by feature or widget. Add tests for new navigation and state behavior, and isolate native-library calls behind testable boundaries. The current widget test is a starter smoke test and may need updating as the UI evolves. Run both `flutter analyze` and `flutter test` before opening a pull request. Hardware-dependent ASR/SLU/TTS changes should also be exercised on Android ARM64 and documented in the PR.

## Assets, Native Libraries, and Configuration

Declare every shipped model or media file in `pubspec.yaml`. Preserve `android:extractNativeLibs="true"` and the native-library replacement settings described in `README.md`. Keep the NeMo-renamed ONNX Runtime library separate from the `libonnxruntime.so` loaded by `sherpa_onnx`; changing this loading order can break TTS or recognition.

## Commit & Pull Request Guidelines

History is currently minimal and uses short imperative subjects (for example, `Add git ignore`). Continue with concise, focused commits. PRs should explain the user-visible change, identify affected speech models/platforms, link relevant issues, and list validation commands and devices. Include screenshots or recordings for UI changes; call out large asset, model, native binary, permission, or manifest updates explicitly.
