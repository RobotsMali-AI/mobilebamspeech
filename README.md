# mobilebamspeech

Deploying ASR, Speech Intent Classification (SIC) and TTS (VITS) on mobile.

This application is RobotsMali's testbed for the deploying our Bambara speech models on mobile platforms (specifically android here). The repository holds the codes and configurations for deploying our small ASR (CTC), Speech Intent Classification and TTS fine-tunes directly on android device with Flutter and shared object C libraries.

The library for inference our NeMo models was compiled with [NeMoOnnxSharp](https://github.com/diarray-hub/NeMoOnnxSharp) while the deployment of our VITS based TTS systems rely on sherpa-onnx flutter package. Since both our NeMoOnnxSharp library and sherpa-onnx may need and reference different onnxruntime versions, we have renamed `libonnxruntime.so` shared object for our custom library to `libonnxruntime_nemo.so`.

However when the app loads `libNeMoOnnxSharp.so` for the ASR and SLU pages it maps all references to `libonnxruntime.so` to `libonnxruntime_nemo.so`, so in tts_page.dart you must force the loading of the original libonnxruntime.so library installed by sherpa-onnx, make sure you call `ffi.DynamicLibrary.open('libonnxruntime.so')` inside _initializeTtsEngine method.

This repository already contains the above mentioned shared objects. Please, refer to the scripts in [utils folder](./utils) to create your own ONNX exports from our [ASR/SLU/TTS Models](https://hf.co/RobotsMali)

## App Config

Because all the speech recognition functionalities in this app are accessed through native libs (shared objects), you should make sure your android app supports extracting native libraries by setting the parameters mentioned below in your AndroidManifest.xml file (under the application tag)

```
android:extractNativeLibs="true"
tools:replace="android:extractNativeLibs">
```

Also, ensure the manifest tag has `xmlns:tools="http://schemas.android.com/tools"`. Lastly, make sure the shared objects are present in [android/app/src/main/jniLibs/arm64-v8a/](./android/app/src/main/jniLibs/arm64-v8a/).

Once you have verified your setup and downloaded the onnx models you should make sure they are correctly referenced in [pubspec.yaml](./pubspec.yaml)

## Run the app

Once you checked the above points you can launch the app with (make sure you use a real android-arm64 device):

```bash
flutter pub get
flutter run
```

## Notes

This app is **not** optimized, it is only intented to showcase the simplest way you can deploy [RobotsMali's Bambara Speech Models](https://hf.co/RobotsMali) on android devices with flutter. but since the models are lightweight, one may find inference speed to be decent on most modern android devices. Also the models used for this demo may not be the latest or most performant versions.