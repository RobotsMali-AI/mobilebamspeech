// native_bindings.dart

import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

final DynamicLibrary nemoLib = Platform.isAndroid
    ? DynamicLibrary.open('libNeMoOnnxSharp.so')
    : throw UnsupportedError("Platform not supported");

// FFI typedefs
typedef InitSession_C = Uint64 Function(
  Pointer<Utf8> onnxPath,
  Int32 modelType,
  Pointer<Utf8> vocabulary,
  Pointer<Utf8> extraPaths
);
typedef InitSession_Dart = int Function(
  Pointer<Utf8>,
  int,
  Pointer<Utf8>,
  Pointer<Utf8>
);

typedef Transcribe_C = Pointer<Utf8> Function(Uint64 handle, Pointer<Utf8> wavPath);
typedef Transcribe_Dart = Pointer<Utf8> Function(int, Pointer<Utf8>);

typedef FreeCString_C = Void Function(Pointer<Utf8>);
typedef FreeCString_Dart = void Function(Pointer<Utf8>);

typedef Dispose_C = Void Function(Uint64);
typedef Dispose_Dart = void Function(int);

// Bind native functions
final InitSession_Dart _nativeInitSession = nemoLib
    .lookup<NativeFunction<InitSession_C>>('InitSession')
    .asFunction();

final Transcribe_Dart transcribe = nemoLib
    .lookup<NativeFunction<Transcribe_C>>('Transcribe')
    .asFunction();

final FreeCString_Dart freeCString = nemoLib
    .lookup<NativeFunction<FreeCString_C>>('FreeCString')
    .asFunction();

final Dispose_Dart disposeSession = nemoLib
    .lookup<NativeFunction<Dispose_C>>('Dispose')
    .asFunction();

/// Clean wrapper utility to simplify passing collections out to the FFI boundary
int initSession({
  required String mainModelPath,
  required int modelType,
  required List<String> vocabulary,
  List<String> extraModelPaths = const [],
}) {
  final Pointer<Utf8> pathPtr = mainModelPath.toNativeUtf8();
  final Pointer<Utf8> vocabPtr = vocabulary.join('\n').toNativeUtf8();
  final Pointer<Utf8> extraPathsPtr = extraModelPaths.join('\n').toNativeUtf8();

  try {
    return _nativeInitSession(pathPtr, modelType, vocabPtr, extraPathsPtr);
  } finally {
    malloc.free(pathPtr);
    malloc.free(vocabPtr);
    malloc.free(extraPathsPtr);
  }
}
