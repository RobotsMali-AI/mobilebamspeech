import 'dart:ffi';
import 'package:flutter/material.dart';
import 'package:ffi/ffi.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import '../../core/native_bindings.dart';
import '../../core/asset_manager.dart';
import '../../core/vocab.dart'; // Imported Vocabulary Engine Map

class AsrPage extends StatefulWidget {
  const AsrPage({super.key});

  @override
  State<AsrPage> createState() => _AsrPageState();
}

class _AsrPageState extends State<AsrPage> {
  final AudioRecorder _audioRecorder = AudioRecorder();

  // State configuration variables
  int? _sessionHandle;
  bool _isLoading = false;
  bool _isRecording = false;
  String _transcription = "No transcription yet.";

  String? _selectedModelAsset;
  String? _selectedAudioKey;

  // Configuration maps updated to target correct Type specifications
  final Map<String, int> _models = {
    'quartznum.onnx': 0,
    'soloni-be-kalan.onnx': 1,
  };

  final Map<String, String> _mockAudioFiles = {
    'Example Audio 1': 'assets/audio/example1.wav',
    'Example Audio 2': 'assets/audio/example2.wav',
  };

  @override
  void dispose() {
    _cleanupSession();
    _audioRecorder.dispose();
    super.dispose();
  }

  void _cleanupSession() {
    if (_sessionHandle != null) {
      disposeSession(_sessionHandle!);
      _sessionHandle = null;
    }
  }

  /// Extracts the model asset and initializes the native C++ inference session
  Future<void> _handleModelChange(String? modelName) async {
    if (modelName == null) return;
    _cleanupSession();

    setState(() {
      _selectedModelAsset = modelName;
      _isLoading = true;
      _transcription = "Initializing model session...";
    });

    try {
      final fullAssetPath = 'assets/asr/$modelName';
      final localModelPath = await AssetManager.copyAssetToLocal(fullAssetPath);
      final int modelType = _models[modelName]!;

      // Fetch vocabulary matching current architecture model type
      final List<String> currentVocabulary = AppVocabularies.modelVocabs[modelType] ?? [];

      // Access wrapper safely - parameters are converted internally to native memory strings
      final int handle = initSession(
        mainModelPath: localModelPath,
        modelType: modelType,
        vocabulary: currentVocabulary,
        extraModelPaths: const [], // Optional property field parameter mapping
      );

      setState(() {
        _sessionHandle = handle;
        _transcription = "Model loaded successfully. Session Handle: $handle";
      });
    } catch (e) {
      setState(() {
        _transcription = "Error loading model structure setup: $e";
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  /// Triggers transcription via FFI using the passed file path
  Future<void> _runInference(String absoluteWavPath) async {
    if (_sessionHandle == null) {
      setState(() => _transcription = "Error: Initialize a model session first.");
      return;
    }
    setState(() {
      _isLoading = true;
      _transcription = "Transcribing voice context...";
    });

    try {
      final Pointer<Utf8> wavPathPtr = absoluteWavPath.toNativeUtf8();

      try {
        final Pointer<Utf8> resultPtr = transcribe(_sessionHandle!, wavPathPtr);
        if (resultPtr != nullptr) {
          final String dartStringValue = resultPtr.toDartString();
          // CRITICAL: Call the native library's freeing function to avoid severe memory leaks
          freeCString(resultPtr);
          setState(() {
            _transcription = dartStringValue.isNotEmpty ? dartStringValue : "[Empty Speech Output]";
          });
        } else {
          setState(() => _transcription = "Inference returned a null string pointer.");
        }
      } finally {
        malloc.free(wavPathPtr);
      }
    } catch (e) {
      setState(() => _transcription = "Inference Exception: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  /// Handles mock audio selector modifications
  Future<void> _handleAudioSelectionChange(String? audioKey) async {
    if (audioKey == null) return;
    setState(() => _selectedAudioKey = audioKey);

    final String relativeAssetPath = _mockAudioFiles[audioKey]!;
    try {
      final String localWavPath = await AssetManager.copyAssetToLocal(relativeAssetPath);
      await _runInference(localWavPath);
    } catch (e) {
      setState(() => _transcription = "Error preparing sample audio file: $e");
    }
  }

  /// Toggles raw microphone recording state using target format metrics (16kHz Mono PCM)
  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final String? recordedFilePath = await _audioRecorder.stop();
      setState(() => _isRecording = false);

      if (recordedFilePath != null) {
        await _runInference(recordedFilePath);
      }
    } else {
      try {
        if (await _audioRecorder.hasPermission()) {
          final tempDir = await getTemporaryDirectory();
          final String targetRecordPath = '${tempDir.path}/live_asr_input.wav';

          // Rigid constraint enforcement configuration for NeMo runtime matching
          const RecordConfig recordConfig = RecordConfig(
            encoder: AudioEncoder.wav,
            sampleRate: 16000,
            numChannels: 1,
          );

          await _audioRecorder.start(recordConfig, path: targetRecordPath);
          setState(() {
            _isRecording = true;
            _selectedAudioKey = null; // Clear static sample selection visually
          });
        } else {
          setState(() => _transcription = "Microphone access recording permission denied.");
        }
      } catch (e) {
        setState(() => _transcription = "Failed to launch device recording: $e");
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text("NeMo ASR Testbed", style: TextStyle(color: Colors.black)),
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Model Selection Dropdown
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: "Select ASR Model Framework"),
              value: _selectedModelAsset,
              items: _models.keys.map((String key) {
                return DropdownMenuItem<String>(value: key, child: Text(key));
              }).toList(),
              onChanged: _isLoading ? null : _handleModelChange,
            ),
            const SizedBox(height: 20),

            // Asset Audio Selection Dropdown
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: "Select Sample Audio File"),
              value: _selectedAudioKey,
              items: _mockAudioFiles.keys.map((String key) {
                return DropdownMenuItem<String>(value: key, child: Text(key));
              }).toList(),
              onChanged: (_isLoading || _sessionHandle == null) ? null : _handleAudioSelectionChange,
            ),
            const SizedBox(height: 40),

            // Centered Recording Configuration Shell View
            Expanded(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (_isLoading)
                    const CircularProgressIndicator()
                  else
                    ElevatedButton(
                      onPressed: (_sessionHandle == null) ? null : _toggleRecording,
                      style: ElevatedButton.styleFrom(
                        shape: const CircleBorder(),
                        padding: const EdgeInsets.all(30),
                        backgroundColor: _isRecording ? Colors.red : Colors.blue,
                        disabledBackgroundColor: Colors.grey.shade300,
                      ),
                      child: Icon(
                        _isRecording ? Icons.stop : Icons.mic,
                        size: 40,
                        color: Colors.white,
                      ),
                    ),
                  const SizedBox(height: 15),
                  Text(
                    _isRecording ? "Recording active... Press to process" : "Tap mic to stream voice input",
                    style: TextStyle(color: _isRecording ? Colors.red : Colors.grey.shade600),
                  ),
                ],
              ),
            ),

            // Transcription Frame Layout UI
            const Text(
              "Inference Pipeline Output:",
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade300),
              ),
              constraints: const BoxConstraints(minHeight: 120, maxHeight: 200),
              child: SingleChildScrollView(
                child: Text(
                  _transcription,
                  style: const TextStyle(fontSize: 15, fontFamily: 'monospace'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}