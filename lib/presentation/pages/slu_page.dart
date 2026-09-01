import 'dart:ffi';
import 'package:flutter/material.dart';
import 'package:ffi/ffi.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import '../../core/native_bindings.dart';
import '../../core/asset_manager.dart';
import '../../core/vocab.dart';

class SluPage extends StatefulWidget {
  const SluPage({super.key});

  @override
  State<SluPage> createState() => _SluPageState();
}

class _SluPageState extends State<SluPage> {
  final AudioRecorder _audioRecorder = AudioRecorder();

  // State configuration variables
  int? _sessionHandle;
  bool _isLoading = false;
  bool _isRecording = false;
  String _transcription = "Initializing SLU Engine components...";
  String? _selectedAudioKey;

  // Single mock audio instance matching the required test files
  final Map<String, String> _mockAudioFiles = {
    'Test Navigate Example': 'assets/audio/test-navigate.wav',
  };

  @override
  void initState() {
    super.initState();
    // Automatically trigger multi-stage SLU model loading on page startup
    _initializeSluSession();
  }

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

  /// Extracts the 4 component ONNX assets and binds them into a unified Type 2 SLU Engine Session
  Future<void> _initializeSluSession() async {
    _cleanupSession();

    setState(() {
      _isLoading = true;
      _transcription = "Extracting multi-stage model network graphs...";
    });

    try {
      // 1. Resolve local filesystem paths for all required ONNX files
      final String encoderPath = await AssetManager.copyAssetToLocal('assets/slurp/soloni-ic-slot-fintech-v0-encoder.onnx');
      final String embeddingPath = await AssetManager.copyAssetToLocal('assets/slurp/soloni-ic-slot-fintech-v0-embedding.onnx');
      final String decoderPath = await AssetManager.copyAssetToLocal('assets/slurp/soloni-ic-slot-fintech-v0-decoder.onnx');
      final String classifierPath = await AssetManager.copyAssetToLocal('assets/slurp/soloni-ic-slot-fintech-v0-classifier.onnx');

      // 2. Fetch vocabulary mapping specified for the SLU architecture (Model Type 2)
      final List<String> currentVocabulary = AppVocabularies.modelVocabs[2] ?? [];

      // 3. Bind everything down through the Native Marshalling Interface
      final int handle = initSession(
        mainModelPath: encoderPath,
        modelType: 2, // Intent Model type selection criteria
        vocabulary: currentVocabulary,
        extraModelPaths: [embeddingPath, decoderPath, classifierPath],
      );

      setState(() {
        _sessionHandle = handle;
        _transcription = "SLU Engine ready for intent classification. Session: $handle";
      });
    } catch (e) {
      setState(() {
        _transcription = "Critical Error loading SLU model setup: $e";
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  /// Triggers transcription and intent profiling via Native Interop FFI layer
  Future<void> _runInference(String absoluteWavPath) async {
    if (_sessionHandle == null) {
      setState(() => _transcription = "Error: SLU structural pipeline session is uninitialized.");
      return;
    }

    setState(() {
      _isLoading = true;
      _transcription = "Processing acoustic features and decoding intent...";
    });

    try {
      final Pointer<Utf8> wavPathPtr = absoluteWavPath.toNativeUtf8();

      try {
        final Pointer<Utf8> resultPtr = transcribe(_sessionHandle!, wavPathPtr);
        if (resultPtr != nullptr) {
          final String dartStringValue = resultPtr.toDartString();
          // Clear native heap memory bounds directly via backend binding export
          freeCString(resultPtr);
          setState(() {
            _transcription = dartStringValue.isNotEmpty ? dartStringValue : "[Unrecognized Intent/Speech]";
          });
        } else {
          setState(() => _transcription = "Inference sequence returned an invalid null string pointer.");
        }
      } finally {
        malloc.free(wavPathPtr);
      }
    } catch (e) {
      setState(() => _transcription = "Inference Execution Exception: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  /// Extracts sample WAV files and routes them into the evaluation routine
  Future<void> _handleAudioSelectionChange(String? audioKey) async {
    if (audioKey == null) return;
    setState(() => _selectedAudioKey = audioKey);

    final String relativeAssetPath = _mockAudioFiles[audioKey]!;
    try {
      final String localWavPath = await AssetManager.copyAssetToLocal(relativeAssetPath);
      await _runInference(localWavPath);
    } catch (e) {
      setState(() => _transcription = "Error preparing sample asset recording: $e");
    }
  }

  /// Toggles mic capturing configuration tailored to match NeMo feature-extractor constraints (16kHz Mono PCM)
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
          final String targetRecordPath = '${tempDir.path}/live_slu_input.wav';

          const RecordConfig recordConfig = RecordConfig(
            encoder: AudioEncoder.wav,
            sampleRate: 16000,
            numChannels: 1,
          );

          await _audioRecorder.start(recordConfig, path: targetRecordPath);
          setState(() {
            _isRecording = true;
            _selectedAudioKey = null; // Reset example audio selection layout hook
          });
        } else {
          setState(() => _transcription = "System permission error: Microphone recording access denied.");
        }
      } catch (e) {
        setState(() => _transcription = "Failed to launch device hardware recording layer: $e");
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text("NeMo SLU Intent Engine", style: TextStyle(color: Colors.black)),
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status Info Banner matching the space of the omitted dropdown selector
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: _sessionHandle != null ? Colors.green.shade50 : Colors.blue.shade50,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: _sessionHandle != null ? Colors.green.shade200 : Colors.blue.shade200),
              ),
              child: Text(
                _sessionHandle != null
                    ? "Status: Soloni Multi-Task Intent Model Bound"
                    : "Status: Connecting Model Pipeline Component Hooks...",
                style: TextStyle(
                    color: _sessionHandle != null ? Colors.green.shade800 : Colors.blue.shade800,
                    fontWeight: FontWeight.w500
                ),
              ),
            ),
            const SizedBox(height: 20),

            // Asset Audio Selection Dropdown Match
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: "Select Target Sample Audio File"),
              value: _selectedAudioKey,
              items: _mockAudioFiles.keys.map((String key) {
                return DropdownMenuItem<String>(value: key, child: Text(key));
              }).toList(),
              onChanged: (_isLoading || _sessionHandle == null) ? null : _handleAudioSelectionChange,
            ),
            const SizedBox(height: 40),

            // Shared Audio Interaction Matrix View
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
                    _isRecording ? "Listening to statement... Press to analyze intent" : "Tap mic to record audio frame inputs",
                    style: TextStyle(color: _isRecording ? Colors.red : Colors.grey.shade600),
                  ),
                ],
              ),
            ),

            // Alignment Target Processing Frame Layout Output
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