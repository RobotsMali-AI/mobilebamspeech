import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;
import 'dart:ffi' as ffi;
import '../../core/asset_manager.dart';

class TtsPage extends StatefulWidget {
  const TtsPage({super.key});

  @override
  State<TtsPage> createState() => _TtsPageState();
}

class _TtsPageState extends State<TtsPage> {
  final TextEditingController _textController = TextEditingController();
  final AudioPlayer _audioPlayer = AudioPlayer();

  sherpa.OfflineTts? _ttsEngine;
  bool _isEngineLoading = true;
  bool _isGenerating = false;
  bool _isPlaying = false;

  String _statusMessage = "Loading Bambara TTS Voice Engine...";
  String? _generatedWavPath;

  final List<String> _bambaraSamples = [
    "i ni sogoma hɛrɛ bɛ",
    "n bɛ a ɲini ka bama jago kɔnti dayɛlɛ dɔrɔmɛ milyɔn saba dɛmɛ kama segu",
    "an ka modɛli bɛ se ka bamanankan fɔ"
  ];

  @override
  void initState() {
    super.initState();
    _initializeTtsEngine();

    _textController.addListener(_handleTextChange);

    _audioPlayer.onPlayerStateChanged.listen((PlayerState state) {
      if (mounted) {
        setState(() => _isPlaying = state == PlayerState.playing);
      }
    });
  }

  void _handleTextChange() {
    if (mounted) {
      setState(() {});
    }
  }

  @override
  void dispose() {
    _textController.removeListener(_handleTextChange);
    _textController.dispose();
    _audioPlayer.dispose();
    super.dispose();
  }

  Future<void> _initializeTtsEngine() async {
    try {
      // Force loading the original onnxruntime library installed by sherpa-onnx
      ffi.DynamicLibrary.open('libonnxruntime.so');
      sherpa.initBindings();

      final String localModelPath = await AssetManager.copyAssetToLocal('assets/vits/model.onnx');
      final String localTokensPath = await AssetManager.copyAssetToLocal('assets/vits/tokens.txt');

      final vitsConfig = sherpa.OfflineTtsVitsModelConfig(
        model: localModelPath,
        tokens: localTokensPath,
        lexicon: "",
        noiseScale: 0.667,
        lengthScale: 1.0,
        noiseScaleW: 0.8,
      );

      final modelConfig = sherpa.OfflineTtsModelConfig(
        vits: vitsConfig,
        numThreads: 2, // ideal on most modern android devices
        debug: false,
        provider: "cpu",
      );

      final ttsConfig = sherpa.OfflineTtsConfig(
        model: modelConfig,
      );

      _ttsEngine = sherpa.OfflineTts(ttsConfig);

      setState(() {
        _isEngineLoading = false;
        _statusMessage = "Bambara VITS Voice Engine Loaded";
      });
    } catch (e) {
      setState(() {
        _isEngineLoading = false;
        _statusMessage = "Engine Load Error: $e";
      });
    }
  }

  void _insertSpecialCharacter(String char) {
    final text = _textController.text;
    final selection = _textController.selection;

    final newText = text.replaceRange(selection.start, selection.end, char);
    final textLength = char.length;

    _textController.text = newText;
    _textController.selection = TextSelection.collapsed(
      offset: selection.start + textLength,
    );
  }

  Future<String> _createWavFile(Float32List samples, int sampleRate) async {
    final tempDir = await getTemporaryDirectory();
    final String wavPath = "${tempDir.path}/generated_tts_output.wav";
    final File file = File(wavPath);

    final int numSamples = samples.length;
    final int numBytes = numSamples * 2; // 16-bit audio = 2 bytes per sample
    final byteData = ByteData(44 + numBytes);

    // RIFF Identifier
    byteData.setUint8(0, 0x52); // R
    byteData.setUint8(1, 0x49); // I
    byteData.setUint8(2, 0x46); // F
    byteData.setUint8(3, 0x46); // F
    byteData.setUint32(4, 36 + numBytes, Endian.little);

    // Format Header
    byteData.setUint8(8, 0x57);  // W
    byteData.setUint8(9, 0x41);  // A
    byteData.setUint8(10, 0x56); // V
    byteData.setUint8(11, 0x45); // E

    // Sub-chunk 1: Format Description Chunk
    byteData.setUint8(12, 0x66); // f
    byteData.setUint8(13, 0x6D); // m
    byteData.setUint8(14, 0x74); // t
    byteData.setUint8(15, 0x20); // ' '
    byteData.setUint32(16, 16, Endian.little); // Sub-chunk size
    byteData.setUint16(20, 1, Endian.little);  // Audio Format (1 = PCM Uncompressed)

    // FIX WAS HERE: Shifted from offset 21 to offset 22 to prevent structural byte overlap
    byteData.setUint16(22, 1, Endian.little);  // Number of Channels (1 = Mono)

    byteData.setUint32(24, sampleRate, Endian.little);
    byteData.setUint32(28, sampleRate * 2, Endian.little); // Byte Rate (SampleRate * Channels * BytesPerSample)
    byteData.setUint16(32, 2, Endian.little);              // Block Align (Channels * BytesPerSample)
    byteData.setUint16(34, 16, Endian.little);             // Bits Per Sample (16-bit)

    // Sub-chunk 2: Audio Data Payload
    byteData.setUint8(36, 0x64); // d
    byteData.setUint8(37, 0x61); // a
    byteData.setUint8(38, 0x74); // t
    byteData.setUint8(39, 0x61); // a
    byteData.setUint32(40, numBytes, Endian.little);

    // Quantize Float32 array down to formal Int16 data blocks safely
    int loopOffset = 44;
    for (int i = 0; i < numSamples; i++) {
      double sample = samples[i].clamp(-1.0, 1.0);
      int pcm16Sample = (sample < 0 ? sample * 32768 : sample * 32767).toInt();
      byteData.setInt16(loopOffset, pcm16Sample, Endian.little);
      loopOffset += 2;
    }

    await file.writeAsBytes(byteData.buffer.asUint8List(), flush: true);
    return wavPath;
  }

  Future<void> _synthesizeSpeech() async {
    final String textToSynthesize = _textController.text.trim();
    if (textToSynthesize.isEmpty || _ttsEngine == null) return;

    setState(() {
      _isGenerating = true;
      _statusMessage = "Running acoustic model graph execution...";
    });

    try {
      final sherpa.GeneratedAudio audioOutput = _ttsEngine!.generate(
          text: textToSynthesize,
          sid: 0,
          speed: 1.0
      );

      if (audioOutput.samples.isNotEmpty) {
        final String savedPath = await _createWavFile(audioOutput.samples, audioOutput.sampleRate);

        setState(() {
          _generatedWavPath = savedPath;
          _statusMessage = "Acoustic generation complete.";
        });

        await _playbackAudio();
      } else {
        setState(() => _statusMessage = "Error: Acoustic generation pass returned an empty frame sequence.");
      }
    } catch (e) {
      setState(() => _statusMessage = "Synthesis Pipeline Exception: $e");
    } finally {
      setState(() => _isGenerating = false);
    }
  }

  Future<void> _playbackAudio() async {
    if (_generatedWavPath == null) return;
    try {
      if (_isPlaying) {
        await _audioPlayer.stop();
      } else {
        await _audioPlayer.play(DeviceFileSource(_generatedWavPath!));
      }
    } catch (e) {
      setState(() => _statusMessage = "Hardware Playback Exception: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text("Sherpa TTS Testbed", style: TextStyle(color: Colors.black)),
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: _isEngineLoading ? Colors.orange.shade50 : Colors.green.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: _isEngineLoading ? Colors.orange.shade200 : Colors.green.shade200),
                ),
                child: Text(
                  "Engine Status: $_statusMessage",
                  style: TextStyle(
                      color: _isEngineLoading ? Colors.orange.shade800 : Colors.green.shade800,
                      fontWeight: FontWeight.w500
                  ),
                ),
              ),
              const SizedBox(height: 20),

              const Text("Select Reference Bambara Phrase:", style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              ..._bambaraSamples.map((phrase) => Card(
                elevation: 0,
                margin: const EdgeInsets.symmetric(vertical: 4),
                color: Colors.grey.shade50,
                shape: RoundedRectangleBorder(
                    side: BorderSide(color: Colors.grey.shade200),
                    borderRadius: BorderRadius.circular(8)
                ),
                child: ListTile(
                  title: Text(phrase, style: const TextStyle(fontSize: 14)),
                  trailing: const Icon(Icons.arrow_forward_rounded, size: 18),
                  onTap: _isEngineLoading || _isGenerating ? null : () {
                    _textController.text = phrase;
                  },
                ),
              )),

              const SizedBox(height: 20),

              TextField(
                controller: _textController,
                maxLines: 3,
                decoration: InputDecoration(
                  hintText: "Sɛbɛnni kɛ yan ka bamanankan fɔkan bɔ...",
                  labelText: "Bambara Input Text Sequence",
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                  alignLabelWithHint: true,
                ),
              ),
              const SizedBox(height: 10),

              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: ['ɛ', 'ɔ', 'ɲ', 'ŋ'].map((char) => Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 4.0),
                    child: OutlinedButton(
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        side: BorderSide(color: Colors.grey.shade300),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
                      ),
                      onPressed: () => _insertSpecialCharacter(char),
                      child: Text(
                          char,
                          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.blue)
                      ),
                    ),
                  ),
                )).toList(),
              ),

              const SizedBox(height: 32),

              if (_generatedWavPath != null) ...[
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.blue.shade50,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    children: [
                      IconButton(
                        icon: Icon(_isPlaying ? Icons.pause_circle_filled : Icons.play_circle_filled),
                        iconSize: 40,
                        color: Colors.blue.shade700,
                        onPressed: _playbackAudio,
                      ),
                      const SizedBox(width: 12),
                      const Expanded(
                        child: Text(
                          "Synthesis WAV Stream Output cached locally and ready for replay evaluation.",
                          style: TextStyle(fontSize: 13, color: Colors.black54),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),
              ],

              ElevatedButton.icon(
                onPressed: (_isEngineLoading || _isGenerating || _textController.text.trim().isEmpty)
                    ? null
                    : _synthesizeSpeech,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  backgroundColor: Colors.blue,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                icon: _isGenerating
                    ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                    : const Icon(Icons.record_voice_over),
                label: Text(_isGenerating ? "Synthesizing Audio Matrix..." : "Generate Bambara Speech Stream"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

extension on TextStyle {
  get colorLinear => const TextStyle(color: Colors.black54);
}
const Color colorsBlueLinear = Color(0xFF1976D2);