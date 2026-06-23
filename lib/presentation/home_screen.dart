import 'package:flutter/material.dart';
import 'pages/asr_page.dart';
import 'pages/slu_page.dart';
import 'pages/tts_page.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;

  final List<Widget> _pages = const [
    AsrPage(),
    SluPage(),
    TtsPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _pages,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.multitrack_audio),
            label: 'ASR',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.check_box), // Using fallback/alternative for SLU
            label: 'SLU',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.record_voice_over),
            label: 'TTS',
          ),
        ],
      ),
    );
  }
}