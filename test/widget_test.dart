import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobilebamspeech/main.dart';

void main() {
  testWidgets('shows the three speech feature destinations', (tester) async {
    tester.view.physicalSize = const Size(1080, 1920);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const LocalSpeechApp());
    await tester.pump();

    expect(find.text('ASR'), findsOneWidget);
    expect(find.text('SLU'), findsOneWidget);
    expect(find.text('TTS'), findsOneWidget);
    expect(find.byType(BottomNavigationBar), findsOneWidget);
  });
}
