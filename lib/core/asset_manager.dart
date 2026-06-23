import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

class AssetManager {
  /// Copies an asset file to local app storage if it doesn't exist yet.
  /// Returns the absolute path of the local file.
  static Future<String> copyAssetToLocal(String assetPath) async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final fileName = p.basename(assetPath);
      final localFile = File(p.join(directory.path, fileName));

      // Avoid redundant copies to optimize startup/resource times
      if (!await localFile.exists()) {
        final byteData = await rootBundle.load(assetPath);
        await localFile.writeAsBytes(
          byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes),
          flush: true,
        );
      }
      return localFile.path;
    } catch (e) {
      throw Exception("Failed to extract asset '$assetPath' to local storage: $e");
    }
  }
}