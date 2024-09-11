import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

class MapScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("지도"),
      ),
      body: WebView(
        initialUrl: 'http://192.168.0.16:8080/convenience_stores_map.html',
        javascriptMode: JavascriptMode.unrestricted,
      ),
    );
  }
}
