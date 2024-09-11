import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:math';
import 'dart:async';
import 'package:url_launcher/url_launcher.dart';

class WeatherNewsScreen extends StatefulWidget {
  @override
  _WeatherNewsScreenState createState() => _WeatherNewsScreenState();
}

class _WeatherNewsScreenState extends State<WeatherNewsScreen> {
  List<Map<String, dynamic>> newsList = [];
  String selectedCategory = "날씨";
  String selectedRegion = "서울 날씨";
  String defaultIconPath = 'assets/news_icon.jpg';
  bool isLoading = false;

  @override
  void initState() {
    super.initState();
    fetchNews();
  }

  fetchNews() async {
    setState(() {
      isLoading = true;
    });
    final response = await http.get(Uri.parse('https://malmungchi.duckdns.org/model2/greet?keyword=$selectedCategory'));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      setState(() {
        newsList = (data as List).map((item) => {
          'title': item['title'],
          'link': item['link'],
          'content': item['content'],
          'summary': List<String>.from(item['summary']),
          'top_image': item['top_image'] ?? defaultIconPath,
        }).toList();
        isLoading = false;
      });
    } else {
      throw Exception('Failed to load news');
    }
  }

  void _selectCategory(String category) {
    setState(() {
      selectedCategory = category;
      fetchNews();
    });
  }

  void _selectRegion(String region) {
    setState(() {
      selectedCategory = region;
      fetchNews();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Text(
              '날씨 뉴스 모아보기',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(width: 8),
            Image.asset(
              'assets/weathy_writer.jpg',
              width: 40,
              height: 40,
              fit: BoxFit.contain,
            ),
          ],
        ),
        backgroundColor: Color(0xFF808080),
        actions: [
          Container(
            margin: EdgeInsets.only(right: 16.0), // 오른쪽 간격 추가
            child: PopupMenuButton<String>(
              onSelected: (String value) {
                if (["날씨", "비", "태풍", "장마", "폭염", "눈", "폭설", "한파", "미세먼지", "자외선"].contains(value)) {
                  _selectCategory(value);
                } else {
                  _selectRegion(value);
                }
              },
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.brown.withOpacity(0.5),
                  borderRadius: BorderRadius.circular(20),
                ),
                padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                child: Row(
                  children: [
                    Icon(Icons.filter_alt, color: Colors.white, size: 16),
                    SizedBox(width: 4),
                    Text('날씨&지역 선택', style: TextStyle(color: Colors.white, fontSize: 12)),
                  ],
                ),
              ),
              itemBuilder: (BuildContext context) {
                return [
                  PopupMenuItem(value: "날씨", child: Text("날씨")),
                  PopupMenuItem(value: "비", child: Text("비")),
                  PopupMenuItem(value: "태풍", child: Text("태풍")),
                  PopupMenuItem(value: "장마", child: Text("장마")),
                  PopupMenuItem(value: "폭염", child: Text("폭염")),
                  PopupMenuItem(value: "눈", child: Text("눈")),
                  PopupMenuItem(value: "폭설", child: Text("폭설")),
                  PopupMenuItem(value: "한파", child: Text("한파")),
                  PopupMenuItem(value: "미세먼지", child: Text("미세먼지")),
                  PopupMenuItem(value: "자외선", child: Text("자외선")),
                  PopupMenuDivider(),
                  PopupMenuItem(value: "서울 날씨", child: Text("서울 날씨")),
                  PopupMenuItem(value: "부산 날씨", child: Text("부산 날씨")),
                  PopupMenuItem(value: "대구 날씨", child: Text("대구 날씨")),
                  PopupMenuItem(value: "인천 날씨", child: Text("인천 날씨")),
                  PopupMenuItem(value: "광주 날씨", child: Text("광주 날씨")),
                  PopupMenuItem(value: "대전 날씨", child: Text("대전 날씨")),
                  PopupMenuItem(value: "울산 날씨", child: Text("울산 날씨")),
                  PopupMenuItem(value: "제주 날씨", child: Text("제주 날씨")),
                ];
              },
            ),
          ),
        ],


      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: Image.asset(
              'assets/news.jpg', // 변경된 배경 이미지 경로
              fit: BoxFit.cover,
              color: Colors.black.withOpacity(0.1),
              colorBlendMode: BlendMode.darken,
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: isLoading
                ? Center(child: LoadingDots(category: selectedCategory))
                : newsList.isEmpty
                ? Center(child: Text('뉴스를 불러올 수 없습니다.'))
                : GridView.builder(
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 10,
                mainAxisSpacing: 20, // 각 줄마다 공백을 늘림
                childAspectRatio: 0.7, // 카드의 세로 길이를 늘림
              ),
              itemCount: newsList.length,
              itemBuilder: (context, index) {
                final news = newsList[index];
                return NewsCard(
                  title: news['title'],
                  content: news['content'],
                  link: news['link'],
                  summary: news['summary'],
                  topImage: news['top_image'],
                  defaultIcon: defaultIconPath,
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class LoadingDots extends StatefulWidget {
  final String category;

  LoadingDots({required this.category});

  @override
  _LoadingDotsState createState() => _LoadingDotsState();
}

class _LoadingDotsState extends State<LoadingDots> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<int> _animation;
  Random random = Random();
  double top = 0.0;
  double left = 0.0;
  bool visible = false; // 초기 상태를 false로 설정
  Timer? timer;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(seconds: 1),
      vsync: this,
    )..repeat();
    _animation = StepTween(
      begin: 0,
      end: 4,
    ).animate(_controller);

    // 처음에는 0.5초 후에 시작하여, 이후에는 2초마다 실행
    timer = Timer.periodic(Duration(seconds: 2), (Timer t) {
      setState(() {
        top = random.nextDouble() * MediaQuery.of(context).size.height * 0.7;
        left = random.nextDouble() * MediaQuery.of(context).size.width * 0.7;
        visible = !visible;
      });
    });

    // 처음 0.5초 후에 visible을 true로 설정
    Timer(Duration(milliseconds: 2000), () {
      setState(() {
        visible = true;
      });
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(
          child: AnimatedBuilder(
            animation: _animation,
            builder: (context, child) {
              String dots = '.' * (_animation.value % 4);
              return Center(
                child: Text(
                  '${widget.category} 뉴스를 웨디가 열심히 취재중이에요$dots',
                  style: TextStyle(color: Colors.black, fontSize: 16),
                ),
              );
            },
          ),
        ),
        AnimatedPositioned(
          top: top,
          left: left,
          duration: Duration(seconds: 2),
          child: Visibility(
            visible: visible,
            child: Image.asset(
              'assets/weathy_writer.jpg',
              width: 50,
              height: 50,
            ),
          ),
        ),
      ],
    );
  }
}

class NewsCard extends StatelessWidget {
  final String title;
  final String content;
  final String link;
  final List<String> summary;
  final String topImage;
  final String defaultIcon;

  NewsCard({required this.title, required this.content, required this.link, required this.summary, required this.topImage, required this.defaultIcon});

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16.0),
      ),
      elevation: 8, // 더 강한 그림자
      shadowColor: Colors.black54, // 그림자 색상
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16.0),
        child: Stack(
          children: [
            Positioned.fill(
              child: Image.asset(
                'assets/news_background.jpg', // 카드의 배경 이미지 경로
                fit: BoxFit.cover,
              ),
            ),
            Column(
              children: [
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.vertical(top: Radius.circular(16.0)),
                    child: Image.network(
                      topImage,
                      fit: BoxFit.cover,
                      width: double.infinity,
                      height: double.infinity,
                      errorBuilder: (BuildContext context, Object exception, StackTrace? stackTrace) {
                        return Image.asset(defaultIcon, fit: BoxFit.cover);
                      },
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(
                    title,
                    style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold), // 제목 크기 조정
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                ButtonBar(
                  alignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    TextButton(
                      onPressed: () => showContentDialog(context, content, link),
                      style: TextButton.styleFrom(
                        backgroundColor: Color(0xFFB1AEAF), // 버튼의 배경색
                        padding: EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0), // 버튼 패딩 조정
                      ),
                      child: Text(
                        '본문 보기',
                        style: TextStyle(
                          color: Colors.white, // 흰색 텍스트
                          fontWeight: FontWeight.bold, // 두꺼운 글씨
                          fontSize: 12, // 텍스트 크기 조정
                        ),
                      ),
                    ),
                    TextButton(
                      onPressed: () => showSummaryDialog(context, summary),
                      style: TextButton.styleFrom(
                        backgroundColor: Color(0xFFB1AEAF), // 버튼의 배경색
                        padding: EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0), // 버튼 패딩 조정
                      ),
                      child: Text(
                        '요약 보기',
                        style: TextStyle(
                          color: Colors.white, // 흰색 텍스트
                          fontWeight: FontWeight.bold, // 두꺼운 글씨
                          fontSize: 12, // 텍스트 크기 조정
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  void showContentDialog(BuildContext context, String content, String link) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: Color(0xFFD6D6D6), // 다이얼로그 배경색 변경
          title: Text(
            '본문',
            style: TextStyle(color: Colors.black), // 텍스트 색상 변경
          ),
          content: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(content),
                SizedBox(height: 8),
                GestureDetector(
                  onTap: () => _launchURL(link),
                  child: Text(
                    'Link: $link',
                    style: TextStyle(color: Colors.blue, decoration: TextDecoration.underline),
                  ),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              child: Text('닫기', style: TextStyle(color: Colors.black)), // 닫기 버튼 텍스트 색상 변경
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  void showSummaryDialog(BuildContext context, List<String> summary) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: Color(0xFFD6D6D6), // 다이얼로그 배경색 변경
          title: Text(
            '요약',
            style: TextStyle(color: Colors.black), // 텍스트 색상 변경
          ),
          content: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: summary.map((s) => Text(s)).toList(),
            ),
          ),
          actions: [
            TextButton(
              child: Text('닫기', style: TextStyle(color: Colors.black)), // 닫기 버튼 텍스트 색상 변경
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  void _launchURL(String url) async {
    final Uri uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    } else {
      throw 'Could not launch $url';
    }
  }
}
