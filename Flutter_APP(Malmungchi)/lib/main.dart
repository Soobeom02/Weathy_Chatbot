import 'package:flutter/material.dart';
import 'weather_detail_screen.dart';  // 날씨 상세 페이지 import
import 'weather_news_screen.dart';   // 날씨 뉴스 페이지 import
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'package:url_launcher/url_launcher.dart';
import 'package:geolocator/geolocator.dart';  // 위치 정보 가져오기

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: ChatScreen(),
      theme: ThemeData(
        primaryColor: Colors.blue[700],
        colorScheme: ColorScheme.fromSwatch().copyWith(secondary: Colors.white),
      ),
    );
  }
}

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<Message> messages = [];
  final TextEditingController textController = TextEditingController();

  double latitude = 37.5665;
  double longitude = 126.9780;
  String? cityName = '현재 위치';

  final Map<String, dynamic> cities = {
    '현재 위치': [null, null],
    '서울': [37.5665, 126.9780],
    '부산': [35.1796, 129.0756],
    '대구': [35.8714, 128.6014],
    '인천': [37.4563, 126.7052],
    '광주': [35.1595, 126.8526],
    '대전': [36.3504, 127.3845],
    '울산': [35.5390, 129.3114],
    '제주': [33.4996, 126.5312],
  };

  bool initialLoad = true;

  @override
  void initState() {
    super.initState();
    _getLocation();  // 앱 시작 시 현재 위치를 가져옵니다.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      setState(() {
        messages.add(Message(
          text: "저는 사전지식 기반 챗봇이 아닌, API 기반 날씨 정보 제공 중심의 챗봇입니다."
              "\n\n"
              "날씨 정보에 대한 질문은 최대 5일 후까지 답변이 가능하며,"
              " 날씨를 바탕으로 음식이나 코디 추천도 가능하며, 우산 관련 질문을 하시면 구입할 수 있는 주변 편의점을 알려드립니다!"
              "\n\n"
              "(주말, 금요일)같이 요일 형식이 아닌 (내일, 3일 뒤 3시)같이 상대적 기준으로 날짜를 학습했으며,"
              " 맞춤법 및 띄어쓰기를 맞추어 주시면 더욱 좋은 답변을 확인하실 수 있습니다!",
          isUser: false,
        ));
      });
    });
  }

  void _sendMessage(String text) async {
    if (text.isNotEmpty) {
      setState(() {
        messages.add(Message(text: text, isUser: true));
        messages.add(Message(text: "웨디가 열심히 답변을 만들고 있어요...", isUser: false, isLoading: true));
      });
      textController.clear();
      try {
        final response = await http.post(
          Uri.parse('https://malmungchi.duckdns.org/model1/greet'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'input': text, 'latitude': latitude, 'longitude': longitude}),
        );

        if (response.statusCode == 200) {
          final responseData = json.decode(response.body);
          setState(() {
            messages.removeLast();
            messages.add(Message(text: responseData['response'], isUser: false));

            if (responseData['response'].contains('지도로 안내해드리겠습니다!')) {
              messages.removeLast();
              messages.add(Message(text: responseData['response'], isUser: false, isMap: true));
            }
          });
        } else {
          print('Server error: ${response.body}');
          setState(() {
            messages.removeLast();
            messages.add(Message(text: "죄송해요, 아직 그 부분은 배우지 못했어요.. (๑´╹‸╹`๑)", isUser: false));
          });
        }
      } catch (e) {
        print('Error sending request: $e');
        setState(() {
          messages.removeLast();
          messages.add(Message(text: "죄송해요, 아직 그 부분은 배우지 못했어요.. (๑´╹‸╹`๑)", isUser: false));
        });
      }
    }
  }

  void _fetchWeatherForSelectedCity(String city) {
    if (city == '현재 위치') {
      if (cityName != '현재 위치') {  // 현재 위치 구분선이 두번 생기는 것을 방지
        messages.add(Message(text: "현재 위치", isUser: false, isDivider: true)); // 현재 위치 메시지 추가
      }
      _getLocation();
    } else {
      setState(() {
        latitude = cities[city][0];
        longitude = cities[city][1];
        cityName = city;

        // Add a divider message to the chat
        messages.add(Message(text: city, isUser: false, isDivider: true)); // 지역명 메시지 추가

        _showLocationSnackBar(cityName!, latitude, longitude);
      });
    }
  }

  void _getLocation() async {
    bool serviceEnabled;
    LocationPermission permission;

    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      setState(() {
        latitude = 37.5665;
        longitude = 126.9780;
        cityName = '서울';
        if (initialLoad) {
          messages.add(Message(text: "현재 위치", isUser: false, isDivider: true)); // 현재 위치 메시지 추가
          initialLoad = false;
        }
      });
      return;
    }

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        setState(() {
          latitude = 37.5665;
          longitude = 126.9780;
          cityName = '서울';
          if (initialLoad) {
            messages.add(Message(text: "현재 위치", isUser: false, isDivider: true)); // 현재 위치 메시지 추가
            initialLoad = false;
          }
        });
        return;
      }
    }

    if (permission == LocationPermission.deniedForever) {
      setState(() {
        latitude = 37.5665;
        longitude = 126.9780;
        cityName = '서울';
        if (initialLoad) {
          messages.add(Message(text: "현재 위치", isUser: false, isDivider: true)); // 현재 위치 메시지 추가
          initialLoad = false;
        }
      });
      return;
    }

    Position position = await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
    setState(() {
      latitude = position.latitude;
      longitude = position.longitude;
      cityName = '현재 위치';
      if (initialLoad) {
        messages.add(Message(text: "현재 위치", isUser: false, isDivider: true)); // 현재 위치 메시지 추가
        initialLoad = false;
      }
    });

    _showLocationSnackBar(cityName!, latitude, longitude);
  }


  void _showLocationSnackBar(String cityName, double latitude, double longitude) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: Colors.lightBlue,
        content: Text('$cityName: 위도 ${latitude.toStringAsFixed(4)}, 경도 ${longitude.toStringAsFixed(4)}'),
        duration: Duration(seconds: 1),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10.0),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).primaryColor,
        title: Row(
          children: [
            Container(
              decoration: BoxDecoration(
                color: Colors.lightBlueAccent.withOpacity(0.7),
                borderRadius: BorderRadius.circular(20),
              ),
              padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              child: InkWell(
                onTap: () {
                  showModalBottomSheet(
                      context: context,
                      builder: (BuildContext context) {
                        return Container(
                          height: 200,
                          child: Column(
                            children: <Widget>[
                              ListTile(
                                leading: Image.asset(
                                  'assets/weathy_chat.jpg',
                                  width: 40,
                                  height: 40,
                                  fit: BoxFit.contain,
                                ),
                                title: Text('날씨 챗봇, 웨디'),
                                onTap: () {
                                  Navigator.pop(context);
                                  Navigator.push(context, MaterialPageRoute(builder: (context) => ChatScreen()));
                                },
                              ),
                              ListTile(
                                leading: Image.asset(
                                  'assets/weathy_reporter.jpg',
                                  width: 40,
                                  height: 40,
                                  fit: BoxFit.contain,
                                ),
                                title: Text('날씨 정보 요약'),
                                onTap: () {
                                  Navigator.pop(context);
                                  Navigator.push(context, MaterialPageRoute(builder: (context) => WeatherScreen()));
                                },
                              ),
                              ListTile(
                                leading: Image.asset(
                                  'assets/weathy_writer.jpg',
                                  width: 40,
                                  height: 40,
                                  fit: BoxFit.contain,
                                ),
                                title: Text('날씨 뉴스 모아보기'),
                                onTap: () {
                                  Navigator.pop(context);
                                  Navigator.push(context, MaterialPageRoute(builder: (context) => WeatherNewsScreen()));
                                },
                              ),
                            ],
                          ),
                        );
                      }
                  );
                },
                child: Row(
                  children: [
                    Icon(Icons.menu, color: Colors.white, size: 16),
                    SizedBox(width: 4),
                    Text('메뉴', style: TextStyle(color: Colors.white, fontSize: 12)),
                  ],
                ),
              ),
            ),
            SizedBox(width: 4),
            Image.asset(
              'assets/weathy_chat.jpg',
              width: 40,
              height: 40,
              fit: BoxFit.contain,
            ),
            SizedBox(width: 4),
            Text(
              '날씨 챗봇, 웨디',
              style: TextStyle(
                color: Theme.of(context).colorScheme.secondary,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.0,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
        actions: [
          Container(
            margin: EdgeInsets.only(right: 5), // 오른쪽 간격 추가
            child: PopupMenuButton<String>(
              onSelected: (String value) {
                _fetchWeatherForSelectedCity(value);
              },
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.lightBlueAccent.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(20),
                ),
                padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                child: Row(
                  children: [
                    Icon(Icons.location_on, color: Colors.white, size: 16),
                    SizedBox(width: 4),
                    Text('지역 선택', style: TextStyle(color: Colors.white, fontSize: 12)),
                  ],
                ),
              ),
              itemBuilder: (BuildContext context) {
                return cities.keys.map((String city) {
                  return PopupMenuItem<String>(
                    value: city,
                    child: Text(city),
                  );
                }).toList();
              },
            ),
          ),
        ],


      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: Image.asset(
              'assets/chat_background.jpg',  // 변경된 배경 이미지 경로
              fit: BoxFit.cover,
              color: Colors.black.withOpacity(0.1),
              colorBlendMode: BlendMode.darken,
            ),
          ),
          Column(
            children: <Widget>[
              WeatherWidget(latitude: latitude, longitude: longitude),  // Pass latitude and longitude to WeatherWidget
              Expanded(
                child: ListView.builder(
                  itemCount: messages.length,
                  reverse: true,
                  itemBuilder: (context, index) {
                    final message = messages[messages.length - 1 - index];
                    return message.isUser ? _buildUserMessage(message) : _buildBotMessage(message);
                  },
                ),
              ),
              Container(
                color: Colors.transparent,
                padding: EdgeInsets.all(8.0),
                child: Row(
                  children: <Widget>[
                    Expanded(
                      child: Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(30.0),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 16.0),
                          child: TextField(
                            controller: textController,
                            decoration: InputDecoration(
                              hintText: "Send a message...",
                              border: InputBorder.none,
                            ),
                          ),
                        ),
                      ),
                    ),
                    SizedBox(width: 8.0),
                    Material(
                      color: Colors.transparent,
                      child: IconButton(
                        icon: Icon(Icons.send, color: Theme.of(context).primaryColor),
                        onPressed: () => _sendMessage(textController.text),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildUserMessage(Message message) {
    return ListTile(
      title: Align(
        alignment: Alignment.centerRight,
        child: Container(
          padding: EdgeInsets.symmetric(vertical: 10.0, horizontal: 14.0),
          decoration: BoxDecoration(
            color: Colors.blue[200],
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            message.text,
            style: TextStyle(color: Colors.white),
          ),
        ),
      ),
    );
  }

  Widget _buildDividerMessage(String text) {
    return Row(
      children: [
        Expanded(
          child: Divider(
            color: Colors.blue[500],
            thickness: 0.5,
            endIndent: 8.0,
          ),
        ),
        Text(
          text,
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: Colors.blue[500],
          ),
        ),
        Expanded(
          child: Divider(
            color: Colors.blue[500],
            thickness: 0.5,
            indent: 8.0,
          ),
        ),
      ],
    );
  }

  Widget _buildBotMessage(Message message) {
    if (message.isDivider) {
      return _buildDividerMessage(message.text);
    }
    if (message.isLoading) {
      return _buildLoadingMessage();
    }
    return ListTile(
      leading: CircleAvatar(
        backgroundImage: AssetImage('assets/weathy_basic.jpg'), // 변경된 프로필 사진 경로
        radius: 20,
      ),
      title: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                '웨디',
                style: TextStyle(fontWeight: FontWeight.bold, color: Colors.blue[900]),
              ),
            ],
          ),
          SizedBox(height: 4),
          Container(
            padding: EdgeInsets.symmetric(vertical: 10.0, horizontal: 14.0),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  message.text,
                  style: TextStyle(color: Colors.black),
                ),
                if (message.isMap)
                  GestureDetector(
                    onTap: () async {
                      final url = 'https://malmungchi.duckdns.org/model1/convenience_stores_map';
                      if (await canLaunch(url)) {
                        await launch(url);
                      } else {
                        throw 'Could not launch $url';
                      }
                    },
                    child: Image.asset(
                      _getImageForLocation(cityName),
                      width: 400,
                      height: 160,
                      fit: BoxFit.cover,
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _getImageForLocation(String? cityName) {
    switch (cityName) {
      case '서울':
        return 'assets/seoul.jpg';
      case '울산':
        return 'assets/ulsan.jpg';
      case '부산':
        return 'assets/busan.jpg';
      case '대전':
        return 'assets/daejeon.jpg'; // Corrected typo
      case '대구':
        return 'assets/daegu.jpg';
      case '광주':
        return 'assets/gwangju.jpg'; // Corrected typo
      case '인천':
        return 'assets/incheon.jpg';
      case '제주':
        return 'assets/jeju.jpg';
      default:
        return 'assets/seoul.jpg';
    }
  }

  Widget _buildLoadingMessage() {
    return ListTile(
      leading: CircleAvatar(
        backgroundImage: AssetImage('assets/weathy_basic.jpg'), // 변경된 프로필 사진 경로
        radius: 20,
      ),
      title: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                '웨디',
                style: TextStyle(fontWeight: FontWeight.bold, color: Colors.blue[900]),
              ),
            ],
          ),
          SizedBox(height: 4),
          LoadingDots(),
        ],
      ),
    );
  }
}

class Message {
  String text;
  bool isUser;
  bool isLoading;
  bool isMap;
  bool isDivider;

  Message({required this.text, required this.isUser, this.isLoading = false, this.isMap = false, this.isDivider = false});
}

class LoadingDots extends StatefulWidget {
  @override
  _LoadingDotsState createState() => _LoadingDotsState();
}

class _LoadingDotsState extends State<LoadingDots> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<int> _animation;

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
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        String dots = '.' * (_animation.value % 4);
        return Container(
          padding: EdgeInsets.symmetric(vertical: 10.0, horizontal: 14.0),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            '웨디가 열심히 답변을 만들고 있어요$dots',
            style: TextStyle(color: Colors.black),
          ),
        );
      },
    );
  }
}

class WeatherWidget extends StatefulWidget {
  final double latitude;
  final double longitude;

  WeatherWidget({required this.latitude, required this.longitude});

  @override
  _WeatherWidgetState createState() => _WeatherWidgetState();
}

class _WeatherWidgetState extends State<WeatherWidget> {
  String _temperature = '';
  String _weatherDescription = '';
  String _weatherIconUrl = '';
  String _error = '';

  // Summary 값의 번역
  final Map<String, String> _translatedSummaries = {
    "The day will start with clear sky through the late morning hours, transitioning to partly cloudy":"오전 늦게까지 맑은 하늘로 시작하여, 부분적으로 흐려지는 날씨",
    "The day will start with partly cloudy through the late morning hours, transitioning to clearing": "오전 늦게까지 흐리다 맑은 날씨",
    "Expect a day of partly cloudy with clear spells": "맑은 가운데 부분적으로 흐린 날씨",
    "Expect a day of partly cloudy with rain": "일부 지역이 흐리고 비가 오는 날씨",
    "There will be clear sky today": "하늘이 맑은 날씨",
    "There will be partly cloudy today": "부분적으로 흐린 날씨",
    "You can expect clear sky in the morning, with partly cloudy in the afternoon": "아침에는 맑고, 오후에는 일부 지역이 흐린 날씨",
    "You can expect partly cloudy in the morning, with clearing in the afternoon":"아침에는 부분적으로 흐리고, 오후에 맑아지는 날씨"
  };

  @override
  void initState() {
    super.initState();
    _fetchWeather(widget.latitude, widget.longitude);
  }

  void _fetchWeather(double latitude, double longitude) async {
    var apiKey = '69688b165f9dc2376196570c34bdd33f';  // Replace with your actual API key
    var url = 'https://api.openweathermap.org/data/3.0/onecall?lat=$latitude&lon=$longitude&exclude=minutely,hourly,alerts&appid=$apiKey&units=metric';

    try {
      var response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        String summary = data['daily'][0]['summary'];
        setState(() {
          _temperature = '${data['daily'][0]['temp']['day']}°C';
          _weatherDescription = _translatedSummaries[summary] ?? summary;
          _weatherIconUrl = 'https://openweathermap.org/img/wn/${data['current']['weather'][0]['icon']}@2x.png';
        });
      } else {
        setState(() {
          _error = 'Failed to fetch weather data';
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Error: $e';
      });
    }
  }

  @override
  void didUpdateWidget(covariant WeatherWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.latitude != widget.latitude || oldWidget.longitude != widget.longitude) {
      _fetchWeather(widget.latitude, widget.longitude);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.symmetric(horizontal: 16, vertical: 8), // 위젯과 다른 요소 사이의 간격 설정
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10), // 모든 모서리를 둥글게 만듦
        color: Colors.blueGrey.withOpacity(0.3), // 배경색에 불투명도 적용
      ),
      child: Padding(
        padding: EdgeInsets.all(8),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_weatherIconUrl.isNotEmpty) Image.network(_weatherIconUrl, width: 50, height: 50),
            SizedBox(width: 10),
            Flexible(
              child: Text(
                _temperature.isEmpty ? '로딩 중...' : '$_temperature, $_weatherDescription',
                style: TextStyle(color: Colors.white, fontSize: 16),
                overflow: TextOverflow.visible,
              ),
            ),
            if (_error.isNotEmpty) Text(_error, style: TextStyle(color: Colors.red, fontSize: 16)),
          ],
        ),
      ),
    );
  }
}
