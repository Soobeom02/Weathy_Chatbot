import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:geolocator/geolocator.dart';

class WeatherScreen extends StatefulWidget {
  @override
  _WeatherScreenState createState() => _WeatherScreenState();
}

class _WeatherScreenState extends State<WeatherScreen> {
  String? description;
  String? cityName = '현재 위치';
  int? temp;
  String? iconUrl;

  double? pressure; // 대기압
  double? visibility; // 가시거리
  double? precipitation;
  double? windSpeed;
  int? humidity;
  int? clouds;
  List hourlyWeather = []; // 시간별 날씨 데이터를 저장할 리스트
  List dailyWeather = []; // 일일 날씨 데이터를 저장할 리스트

  bool isLoading = true;
  String? errorMessage;

  // Default location is Seoul
  double latitude = 37.5665;
  double longitude = 126.9780;

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

  Map<dynamic, dynamic> weatherDescKo = {
    201: '가벼운 비를 동반한 천둥구름',
    200: '비를 동반한 천둥구름',
    202: '폭우를 동반한 천둥구름',
    210: '약한 천둥구름',
    211: '천둥구름',
    212: '강한 천둥구름',
    221: '불규칙적 천둥구름',
    230: '약한 연무를 동반한 천둥구름',
    231: '연무를 동반한 천둥구름',
    232: '강한 안개비를 동반한 천둥구름',
    300: '가벼운 안개비',
    301: '안개비',
    302: '강한 안개비',
    310: '가벼운 적은비',
    311: '적은비',
    312: '강한 적은비',
    313: '소나기와 안개비',
    314: '강한 소나기와 안개비',
    321: '소나기',
    500: '약한 비',
    501: '중간 비',
    502: '강한 비',
    503: '매우 강한 비',
    504: '극심한 비',
    511: '우박',
    520: '약한 소나기 비',
    521: '소나기 비',
    522: '강한 소나기 비',
    531: '불규칙적 소나기 비',
    600: '가벼운 눈',
    601: '눈',
    602: '강한 눈',
    611: '진눈깨비',
    612: '소나기 진눈깨비',
    615: '약한 비와 눈',
    616: '비와 눈',
    620: '약한 소나기 눈',
    621: '소나기 눈',
    622: '강한 소나기 눈',
    701: '박무',
    711: '연기',
    721: '연무',
    731: '모래 먼지',
    741: '안개',
    751: '모래',
    761: '먼지',
    762: '화산재',
    771: '돌풍',
    781: '토네이도',
    800: '구름 한 점 없는 맑은 하늘',
    801: '약간의 구름이 낀 하늘',
    802: '드문드문 구름이 낀 하늘',
    803: '구름이 거의 없는 하늘',
    804: '구름으로 뒤덮인 흐린 하늘',
    900: '토네이도',
    901: '태풍',
    902: '허리케인',
    903: '한랭',
    904: '고온',
    905: '바람부는',
    906: '우박',
    951: '바람이 거의 없는',
    952: '약한 바람',
    953: '부드러운 바람',
    954: '중간 세기 바람',
    955: '신선한 바람',
    956: '센 바람',
    957: '돌풍에 가까운 센 바람',
    958: '돌풍',
    959: '심각한 돌풍',
    960: '폭풍',
    961: '강한 폭풍',
    962: '허리케인',
    'pressure': '대기압', // 추가
    'visibility': '가시거리', // 추가
  };

  @override
  void initState() {
    super.initState();
    _getLocation();
  }

  Future<void> _fetchWeatherData() async {
    setState(() {
      isLoading = true;
    });

    final apiKey = '69688b165f9dc2376196570c34bdd33f'; // 발급받은 API 키로 교체하세요.
    final url =
        'https://api.openweathermap.org/data/3.0/onecall?lat=$latitude&lon=$longitude&exclude=minutely,alerts&appid=$apiKey&units=metric';

    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        int weatherId = data['current']['weather'][0]['id']; // 날씨 ID 추출

        setState(() {
          description = weatherDescKo[weatherId] ?? '설명 없음'; // 맵핑된 한국어 설명 사용
          iconUrl =
          'https://openweathermap.org/img/wn/${data['current']['weather'][0]['icon']}@2x.png';
          temp = data['current']['temp'].round();
          pressure = data['current']['pressure']; // 대기압 추가
          visibility = data['current']['visibility'] / 1000; // 가시거리 추가, 미터를 킬로미터로 변환
          precipitation = data['minutely']?[0]['precipitation'] ?? 0.0;
          windSpeed = data['current']['wind_speed'];
          humidity = data['current']['humidity'];
          clouds = data['current']['clouds'];
          cityName = cityName ?? data['timezone'];
          hourlyWeather = data['hourly'].take(24).toList(); // 24시간 데이터 저장
          dailyWeather = data['daily'].take(7).toList(); // 7일 데이터 저장
          isLoading = false;
          errorMessage = null; // 에러 메시지 초기화
        });
      } else {
        setState(() {
          isLoading = false;
          errorMessage = 'Failed to fetch weather data: ${response.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        isLoading = false;
        errorMessage = 'Error fetching weather data: $e';
      });
    }
  }

  void _getLocation() async {
    try {
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      setState(() {
        latitude = position.latitude;
        longitude = position.longitude;
        cityName = '현재 위치';
      });
      _fetchWeatherData();
    } catch (e) {
      setState(() {
        latitude = 37.5665;
        longitude = 126.9780;
        cityName = '서울'; // 권한 거부 시 서울로 고정
        if (errorMessage == null) {
          errorMessage = '위치 권한이 거부되었습니다. 기본 위치(서울)로 설정합니다.';
        }
      });
      _fetchWeatherData();
    }
  }

  Widget _buildWeatherCard() {
    return Card(
      color: Colors.blue.withOpacity(0.3),
      elevation: 4.0,
      margin: EdgeInsets.all(16),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
      ),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            FittedBox(
              child: Text(
                cityName ?? 'City not found',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 10.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (iconUrl != null) Image.network(iconUrl!, width: 100),
                  SizedBox(width: 20),
                  FittedBox(
                    child: Text(
                      '${temp ?? '?'}°C',
                      style: TextStyle(fontSize: 54, color: Colors.white),
                    ),
                  ),
                ],
              ),
            ),
            Text(
              description ?? 'No description available',
              style: TextStyle(fontSize: 18, color: Colors.white),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildWeatherGrid() {
    return GridView.count(
      shrinkWrap: true,
      crossAxisCount: 3,
      children: <Widget>[
        _buildWeatherGridItem(Icons.speed, '대기압', '$pressure hPa'), // 아이콘과 설명 변경
        _buildWeatherGridItem(Icons.visibility, '가시거리', '$visibility km'), // 아이콘과 설명 변경
        _buildWeatherGridItem(Icons.water_drop, '강수량', '${precipitation?.round()} mm'),
        _buildWeatherGridItem(Icons.air, '풍속', '${windSpeed ?? 0} m/s'),
        _buildWeatherGridItem(Icons.opacity, '습도', '$humidity%'),
        _buildWeatherGridItem(Icons.cloud, '흐림 정도', '$clouds%'),
      ],
    );
  }

  Widget _buildWeatherGridItem(IconData icon, String label, String value) {
    return Container(
      margin: EdgeInsets.all(4),
      padding: EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.blue.withOpacity(0.6),
        borderRadius: BorderRadius.circular(10),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            spreadRadius: 2,
            blurRadius: 5,
          ),
        ],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          Icon(icon, size: 40, color: Colors.white),
          SizedBox(height: 4),
          FittedBox(
            child: Text(label, style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
          ),
          FittedBox(
            child: Text(value, style: TextStyle(fontSize: 16, color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Widget _buildHourlyWeatherList() {
    return Container(
      height: 150,
      padding: EdgeInsets.all(8),
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: hourlyWeather.length,
        itemBuilder: (context, index) {
          var hourlyData = hourlyWeather[index];
          var time = DateTime.fromMillisecondsSinceEpoch(hourlyData['dt'] * 1000);
          var formattedTime = '${time.hour}:00';
          var pop = (hourlyData['pop'] * 100).toInt(); // 강수확률을 정수로 변환
          return Container(
            width: 80,
            margin: EdgeInsets.symmetric(horizontal: 4),
            child: Card(
              color: Colors.blue.withOpacity(0.6),
              elevation: 2,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
              child: Padding(
                padding: EdgeInsets.all(8.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    FittedBox(
                      child: Text(formattedTime, style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white, fontSize: 14)),
                    ),
                    if (hourlyData['weather'][0]['icon'] != null) Image.network(
                      'http://openweathermap.org/img/wn/${hourlyData['weather'][0]['icon']}@2x.png',
                      width: 40,
                      height: 40,
                    ),
                    FittedBox(
                      child: Text('${hourlyData['temp']}°C', style: TextStyle(color: Colors.white, fontSize: 12)),
                    ),
                    FittedBox(
                      child: Text(
                        '비: $pop%',
                        style: TextStyle(fontSize: 12, color: Colors.white),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }


  Widget _buildWeeklyForecast() {
    return Container(
      padding: EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '주간 날씨 예보',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white),
          ),
          SizedBox(height: 10),
          ListView.builder(
            shrinkWrap: true,
            physics: NeverScrollableScrollPhysics(),
            itemCount: dailyWeather.length,
            itemBuilder: (context, index) {
              var dailyData = dailyWeather[index];
              var dayOfWeek = DateTime.fromMillisecondsSinceEpoch(dailyData['dt'] * 1000).weekday;
              var weekDay = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'];
              var weatherId = dailyData['weather'][0]['id']; // 날씨 ID 추출
              var weatherDescriptionKo = weatherDescKo[weatherId] ?? '설명 없음'; // 맵핑된 한국어 설명 사용
              var pop = (dailyData['pop'] * 100).toInt(); // 강수확률을 정수로 변환

              return Card(
                color: Colors.blue.withOpacity(0.6),
                margin: EdgeInsets.symmetric(vertical: 4),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 8.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      // Left Section
                      Expanded(
                        flex: 2,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            FittedBox(
                              child: Text(
                                weekDay[dayOfWeek - 1],
                                style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white, fontSize: 14),
                              ),
                            ),
                            SizedBox(height: 5),
                            Row(
                              children: [
                                if (dailyData['weather'][0]['icon'] != null) Image.network(
                                  'http://openweathermap.org/img/wn/${dailyData['weather'][0]['icon']}@2x.png',
                                  width: 30,
                                  height: 30,
                                ),
                                SizedBox(width: 10),
                                Flexible(
                                  child: Text(
                                    weatherDescriptionKo, // 한글 날씨 설명
                                    style: TextStyle(color: Colors.white, fontSize: 12),
                                    overflow: TextOverflow.ellipsis,
                                    maxLines: 2,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      // Center Section
                      Expanded(
                        flex: 1,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.start,
                              children: [
                                Icon(Icons.thermostat, size: 16, color: Colors.white),
                                SizedBox(width: 4),
                                FittedBox(
                                  child: Text('${dailyData['temp']['day']}°C', style: TextStyle(color: Colors.white, fontSize: 12)),
                                ),
                              ],
                            ),
                            SizedBox(height: 10),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.start,
                              children: [
                                Icon(Icons.cloud, size: 16, color: Colors.white),
                                SizedBox(width: 4),
                                FittedBox(
                                  child: Text('${dailyData['clouds']}%', style: TextStyle(color: Colors.white, fontSize: 12)),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      // Right Section
                      Expanded(
                        flex: 1,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.start,
                              children: [
                                Icon(Icons.air, size: 16, color: Colors.white),
                                SizedBox(width: 4),
                                FittedBox(
                                  child: Text('${dailyData['wind_speed']} m/s', style: TextStyle(color: Colors.white, fontSize: 12)),
                                ),
                              ],
                            ),
                            SizedBox(height: 10),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.start,
                              children: [
                                Icon(Icons.water_drop, size: 16, color: Colors.white),
                                SizedBox(width: 4),
                                FittedBox(
                                  child: Text('강수확률: $pop%', style: TextStyle(color: Colors.white, fontSize: 12)),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.blue[700],
        title: Row(
          children: [
            Text(
              "날씨 정보 요약",
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.0,
              ),
            ),
            SizedBox(width: 8),
            Image.asset(
              'assets/weathy_reporter.jpg',
              width: 40,
              height: 40,
              fit: BoxFit.contain,
            ),
          ],
        ),
        actions: [
          Container(
            margin: EdgeInsets.only(right: 5), // 오른쪽 간격 추가
            child: PopupMenuButton<String>(
              onSelected: (String value) {
                setState(() {
                  if (value == '현재 위치') {
                    _getLocation();
                  } else {
                    latitude = cities[value][0];
                    longitude = cities[value][1];
                    cityName = value;
                    errorMessage = null; // 에러 메시지 초기화
                    _fetchWeatherData();
                  }
                });
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
              'assets/weather.jpg',  // 변경된 배경 이미지 경로
              fit: BoxFit.cover,
              color: Colors.black.withOpacity(0.1),
              colorBlendMode: BlendMode.darken,
            ),
          ),
          Center(
            child: isLoading
                ? CircularProgressIndicator()
                : errorMessage != null
                ? Text(errorMessage!, style: TextStyle(color: Colors.white))
                : ListView(
              children: [
                _buildWeatherCard(),
                _buildWeatherGrid(),
                _buildHourlyWeatherList(), // 24시간 날씨 예보 추가
                _buildWeeklyForecast()
              ],
            ),
          ),
        ],
      ),
    );
  }
}
