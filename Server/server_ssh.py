from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import re
import tensorflow_datasets as tfds
import tensorflow as tf
import requests
import json
from datetime import datetime, timedelta, date
import numpy as np
import pytz
import os
from geopy.geocoders import Nominatim
import random
import folium
from folium import Popup

app = Flask(__name__)
CORS(app)

# 상대 경로로 변경
file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
train_data = pd.read_csv(file_path)

questions = []
for sentence in train_data['Q']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2
MAX_LENGTH = 40

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = float(d_model)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step_float)
        arg2 = step_float * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    if mask is not None:
        logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(CustomMultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        outputs = self.dense(concat_attention)
        return outputs

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

tf.keras.utils.get_custom_objects().update({'CustomSchedule': CustomSchedule})
tf.keras.utils.get_custom_objects().update({'loss_function': loss_function})
tf.keras.utils.get_custom_objects().update({'create_padding_mask': create_padding_mask})
tf.keras.utils.get_custom_objects().update({'create_look_ahead_mask': create_look_ahead_mask})
tf.keras.utils.get_custom_objects().update({'CustomMultiHeadAttention': CustomMultiHeadAttention})
tf.keras.utils.get_custom_objects().update({'PositionalEncoding': PositionalEncoding})

# 상대 경로로 변경
saved_model_path = os.path.join(os.path.dirname(__file__), 'transformer_v10')
loaded_model = tf.saved_model.load(saved_model_path)

def evaluate(sentence, model):
    sentence = preprocess_sentence(sentence)
    sentence = START_TOKEN + tokenizer.encode(sentence) + END_TOKEN
    sentence = tf.cast(sentence, dtype=tf.float32)
    sentence = tf.expand_dims(sentence, axis=0)
    output = tf.expand_dims(START_TOKEN + [tokenizer.vocab_size], axis=0)
    output = tf.cast(output, dtype=tf.float32)
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.equal(predicted_id, END_TOKEN[0]):
            break
        output = tf.concat([output, tf.cast(predicted_id, dtype=tf.float32)], axis=-1)
    return tf.squeeze(output, axis=0)

def predict(sentence, model):
    prediction = evaluate(sentence, model)
    prediction = [int(i) for i in prediction.numpy() if i < tokenizer.vocab_size]
    predicted_sentence = tokenizer.decode(prediction)
    return predicted_sentence

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def fetch_weather_data(lat, lon):
    api_key = '69688b165f9dc2376196570c34bdd33f'
    url_3_0 = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,alerts&appid={api_key}&units=metric'
    url_2_5 = f'http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={api_key}'
    
    response_3_0 = requests.get(url_3_0)
    response_2_5 = requests.get(url_2_5)
    
    if response_3_0.status_code == 200:
        data_3_0 = response_3_0.json()
    else:
        data_3_0 = None
        print(f'Failed to fetch weather data from API 3.0: {response_3_0.status_code}')
    
    if response_2_5.status_code == 200:
        data_2_5 = response_2_5.json()
    else:
        data_2_5 = None
        print(f'Failed to fetch weather data from API 2.5: {response_2_5.status_code}')
    
    return data_3_0, data_2_5

def print_daily_weather(data):
    daily_list = []
    if data:
        daily_weather = data['daily']
        for day in daily_weather:
            dt = day['dt']
            wet = day['summary']
            temp_day = day['temp']['day']
            temp_day_max = day['temp']['max']
            temp_day_min = day['temp']['min']
            wind_speed = day['wind_speed']
            day_rain = day.get('rain', 0)
            day_pop = str(float(day['pop']) * 100)
            uvi = day['uvi']
            day_pressure = day['pressure']
            day_humidity = day['humidity']
            day_snow = day.get('snow', 0)
            day_visibility = '측정값 없음'
            daily = {
                'dt': dt, 'wet': wet, 'temp_day': temp_day, 'temp_day_max': temp_day_max,
                'temp_day_min': temp_day_min, 'wind_speed': wind_speed, 'day_rain': day_rain,
                'day_pop': day_pop, 'uvi': uvi, 'day_pressure': day_pressure,
                'day_humidity': day_humidity, 'day_snow': day_snow, 'day_visibility': day_visibility
            }
            daily_list.append(daily)
    return daily_list



def senetence_completion(input, temp_3_0, temp_2_5, lat, lon):

    global avg

    api_key = '69688b165f9dc2376196570c34bdd33f'  # 발급받은 API 키로 교체하세요.
    weather_data_3_0, weather_data_2_5 = fetch_weather_data(lat, lon)
    daily_list = print_daily_weather(weather_data_3_0)
    daily_list = daily_list[:6]

    timezone = pytz.timezone('Asia/Seoul')
    now = datetime.now(timezone)
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    date_list_after = ['내일', '모레', '글피', '내일 모레', '1일 뒤', '2일 뒤', '3일 뒤', '4일 뒤', '5일 뒤']
    time_list = ['지금','현재','아침', '점심', '저녁', '오전', '오후', '낮', '밤', '새벽']
    hour_list = ['0시', '1시', '2시', '3시', '4시', '5시', '6시', '7시', '8시', '9시', '10시', '11시', '12시',
                 '13시', '14시', '15시', '16시', '17시', '18시', '19시', '20시', '21시', '22시', '23시', '24시']
    temp_need = []
    found_keyword = None
    for keyword in date_list_after:
        if keyword in input:
            found_keyword = keyword
            break
    day = 0
    if found_keyword:
        if found_keyword in ['지금', '현재']:
            day = 0
            daily = daily_list [0]
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+ 1 == int(temp_2_5['list'][i]['dt_txt'][8]+temp_2_5['list'][i]['dt_txt'][9]):

                    temp_need.append(temp_2_5['list'][i])
                    #temp_city.append(temp['city'][i])
        if found_keyword in ['내일', '1일 뒤']:
            day = 1
            daily = daily_list[1]
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+1 == int(temp_2_5['list'][i]['dt_txt'][8]+temp_2_5['list'][i]['dt_txt'][9]):
                    temp_need.append(temp_2_5['list'][i])
        if found_keyword in ['모레', '2일 뒤', '내일 모레']:
            day = 2
            daily = daily_list[2]
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+2 == int(temp_2_5['list'][i]['dt_txt'][8]+temp_2_5['list'][i]['dt_txt'][9]):
                    temp_need.append(temp_2_5['list'][i])
        if found_keyword in ['글피', '3일 뒤']:
            day = 3
            daily = daily_list[3]
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+3 == int(temp_2_5['list'][i]['dt_txt'][8]+temp_2_5['list'][i]['dt_txt'][9]):
                    temp_need.append(temp_2_5['list'][i])
        if found_keyword in ['4일 뒤']:
            day = 4
            daily = daily_list[4]
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+4 == int(temp_2_5['list'][i]['dt_txt'][8]+temp_2_5['list'][i]['dt_txt'][9]):
                    temp_need.append(temp_2_5['list'][i])
        if found_keyword in ['5일 뒤']:
            day = 5
            daily = daily_list[5]
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+5 == int(temp_2_5['list'][i]['dt_txt'][8]+temp_2_5['list'][i]['dt_txt'][9]):
                    temp_need.append(temp_2_5['list'][i])
    else:
        for i in range(40):
            daily = daily_list[0]
            if int(formatted_now[8]+formatted_now[9]) == int(temp_2_5['list'][i]['dt_txt'][8]+temp_2_5['list'][i]['dt_txt'][9]):
                temp_need.append(temp_2_5['list'][i])
    ine = 0
    near_time = []

    if (any(hour in input for hour in hour_list)):
        if ("오후" in input) or ("낮" in input) or ("점심" in input) or ("저녁" in input) or ("밤" in input):
            for i in range(len(temp_need)):
                matchs = re.search(r'(\d+)시', input)
                hour_str = matchs.group(1)
                hour = int(hour_str)
                if hour < 12:
                    hour += 12
                near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                near_time.append(abs(hour-near))
            ine = np.argmin(near_time)
        else:
            for i in range(len(temp_need)):
                matchs = re.search(r'(\d+)시', input)
                hour_str = matchs.group(1)
                hour = int(hour_str)
                near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                near_time.append(abs(hour-near))
            ine = np.argmin(near_time)
        i = ine
        timestamp = temp_need[i]['dt_txt']
        temperature = temp_need[i]['main']['temp']
        lowest_temperature = temp_need[i]['main']['temp_min']
        highest_temperature = temp_need[i]['main']['temp_max']
        humidity = temp_need[i]['main']['humidity']
        weather = temp_need[i]['weather'][0]['main']
        rain = temp_need[i]['rain']['3h'] if temp_need[i]['weather'][0]['main'] == 'rain' else 0
        rain_pop = str(float(temp_need[i]['pop']) * 100)
        uvi = "측정값 없음"
        visibility = temp_need[i]['visibility']
        wind = temp_need[i]['wind']['speed']
        pressure = temp_need[i]['main']['pressure']
        snow = temp_need[i]['snow']['3h'] if temp_need[i]['weather'][0]['main'] == 'snow' else 0
    else:
        if (any(time in input for time in time_list)):
            avg = 0 
            if ("지금" in input) or ("현재" in input):
            #(2) 시간대 키워드 값이 없는 경우 현재 시간과 가장 가까운 측정 시간
                for i in range(len(temp_need)):
                    now = int(formatted_now[11]+formatted_now[12])
                    near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                    #print('절대값',abs(now-near))
                    near_time.append(abs(now-near))
                #print(near_time)
                ine = np.argmin(near_time)
                # 질문한 시각과 가장 가까운 날씨 정보 값의 index
            elif ("오전" in input) or ("아침" in input):
                for i in range(len(temp_need)):
                    near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                    near_time.append((near))
                if 6 in near_time:
                    ine = near_time.index(6)
                elif 9 in near_time:
                    ine = near_time.index(9)
                elif 12 in near_time:
                    ine = near_time.index(12)
            elif ("낮" in input) or ("오후" in input) or ("점심" in input):
                for i in range(len(temp_need)):
                    near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                    near_time.append((near))
                if 15 in near_time:
                    ine = near_time.index(15)
                elif 12 in near_time:
                    ine = near_time.index(12)
            elif ("저녁" in input) or ("밤" in input):
                for i in range(len(temp_need)):
                    near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                    near_time.append((near))
                if 18 in near_time:
                    ine = near_time.index(18)
                elif 21 in near_time:
                    ine = near_time.index(21)
                elif 0 in near_time:
                    ine = near_time.index(0)
            elif ("새벽" in input):
                for i in range(len(temp_need)):
                    near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                    near_time.append((near))
                if 3 in near_time:
                    ine = near_time.index(3)
                elif 6 in near_time:
                    ine = near_time.index(6)
                elif 0 in near_time:
                    ine = near_time.index(0)
            i = ine
            timestamp = temp_need[i]['dt_txt']
            temperature = temp_need[i]['main']['temp']
            lowest_temperature = temp_need[i]['main']['temp_min']
            highest_temperature = temp_need[i]['main']['temp_max']
            humidity = temp_need[i]['main']['humidity']
            weather = temp_need[i]['weather'][0]['main']
            rain = temp_need[i]['rain']['3h'] if temp_need[i]['weather'][0]['main'] == 'rain' else 0
            rain_pop = str(float(temp_need[i]['pop']) * 100)
            uvi = "측정값 없음"
            visibility = temp_need[i]['visibility']
            wind = temp_need[i]['wind']['speed']
            pressure = temp_need[i]['main']['pressure']
            snow = temp_need[i]['snow']['3h'] if temp_need[i]['weather'][0]['main'] == 'snow' else 0
        else:
            today = date.today()
            days_later = today + timedelta(days=day)
            timestamp = days_later.strftime('%Y-%m-%d')
            temperature = daily['temp_day']
            lowest_temperature = daily['temp_day_min']
            highest_temperature = daily['temp_day_max']
            humidity = daily['day_humidity']
            weather = daily['wet']
            rain = daily['day_rain']
            rain_pop = daily['day_pop']
            uvi = daily['uvi']
            visibility = daily['day_visibility']
            wind = daily['wind_speed']
            pressure = daily['day_pressure']
            snow = 0

    rain_pop = int(float(rain_pop))
    
    return {
        "timestamp": timestamp,
        "temperature": temperature,
        "lowest_temperature": lowest_temperature,
        "highest_temperature": highest_temperature,
        "humidity": humidity,
        "weather": weather,
        "rain": rain,
        "rain_pop": rain_pop,
        "visibility": visibility,
        "wind": wind,
        "pressure": pressure,
        "snow": snow,
        "uvi": uvi
    }

def replace_time_in_output(input_str, output_str, time_list):
    for time_word in time_list:
        if time_word in input_str:
            for out_time_word in time_list:
                if out_time_word in output_str:
                    return output_str.replace(out_time_word, time_word)
    return output_str

def remove_space_before_punctuation(text):
    corrected_text = re.sub(r'(?<=\S) (?=[.,!?])', '', text)
    corrected_text = re.sub(r'([.,!?])(\s)(\s*)', r'\1\3', corrected_text)
    return corrected_text



# 7개의 리스트를 파이썬 list 문법으로 변환
temperature_list = ['사람마다 느끼는 체감온도는 다를 수 있습니다.', '온도가 높을 때는 열사병에 주의해주세요.', '대체로 온도가 높은 날이 유지됩니다.', '기온에 맞는 옷차림을 준비해주세요.', '더위를 많이 타신다면 가벼운 옷차림을 추천드려요.']
precipitation_list = ['지역마다 강수확률은 차이가 있을 수 있습니다.', '강수확률은 같은 조건일 때 과거에 비가 몇 번 왔다는 것을 의미합니다.', '강수확률이 있다면 습도가 평소보다 높을 수 있습니다.', '강수량이 높다면 실내 창문을 닫아주세요.', '강수량이 높을 시에 보행과 운전에 주의하시기바랍니다.']
pressure_list = ['대기압은 공기의 무게로 한 표면에 가해지는 단위 면적당 힘으로 정의됩니다.', '주변 지역보다 대기압이 높다면 맑은 하늘을 보실 수 있습니다.', '주변 지역보다 대기압이 낮다면 하늘이 흐릴 수 있습니다.', '주변 지역과 기압 차이가 클 시 바람이 많이 불 수 있습니다.', '대기압의 기준은 주변 지역에 비해서 기압이 높은가 낮은가로 판단됩니다.']
humidity_list = ['높은 습도가 지속된다면 곰팡이 등을 주의해주세요.', '습도가 높을 시 불쾌지수가 높을 수 있습니다.', '습도가 낮을 시 건조함을 느낄 수 있습니다.', '습도가 낮다면 화재에 주의하셔야합니다.', '적절한 실내 습도 유지가 중요합니다.']
visibility_list = ['가시거리는 다양한 조건에 의해 결정됩니다.', '가시거리가 낮을 시 건물 등이 잘 확인 되지 않을 수 있습니다.', '가시거리가 낮다면 보행과 운전에 주의해주세요.', '시정이 낮다면 운전 등에 힘이 들 수 있습니다.', '날씨와 가시거리 둘 다 좋다면 등산을 하시는 것도 좋습니다.']
snowfall_list = ['눈이 조금 쌓였어도 마찰력이 크게 줄어드니 보행과 운전에 주의해주세요.', '눈이 많이 올 경우 도로상황에 지장을 줄 수 있으니 주의해주세요.', '눈이 많이 왔다면 외출에 주의해주세요.', '눈이 올 때는 따뜻한 호빵 어떠세요?', '눈이 오면 시야에 방해가 있을 수 있습니다.']
wind_speed_list = ['바람이 많이 불 경우 체온이 급격히 낮아질 수 있습니다.', '적절한 바람은 불쾌지수를 낮춰줍니다.', '바람이 많이 분다면 적절한 겉옷을 챙겨주세요.', '풍속 14m/s부터 강풍주의보가 발령됩니다.', '풍향은 바람이 불어오는 방향을 나타냅니다.']

# output2 문자열에서 @@공통@@ 패턴을 랜덤한 리스트 요소로 바꾸는 코드
def replace_common_patterns(output2):
    if "@@공통온도@@" in output2:
        x1 = random.choice(temperature_list) + "🌡"
        output2 = output2.replace("@@공통온도@@", "\n" + x1)
    if "@@공통강수@@" in output2:
        x2 = random.choice(precipitation_list) + "☂"
        output2 = output2.replace("@@공통강수@@","\n" +  x2)
    if "@@공통대기압@@" in output2:
        x3 = random.choice(pressure_list) + "☁"
        output2 = output2.replace("@@공통대기압@@", "\n" + x3)
    if "@@공통습도@@" in output2:
        x4 = random.choice(humidity_list) + "💧"
        output2 = output2.replace("@@공통습도@@", "\n" + x4)
    if "@@공통가시거리@@" in output2:
        x5 = random.choice(visibility_list) + "🌟"
        output2 = output2.replace("@@공통가시거리@@","\n" +  x5)
    if "@@공통적설량@@" in output2:
        x6 = random.choice(snowfall_list) + "❄"
        output2 = output2.replace("@@공통적설량@@","\n" + x6)
    if "@@공통풍속@@" in output2:
        x7 = random.choice(wind_speed_list) + "🌫"
        output2 = output2.replace("@@공통풍속@@", "\n" + x7)
    return output2

good_food = { "수박": "여름 대표 제철 과일입니다. 당도와 수분이 풍부합니다.", "포도": "제철 과일입니다. 씨없는 포도도 있습니다!", "토마토": "수분이 풍부하고 건강에 좋습니다!", "매실": "제철 음식입니다. 여름에 배탈이 나면 매실차를 먹으라는 말도 있습니다.", "참외": "제철 과일입니다. 식감이 뛰어납니다.", "복숭아": "여름 대표 제철 과일입니다. 당도와 맛이 훌륭합니다.", "갈치조림": "제철 요리입니다. 제주도에서 갈치가 유명합니다.", "전복": "대표적인 보양식이며, 익혀먹는 것이 더욱 좋습니다.", "추어탕": "여름철 보양식입니다. 지역마다 특징이 다르다고 합니다.", "장어": "대표 보양식 입니다. 원기 회복에 뛰어납니다.", "삼계탕": "대표 보양식 입니다. 원기 회복에 뛰어납니다.", "비빔밥": "한국의 대표적인 음식입니다. 소화가 편하고 사계절 먹습니다.", "오리고기": "활동량이 많은 여름에 불포화 지방산이 피로회복에 도움을 줍니다.", "비빔국수": "시원하고 매운 양념이 입맛을 돋움니다.", "물냉면": "차가운 음식으로 열감 해소에 좋습니다.", "비빔냉면": "시원하고 매운 양념이 입맛을 돋움니다.", "열무국수": "열무를 곁들인 시원한 음식으로 열감 해소에 좋습니다.", "화채": "차가운 음식으로 열감 해소에 좋고, 디저트로 많이 먹습니다.", "냉채족발": "열감해소와 보양에 도움이 됩니다.", "콩국수": "여름을 대표하는 계절 메뉴입니다. 고소한 맛이 일품입니다.", "막국수": "물막국수, 비빔막국수 등 다양하며 일부 지역마다 먹는 방법이 재미있는 여름 요리입니다.", "냉모밀": "차가운 음식으로 열감 해소에 도움이 됩니다.", "밀면": "차가운 음식으로 열감 해소에 도움이 됩니다. 부산에서 유명합니다.", "오이냉국": "시원한 음식으로 여름철 밥상에 자주 등장합니다.", "아이스크림": "열감해소와 당분보풍에 뛰어난 간식입니다.", "팥빙수": "여름 대표 간식입니다. 다양한 빙수들이 있습니다." }
bad_food = { "생선회": "여름철 회는 상하기 쉬워 주의해야합니다.", "활어회": "여름철 회는 상하기 쉬워 주의해야합니다.", "전복회": "여름철 회는 상하기 쉬워 주의해야합니다. 익혀먹는 것이 좋습니다.", "생굴": "여름철 생굴은 상하기 쉬워 주의해야합니다.", "육회": "여름철 회는 상하기 쉬워 주의해야합니다.", "날고기": "여름철 날고기는 상하기 쉬워 주의해야합니다. 익혀먹는 것이 좋습니다.", "날달걀": "여름철 날달걀은 상하기 쉬워 주의해야합니다.", "채소샐러드": "생채소가 섞여 있어 어떤 음식으로 탈이 났는지 알기 어렵습니다.", "유제품": "유제품은 상하기 쉽습니다.", "생크림 케이크": "유제품은 상하기 쉽습니다.", "푸딩": "유제품은 상하기 쉽습니다.", "조개구이": "해산물, 특히 조개는 여름철 반드시 익혀 먹어야 합니다. 또한, 상한 조개가 아닌지 확인하세요.", "두부조림": "두부는 상하기 쉬워요", "마요네즈": "마요네즈는 상하기 쉬워요", "에그마요": "마요네즈는 상하기 쉬워요", "간장게장": "게장과 젓갈류는 상하기 쉬워요", "양념게장": "게장과 젓갈류는 상하기 쉬워요", "굴젓": "게장과 젓갈류는 상하기 쉬워요", "낙지젓갈": "게장과 젓갈류는 상하기 쉬워요", "갈치젓갈": "게장과 젓갈류는 상하기 쉬워요", "물회": "날것의 회를 사용하기에 신선한지 잘 확인해야합니다.", "회덮밥": "날것의 회를 사용하기에 신선한지 잘 확인해야합니다." }

def get_nearby_convenience_stores(lat, lon, api_key):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": 1000,  # 반경 1km 내 검색
        "type": "convenience_store",
        "key": api_key
    }
    response = requests.get(url, params=params)
    return response.json()

def plot_convenience_stores(stores, lat, lon):
    # 지도 생성 (줌 레벨을 17로 설정, 현재 위치를 중심으로 설정)
    map_ = folium.Map(location=[lat, lon], zoom_start=16)

    # 현재 위치 마커 추가
    current_location_popup = """
    <div style="white-space: nowrap; font-size: 12px; line-height: 0.5;">
        현재 위치
    </div>
    """
    folium.Marker(
        location=[lat, lon],
        popup=Popup(current_location_popup, show=True),
        icon=folium.Icon(color='blue')
    ).add_to(map_)

    # 편의점 마커 추가
    for store in stores['results']:
        # 간략한 팝업 내용
        brief_popup_content = f"""
        <div style="white-space: nowrap; font-size: 10px; line-height: 0.5;">
            <b>{store['name']}</b>
        </div>
        """
        detailed_popup_content = f"""
        <div style="white-space: nowrap; font-size: 10px; line-height: 0.5;">
            <b>{store['name']}</b><br>
            {store['vicinity']}
        </div>
        """

        # 마커 추가 (클릭 시 상세 내용 표시)
        marker = folium.Marker(
            location=[store['geometry']['location']['lat'], store['geometry']['location']['lng']],
            popup=Popup(brief_popup_content, show=True),
            icon=folium.Icon(color='red')
        ).add_to(map_)

        # JavaScript를 사용하여 클릭 시 상세 주소를 표시
        marker.add_child(folium.Popup(detailed_popup_content))
        marker.add_child(folium.ClickForMarker(popup=detailed_popup_content))

    return map_

@app.route('/greet', methods=['POST'])
def greet():

    avg=0

    index=0
    data = request.get_json()
    user_input = data['input']
    if '엔' in user_input: user_input = user_input.replace('엔', '')
    if '요즘에는' in user_input: user_input = user_input.replace('요즘에는', '')
    if '지금은' in user_input: user_input = user_input.replace('지금은', '지금')
    if '낼' in user_input: user_input = user_input.replace('낼', '내일')
    if '어떤데' in user_input: user_input = user_input.replace('어떤데', '어때')
    if '더운데' in user_input: user_input = user_input.replace('더운데', '더워')



    latitude = data['latitude']
    longitude = data['longitude']



    temp_3_0, temp_2_5 = fetch_weather_data(latitude, longitude)
    output1 = predict(user_input, loaded_model)
    
    
    output1 = remove_space_before_punctuation(output1)
    output1 = replace_time_in_output(user_input, output1, ['{아침}', '{점심}', '{저녁}', '{오전}', '{오후}', '{낮}', '{밤}', '{새벽}'])

    if '지금' in user_input:
        output1 = output1.replace("오늘", "지금")
    elif '현재' in user_input:
        output1 = output1.replace("오늘", "현재")


    if "{" in output1 or "@@" in output1:
        output2 = output1
        weather_data = senetence_completion(output1, temp_3_0, temp_2_5, latitude, longitude)
        if "{온도}" in output1:
            if avg == 0: output2 = output2.replace("{온도}", str(weather_data["temperature"])+"℃")
            else: output2 = output2.replace("{온도}", "평균 " + str(weather_data["temperature"])+"℃")
        if "{최저온도}" in output1:
            output2 = output2.replace("{최저온도}", str(weather_data["lowest_temperature"])+"℃")
        if "{최고온도}" in output1:
            output2 = output2.replace("{최고온도}", str(weather_data["highest_temperature"])+"℃")
        if "{습도}" in output1:
            if avg == 0: output2 = output2.replace("{습도}", str(weather_data["humidity"])+"%")
            else: output2 = output2.replace("{습도}", "평균 " + str(weather_data["humidity"])+"%")
        if "{날씨}" in output1:
            if weather_data["weather"] == 'Clear': weather_data["weather"] = '맑은 날씨'
            elif weather_data["weather"] == 'Rain': weather_data["weather"] = '비오는 날씨'
            elif weather_data["weather"] == 'Clouds': weather_data["weather"] = '흐린 날씨'
            elif weather_data["weather"] == 'Snow': weather_data["weather"] = '눈오는 날씨'
            elif weather_data["weather"] == 'Expect a day of partly cloudy with clear spells':
                weather_data["weather"] = '맑은 가운데 부분적으로 흐린 날씨'
            elif weather_data["weather"] == 'The day will start with clear sky through the late morning hours, transitioning to partly cloudy':
                weather_data["weather"] = '오전 늦게까지 맑은 하늘로 시작하여, 부분적으로 흐려지는 날씨'
            elif weather_data["weather"] == 'Expect a day of partly cloudy with rain':
                weather_data["weather"] = '일부 지역이 흐리고 비가 오는 날씨'
            elif weather_data["weather"] == 'There will be clear sky today':
                weather_data["weather"] = '하늘이 맑은 날씨'
            elif weather_data["weather"] == 'You can expect clear sky in the morning, with partly cloudy in the afternoon':
                weather_data["weather"] = '오전에는 맑고, 오후에는 부분적으로 흐린 날씨'
            elif weather_data["weather"] == 'There will be partly cloudy today':
                weather_data["weather"] = '아침에는 맑고, 오후에는 일부 지역이 흐린 날씨'
            elif weather_data["weather"] == 'The day will start with partly cloudy through the late morning hours, transitioning to clearing':
                weather_data["weather"] = '오전 늦게까지 흐리다 맑은 날씨'
            output2 = output2.replace("{날씨}", weather_data["weather"])
        if "{강수량}" in output1:
            if avg == 0: output2 = output2.replace("{강수량}", str(weather_data["rain"])+"mm")
            else: output2 = output2.replace("{강수량}", "평균 " + str(weather_data["rain"])+"mm")
        if "{강수확률}" in output1:
            output2 = output2.replace("{강수확률}",  str(weather_data["rain_pop"])+"%")
        if "{가시거리}" in output1:
            if avg == 0: output2 = output2.replace("{가시거리}", str(weather_data["visibility"])+"m")
            else: output2 = output2.replace("{가시거리}", "평균 " + str(weather_data["visibility"])+"m")
        if "{풍속}" in output1:
            if avg == 0: output2 = output2.replace("{풍속}", str(weather_data["wind"])+"m/s")
            else: output2 = output2.replace("{풍속}", "평균 " + str(weather_data["wind"])+"m/s")
        if "{대기압}" in output1:
            if avg == 0: output2 = output2.replace("{대기압}", str(weather_data["wind"])+"hPa")
            else: output2 = output2.replace("{대기압}", "평균 " + str(weather_data["wind"])+"hPa")
        if "{적설량}" in output1:
            if avg == 0: output2 = output2.replace("{적설량}", str(weather_data["snow"])+"mm")
            else: output2 = output2.replace("{적설량}", "평균 " + str(weather_data["snow"])+"mm")
        output2 = output2.replace("{", "").replace("}", "")

        # 2차 공통 템플릿 적용
        output2 = replace_common_patterns(output2)

        weather_data["rain_pop"] = float(weather_data["rain_pop"])
        weather_data["rain"] = float(weather_data["rain"])
        
        # 3차 날씨 세부 템플릿 적용
        if "@@비예보@@" in output2:
            if weather_data["rain"] == 0: output2 = output2.replace("@@비예보@@", "\n비가 오지 않습니다.")
            elif weather_data["rain"] < 0.01: output2 = output2.replace("@@비예보@@", "\n체감상으로 느껴지지 않는 빗방울입니다.")
            elif weather_data["rain"] >= 0.01 and weather_data["rain"] < 0.03: output2 = output2.replace("@@비예보@@", "\n옷이 젖지 않을 정도의 강수입니다.")
            elif weather_data["rain"] >= 0.03 and weather_data["rain"] < 0.15: output2 = output2.replace("@@비예보@@", "\n지면에 빗방울이 튀어 발밑이 젖을 수 있습니다.")
            elif weather_data["rain"] >= 0.15 and weather_data["rain"] < 0.3: output2 = output2.replace("@@비예보@@", "\n하수도가 넘칠 수 있으니 주의가 필요합니다.\n\n⭐웨디의 꿀팁⭐ 강한 비가 예상되니 창문을 닫는 것을 추천드립니다.")
            elif weather_data["rain"] >= 0.3: output2 = output2.replace("@@비예보@@", "\n해당 비가 3시간동안 지속될 경우 호우주의보가 발령됩니다. 주의가 필요합니다.\n\n⭐웨디의 꿀팁⭐ 비가 많이 올 때엔 자주 TV등을 통해 기상상황을 확인하시는 것이 좋습니다.")

            output2 = output2 + "\n\n우산이나 우의가 없으신 경우를 대비하여 근방의 편의점을 지도로 안내해드리겠습니다!"

            api_key = 'AIzaSyC-4c5H0cNBdele6suDFNBuec-155A7Guo'
            
            # 현재 위치와 편의점 데이터를 사용하여 지도 생성
            convenience_stores = get_nearby_convenience_stores(latitude, longitude, api_key)
            map_ = plot_convenience_stores(convenience_stores, latitude, longitude)
            
            # 지도 저장
            map_.save("convenience_stores_map.html")
            index = 1

        if "@@우의@@" in output2:
            if weather_data["rain_pop"] == 0: output2 = output2.replace("@@우의@@", "\n비가 오지 않습니다.")
            elif weather_data["rain_pop"] < 30: output2 = output2.replace("@@우의@@", "\n가벼운 외출이라면 우산이나 우의를 챙기지 않아도 괜찮습니다.")
            elif weather_data["rain_pop"] >= 30 and weather_data["rain_pop"] < 50: output2 = output2.replace("@@우의@@", "\n작은 단우산을 가방에 넣고 가시는 걸 추천드립니다.")
            elif weather_data["rain_pop"] >= 50 and weather_data["rain_pop"] < 70: output2 = output2.replace("@@우의@@", "\n우산이나 우의를 챙기시는 게 좋습니다.\n\n⭐웨디의 꿀팁⭐ 비가 올 때는 창문을 닫는게 좋습니다.")
            elif weather_data["rain_pop"] >= 70 and weather_data["rain_pop"] < 90: output2 = output2.replace("@@우의@@", "\n비가 올 확률이 높으므로 반드시 우산이나 우의를 챙겨주세요.\n\n⭐웨디의 꿀팁⭐ 비가 많이 올 때는 보행해 주의하셔야 합니다.")
            elif weather_data["rain_pop"] >= 90: output2 = output2.replace("@@우의@@", "\n비가 오고 있으므로 우의나 우산을 챙기시는 게 좋겠습니다.\n\n⭐웨디의 꿀팁⭐ 바람이 많이 분다면 우산보다는 우의가 더 안전합니다.")

            output2 = output2 + "\n\n우산이나 우의가 없으신 경우를 대비하여 근방의 편의점을 지도로 안내해드리겠습니다!"

            api_key = 'AIzaSyC-4c5H0cNBdele6suDFNBuec-155A7Guo'
            
            # 현재 위치와 편의점 데이터를 사용하여 지도 생성
            convenience_stores = get_nearby_convenience_stores(latitude, longitude, api_key)
            map_ = plot_convenience_stores(convenience_stores, latitude, longitude)
            
            # 지도 저장
            map_.save("convenience_stores_map.html")
            index = 1
        


        if "@@장화@@" in output2:
            if weather_data["rain_pop"] == 0: output2 = output2.replace("@@장화@@", "\n장화를 신지 않으시는 것이 좋을 것 같습니다.\n\n⭐웨디의 꿀팁⭐ 비가 오지 않는 여름엔 샌들을 신는 것도 좋습니다!")
            elif weather_data["rain_pop"] < 30: output2 = output2.replace("@@장화@@", "\n가벼운 외출이라면 장화보단 다른 신발을 추천합니다.")
            elif weather_data["rain_pop"] >= 30 and weather_data["rain_pop"] < 50: output2 = output2.replace("@@장화@@", "\n비가 올 수 있으나 장화보다는 다른 신발을 추천합니다.")
            elif weather_data["rain_pop"] >= 50 and weather_data["rain_pop"] < 70: output2 = output2.replace("@@장화@@", "\n좋아하시는 장화가 있으시다면 신는 것이 좋습니다.")
            elif weather_data["rain_pop"] >= 70 and weather_data["rain_pop"] < 90: output2 = output2.replace("@@장화@@", "\n비가 올 확률이 높으므로 장화를 신는 것도 좋겠습니다.\n\n⭐웨디의 꿀팁⭐ 비가 많이 온다면 길이가 긴 장화가 좋습니다.")
            elif weather_data["rain_pop"] >= 90: output2 = output2.replace("@@장화@@", "\n비가 올 확률이 높으므로 장화를 신는 것을 추천드립니다.\n\n⭐웨디의 꿀팁⭐ 장화를 신으면 통풍이 안 되니 너무 오래 신지 않는 것이 좋습니다.")

            output2 = output2 + "\n\n우산이나 우의가 없으신 경우를 대비하여 근방의 편의점을 지도로 안내해드리겠습니다!"

            api_key = 'AIzaSyC-4c5H0cNBdele6suDFNBuec-155A7Guo'
            
            # 현재 위치와 편의점 데이터를 사용하여 지도 생성
            convenience_stores = get_nearby_convenience_stores(latitude, longitude, api_key)
            map_ = plot_convenience_stores(convenience_stores, latitude, longitude)
            
            # 지도 저장
            map_.save("convenience_stores_map.html")
            index = 1

        if "@@빨래@@" in output2:
            if weather_data["rain_pop"] < 30: output2 = output2.replace("@@빨래@@","\n강수확률이 낮습니다. 필요시 빨래를 돌려도 될 것 같습니다.\n\n⭐웨디의 꿀팁⭐ 햇빛과 공기가 잘 통하는 곳에 빨래를 말려보세요!")
            elif weather_data["rain_pop"] >= 30 and weather_data["rain_pop"] < 50: output2 = output2.replace("@@빨래@@","\n강수확률이 있습니다. 빨래는 추천드리지 않습니다.\n\n⭐웨디의 꿀팁⭐ 소량의 빨래라면 괜찮을 것 같아요!")
            elif weather_data["rain_pop"] >= 50 and weather_data["rain_pop"] < 70: output2 = output2.replace("@@빨래@@","\n강수확률이 높은 편입니다. 빨래는 추천드리지 않습니다.")
            elif weather_data["rain_pop"] >= 70 and weather_data["rain_pop"] < 90: output2 = output2.replace("@@빨래@@","\n강수확률이 높습니다. 다른 날에 빨래를 하는 것을 추천드립니다.\n\n⭐웨디의 꿀팁⭐ 습도가 높을 가능성이 커 빨래를 돌리신다면 제습기를 틀어주세요.")
            elif weather_data["rain_pop"] >= 90: output2 = output2.replace("@@빨래@@","\n강수확률이 매우 높습니다. 빨래는 하면 안 될 것 같습니다.\n\n⭐웨디의 꿀팁⭐ 습도가 높을 가능성이 크니 온도가 낮지 않다면 에어컨을 트시는 것을 추천드립니다.")
        
        if "@@날씨@@" in output2:
            if weather_data["rain_pop"] < 30: output2 = output2.replace("@@날씨@@","\n강수확률이 낮습니다. 야외 활동을 하셔도 될 것 같습니다.")
            elif weather_data["rain_pop"] >= 30 and weather_data["rain_pop"] < 50: output2 = output2.replace("@@날씨@@","\n강수확률이 있으므로 가벼운 외출이 바람직합니다.")
            elif weather_data["rain_pop"] >= 50 and weather_data["rain_pop"] < 70: output2 = output2.replace("@@날씨@@","\n강수확률이 높은 편입니다. 야외활동 시 우산을 챙겨주세요.\n\n⭐웨디의 꿀팁⭐ 야외활동이 길다면 우산은 작고 가벼운 것이 좋습니다.")
            elif weather_data["rain_pop"] >= 70 and weather_data["rain_pop"] < 90: output2 = output2.replace("@@날씨@@","\n강수확률이 높으므로 야외활동보단 실내활동을 권장합니다.\n\n⭐웨디의 꿀팁⭐ 실내에서 영화보는 것은 어떠신가요?")
            elif weather_data["rain_pop"] >= 90: output2 = output2.replace("@@날씨@@","\n강수확률이 높으므로 가급적 실내에 있는 것이 쾌적할 것 같습니다.\n\n⭐웨디의 꿀팁⭐ 습도가 너무 높다면 제습기나 에어컨을 트시는 것을 추천합니다.")

        if "@@세차@@" in output2:
            if weather_data["rain_pop"] < 30: output2 = output2.replace("@@세차@@","\n강수확률이 낮습니다. 필요시 세차를 하셔도 될 것 같습니다.")
            elif weather_data["rain_pop"] >= 30 and weather_data["rain_pop"] < 50: output2 = output2.replace("@@세차@@","\n강수확률이 있습니다. 세차는 다음에 하시는 것을 추천드립니다.")
            elif weather_data["rain_pop"] >= 50 and weather_data["rain_pop"] < 70: output2 = output2.replace("@@세차@@","\n강수확률이 높은 편입니다. 세차장과 주차장이 실내로 연결되어있지 않는 한 다음에 해주세요.\n\n⭐웨디의 꿀팁⭐ 자동차 내부를 먼저 청소하는 것도 좋은 방법입니다.")
            elif weather_data["rain_pop"] >= 70 and weather_data["rain_pop"] < 90: output2 = output2.replace("@@세차@@","\n강수확률이 높습니다. 세차는 다음에 하시는 것을 추천드립니다.\n\n⭐웨디의 꿀팁⭐ 장마철에는 먼지가 쌓이는 것을 방지하기 위해 덮개를 사용하는 것도 좋습니다.")
            elif weather_data["rain_pop"] >= 90: output2 = output2.replace("@@세차@@","\n강수확률이 매우 높으므로 세차는 다른 날을 고려하셔야 합니다.\n\n⭐웨디의 꿀팁⭐ 가볍게 먼지를 턴 후 덮개를 사용하는 건 어떠실까요?")

        if "@@눅눅함@@" in output2:
            if weather_data["temperature"] < 18:
                if weather_data["humidity"] >= 75: output2 = output2.replace("@@눅눅함@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 눅눅함이 지속되는 장마철의 경우 방 안에 숯을 놓는 것도 좋습니다.")
                elif weather_data["humidity"] >= 65: output2 = output2.replace("@@눅눅함@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 65: output2 = output2.replace("@@눅눅함@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 주무실 때 젖은 수건을 옆에 두는 것을 추천합니다.")
            elif weather_data["temperature"] <= 20:
                if weather_data["humidity"] >= 65: output2 = output2.replace("@@눅눅함@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 곰팡이 예방을 위해 제습기나 에어컨을 트시는 것을 추천합니다.")
                elif weather_data["humidity"] >= 55: output2 = output2.replace("@@눅눅함@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 55: output2 = output2.replace("@@눅눅함@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 관엽 식물이 도움을 줄 수 있습니다.")
            elif weather_data["temperature"] <= 23:
                if weather_data["humidity"] >= 55: output2 = output2.replace("@@눅눅함@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 적절환 환기가 필요할 수 있습니다.")
                elif weather_data["humidity"] >= 45: output2 = output2.replace("@@눅눅함@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 45: output2 = output2.replace("@@눅눅함@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 가습기를 트는 것은 어떨까요?")
            elif weather_data["temperature"] > 23:
                if weather_data["humidity"] >= 45: output2 = output2.replace("@@눅눅함@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 환기를 하는 것을 추천드립니다.")
                elif weather_data["humidity"] >= 35: output2 = output2.replace("@@눅눅함@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 35: output2 = output2.replace("@@눅눅함@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 밤까지 건조하다면 물 한 컵을 옆에 두고 주무시는 것도 좋습니다.")

        if "@@건조함@@" in output2:
            if weather_data["temperature"] < 18:
                if weather_data["humidity"] >= 75: output2 = output2.replace("@@건조함@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 눅눅함이 지속되는 장마철의 경우 방 안에 숯을 놓는 것도 좋습니다.")
                elif weather_data["humidity"] >= 65: output2 = output2.replace("@@건조함@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 65: output2 = output2.replace("@@건조함@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 주무실 때 젖은 수건을 옆에 두는 것을 추천합니다.")
            elif weather_data["temperature"] <= 20:
                if weather_data["humidity"] >= 65: output2 = output2.replace("@@건조함@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 곰팡이 예방을 위해 제습기나 에어컨을 트시는 것을 추천합니다.")
                elif weather_data["humidity"] >= 55: output2 = output2.replace("@@건조함@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 55: output2 = output2.replace("@@건조함@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 관엽 식물이 도움을 줄 수 있습니다.")
            elif weather_data["temperature"] <= 23:
                if weather_data["humidity"] >= 55: output2 = output2.replace("@@건조함@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 적절환 환기가 필요할 수 있습니다.")
                elif weather_data["humidity"] >= 45: output2 = output2.replace("@@건조함@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 45: output2 = output2.replace("@@건조함@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 가습기를 트는 것은 어떨까요?")
            elif weather_data["temperature"] > 23:
                if weather_data["humidity"] >= 45: output2 = output2.replace("@@건조함@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 환기를 하는 것을 추천드립니다.")
                elif weather_data["humidity"] >= 35: output2 = output2.replace("@@건조함@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 35: output2 = output2.replace("@@건조함@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 밤까지 건조하다면 물 한 컵을 옆에 두고 주무시는 것도 좋습니다.")

        if "@@공기@@" in output2:
            if weather_data["temperature"] < 18:
                if weather_data["humidity"] >= 75: output2 = output2.replace("@@공기@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 눅눅함이 지속되는 장마철의 경우 방 안에 숯을 놓는 것도 좋습니다.")
                elif weather_data["humidity"] >= 65: output2 = output2.replace("@@공기@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 65: output2 = output2.replace("@@공기@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 주무실 때 젖은 수건을 옆에 두는 것을 추천합니다.")
            elif weather_data["temperature"] <= 20:
                if weather_data["humidity"] >= 65: output2 = output2.replace("@@공기@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 곰팡이 예방을 위해 제습기나 에어컨을 트시는 것을 추천합니다.")
                elif weather_data["humidity"] >= 55: output2 = output2.replace("@@공기@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 55: output2 = output2.replace("@@공기@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 관엽 식물이 도움을 줄 수 있습니다.")
            elif weather_data["temperature"] <= 23:
                if weather_data["humidity"] >= 55: output2 = output2.replace("@@공기@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 적절환 환기가 필요할 수 있습니다.")
                elif weather_data["humidity"] >= 45: output2 = output2.replace("@@공기@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 45: output2 = output2.replace("@@공기@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 가습기를 트는 것은 어떨까요?")
            elif weather_data["temperature"] > 23:
                if weather_data["humidity"] >= 45: output2 = output2.replace("@@공기@@", "\n습도가 높아 눅눅함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 환기를 하는 것을 추천드립니다.")
                elif weather_data["humidity"] >= 35: output2 = output2.replace("@@공기@@", "\n적정습도 범주입니다.")
                elif weather_data["humidity"] < 35: output2 = output2.replace("@@공기@@", "\n습도가 낮아 건조함을 느낄 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 밤까지 건조하다면 물 한 컵을 옆에 두고 주무시는 것도 좋습니다.")

        if "@@가습기@@" in output2:
            if weather_data["humidity"] < 40: output2 = output2.replace("@@가습기@@", "\n건조합니다. 가습기를 트는 것을 권해드립니다.\n\n⭐웨디의 꿀팁⭐ 가습기 필터는 주기적으로 가는 것이 좋습니다.")
            elif weather_data["humidity"] >= 40 and weather_data["humidity"] <= 60: output2 = output2.replace("@@가습기@@", "\n실내 적정 습도인 40-60% 사이 입니다.")
            elif weather_data["humidity"] > 60: output2 = output2.replace("@@가습기@@", "\n건조하지 않습니다. 가습기는 틀지 않으셔도 됩니다.")

        if "@@제습기@@" in output2:
            if weather_data["humidity"] < 40: output2 = output2.replace("@@제습기@@", "\n습하지 않습니다. 제습기는 틀지 않으셔도 됩니다.")
            elif weather_data["humidity"] >= 40 and weather_data["humidity"] <= 60: output2 = output2.replace("@@제습기@@", "\n실내 적정 습도인 40-60% 사이 입니다.")
            elif weather_data["humidity"] > 60: output2 = output2.replace("@@제습기@@", "\n실내 적정습도를 넘어섰습니다. 제습기를 트는 것을 권해드립니다.")

        if "@@곰팡이@@" in output2:
            if weather_data["humidity"] > 60: output2 = output2.replace("@@곰팡이@@", "\n실내 적정습도를 넘어섰습니다. 상태가 유지된다면 곰팡이가 필 수 있으므로 제습기를 사용해주세요.\n\n⭐웨디의 꿀팁⭐ 제습기가 없으시다면 숯을 이용하는 것도 방법입니다.")
            elif weather_data["humidity"] >= 40: output2 = output2.replace("@@곰팡이@@", "\n현재 실내 정적 습도인 40-60% 사이이나, 여름철에는 곰팡이에 유의해주세요.\n\n⭐웨디의 꿀팁⭐ 주변 환경에 따라 습하거나 건조할 수도 있습니다.")
            elif weather_data["humidity"] < 40: output2 = output2.replace("@@곰팡이@@", "\n곰팡이가 잘 생기지 않을 것으로 보이지만, 건조할 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 식물을 좋아하신다면 관엽식물을 키우시는 것도 방법입니다.")
        
        if "@@세탁기@@" in output2:
            if weather_data["humidity"] < 40: output2 = output2.replace("@@세탁기@@", "\n습하지 않으므로 세탁을 하셔도 됩니다.")
            elif weather_data["humidity"] >= 40 and weather_data["humidity"] <= 60: output2 = output2.replace("@@세탁기@@", "\n실내 적정 습도인 40-60% 사이 입니다. 세탁을 하셔도 됩니다.")
            elif weather_data["humidity"] > 60: output2 = output2.replace("@@세탁기@@", "\n실내 적정습도를 넘어섰습니다. 세탁을 하신다면 제습기나 에어컨으로 습도를 조절해주세요.")
        
        if "@@습도빨래@@" in output2:
            if weather_data["humidity"] > 60: output2 = output2.replace("@@습도빨래@@", "\n실내 적정습도를 넘어섰습니다. 빨래를 하신다면 제습기나 에어컨으로 습도를 조절해주세요.")
            elif 40 <= weather_data["humidity"] <= 60: output2 = output2.replace("@@습도빨래@@", "\n실내 적정 습도인 40-60% 사이 입니다. 빨래를 하셔도 됩니다.")
            elif weather_data["humidity"] < 40: output2 = output2.replace("@@습도빨래@@", "\n습하지 않으므로 빨래를 하셔도 됩니다.")
        
        if "@@바람@@" in output2:
            if weather_data["wind"] < 4: output2 = output2.replace("@@바람@@", "\n바람이 불지 않거나 잎사귀가 흔들릴 정도의 약한 바람이 불 수 있습니다.")
            elif weather_data["wind"] >= 4 and weather_data["wind"] < 7: output2 = output2.replace("@@바람@@", "\n체감상으로 시원할 정도의 바람이 붑니다.")
            elif weather_data["wind"] >= 7 and weather_data["wind"] < 9: output2 = output2.replace("@@바람@@", "\n사람의 따라 추위를 느낄 정도의 바람입니다.")
            elif weather_data["wind"] >= 9 and weather_data["wind"] < 14: output2 = output2.replace("@@바람@@", "\n나무 전체가 흔들리며 급격히 체온이 떨어질 수 있습니다.")
            elif weather_data["wind"] >= 14: output2 = output2.replace("@@바람@@", "\n강풍주의보가 발령될 수 있으니 주의가 필요합니다.")
            elif weather_data["wind"] >= 21: output2 = output2.replace("@@바람@@", "\n강풍경보가 발령될 수 있으니 물건 등의 주의가 필요합니다.")

        if "@@강풍@@" in output2:
            if weather_data["wind"] < 9: output2 = output2.replace("@@강풍@@", "\n강풍으로 인한 피해는 걱정하지 않으셔도 됩니다.")
            elif weather_data["wind"] >= 9 and weather_data["wind"] < 14: output2 = output2.replace("@@강풍@@", "\n나무 전체가 흔들리며 급격히 체온이 떨어질 수 있습니다.")
            elif weather_data["wind"] >= 14: output2 = output2.replace("@@강풍@@", "\n강풍주의보가 발령되니 주의가 필요합니다.")
            elif weather_data["wind"] >= 21: output2 = output2.replace("@@강풍@@", "\n강풍경보가 발령되니 물건 등의 주의가 필요합니다.")

        if "@@바람막이@@" in output2:
            if weather_data["wind"] >= 10: output2 = output2.replace("@@바람막이@@", "\n바람이 많이 불어 체온이 떨어질 수 있으니 바람막이를 챙기세요.")
            else: output2 = output2.replace("@@바람막이@@", "\n바람이 많이 불지 않겠으나 노약자와 어린이는 바람막이가 필요할 수 있습니다.")

        if "@@파도@@" in output2:
            if weather_data["wind"] <= 10: output2 = output2.replace("@@파도@@", "\n파도가 잔잔한 편으로 예상됩니다.")
            elif 10 < weather_data["wind"] <= 14: output2 = output2.replace("@@파도@@", "\n흰파도가 생기며 파도가 높아지기 시작합니다.")
            elif 14 < weather_data["wind"] <= 17: output2 = output2.replace("@@파도@@", "\n파장이 길어지고 마루의 끝이 거꾸로 된 파도가 생깁니다.\n\n⭐웨디의 꿀팁⭐ 해안가라면 보행이 힘드실 수 있습니다.")
            elif 18 < weather_data["wind"] <= 21: output2 = output2.replace("@@파도@@", "\n물거품이 강풍에 날리기 시작합니다.")
            elif weather_data["wind"] > 21: output2 = output2.replace("@@파도@@", "\n파도가 커지며 물보라 때문에 시정이 나빠집니다.\n\n⭐웨디의 꿀팁⭐ 바닷가에서 벗어나는 게 안전합니다.")

        if "@@눈예보@@" in output2:
            if weather_data["snow"] == 0: output2 = output2.replace("@@눈예보@@", "\n눈이 오지 않습니다.")
            elif weather_data["snow"] < 0.1: output2 = output2.replace("@@눈예보@@", "\n육안으로는 볼 수 있으나 적설량을 젤 수 없는 눈이 날립니다.")
            elif weather_data["snow"] < 0.2: output2 = output2.replace("@@눈예보@@", "\n도로와 일상 생활에 영향을 주지 않습니다.")
            elif weather_data["snow"] < 1: output2 = output2.replace("@@눈예보@@", "\n길에 얇은 눈의 층이 형성될 수 있습니다.")
            elif weather_data["snow"] < 5: output2 = output2.replace("@@눈예보@@", "\n가볍게 쌓인 눈을 볼 수 있으며 이동시 약간의 주의가 필요합니다.")
            elif weather_data["snow"] < 10: output2 = output2.replace("@@눈예보@@", "\n주의보가 발령됩니다. 일상생활에 영향을 줄 수 있으며 제설 작업이 필요합니다.")
            elif weather_data["snow"] < 20: output2 = output2.replace("@@눈예보@@", "\n일반지역까지 경가 발령됩니다. 가급적 실내에 머무는 것을 권장합니다.")
            elif weather_data["snow"] >= 20: output2 = output2.replace("@@눈예보@@", "\n눈이 오지 않습니다.")

        if "@@온도@@" in output2:
            if weather_data["temperature"] > 30: output2 = output2.replace("@@온도@@", "\n무더위입니다. 열사병에 주의하시고, 실내에서 활동하세요.\n\n⭐웨디의 꿀팁⭐ 선크림을 발라 자외선 노출을 막아주세요!")
            elif weather_data["temperature"] > 20: output2 = output2.replace("@@온도@@", "\n더운 날씨 입니다. 일교차가 심할 수 있습니다.\n\n⭐웨디의 꿀팁⭐ 땀이 많이 났다면 수분을 충분히 섭취해야 해요.")
            elif weather_data["temperature"] > 10: output2 = output2.replace("@@온도@@", "\n일교차가 심할 수 있습니다. 쌀쌀한 날씨입니다.")
            elif weather_data["temperature"] > 0: output2 = output2.replace("@@온도@@", "\n조금 추운 날씨 입니다. 외투를 신경쓰는 것이 좋습니다.")
            elif weather_data["temperature"] <= 0: output2 = output2.replace("@@온도@@", "\n추운 날씨 입니다. 감기 조심하세요.")

        if "@@난방@@" in output2:
            if weather_data["temperature"] < 10: output2 = output2.replace("@@난방@@", "\n추위가 느껴진다면 난방을 유지해주세요.")
            elif weather_data["temperature"] < 20: output2 = output2.replace("@@난방@@", "\n춥다고 늦겨지거나 온도가 낮을 경우 작게 난방을 트는 것도 괜찮습니다.")
            elif weather_data["temperature"] > 20: output2 = output2.replace("@@난방@@", "\n난방을 키기엔 기온이 높습니다.")

        if "@@냉방@@" in output2:
            if weather_data["temperature"] > 30: output2 = output2.replace("@@냉방@@", "\n열사병의 위험이 있으니 적절히 냉방을 해주세요.\n\n⭐웨디의 꿀팁⭐ 가급적 실내에 머물러주세요.")
            elif weather_data["temperature"] > 20: output2 = output2.replace("@@냉방@@", "\n온도가 높을 경우 작게 냉방을 트는 것도 괜찮습니다.")
            elif weather_data["temperature"] > 10: output2 = output2.replace("@@냉방@@", "\n냉방을 켜기엔 기온이 낮습니다.")

        if "@@에어컨@@" in output2:
            if weather_data["temperature"] > 30: output2 = output2.replace("@@에어컨@@", "\n열사병의 위험이 있으니 적절히 에어컨을 켜주세요.\n\n⭐웨디의 꿀팁⭐ 에너지 절약모드를 사용하시면 전기세 걱정을 덜 수 있습니다.")
            elif weather_data["temperature"] > 20: output2 = output2.replace("@@에어컨@@", "\n온도가 높을 경우 약하게 에어컨을 트는 것도 괜찮습니다.\n\n⭐웨디의 꿀팁⭐ 얇은 옷으로 갈아입으시는 것도 좋습니다.")
            elif weather_data["temperature"] > 10: output2 = output2.replace("@@에어컨@@", "\n에어컨을 켜기엔 기온이 낮습니다.")

        if "@@선풍기@@" in output2:
            if weather_data["temperature"] > 30: output2 = output2.replace("@@선풍기@@", "\n열사병의 위험이 있으니 선풍기를 켜주세요.\n\n⭐웨디의 꿀팁⭐ 얇은 옷으로 갈아입으시는 것도 좋습니다.")
            elif weather_data["temperature"] > 20: output2 = output2.replace("@@선풍기@@", "\n온도가 높을 경우 선풍기를 트는 것도 괜찮습니다.\n\n⭐웨디의 꿀팁⭐ 얇은 옷으로 갈아입으시는 것도 좋습니다.")
            elif weather_data["temperature"] > 10: output2 = output2.replace("@@선풍기@@", "\n선풍기를 켜기엔 기온이 낮습니다.")

        # 3차 음식 템플릿 적용
        if "@@시의성좋은음식@@" in output2:
            random_key, random_value = random.choice(list(good_food.items()))
            sentence = f"\n{random_key}은(는) {random_value} ╰(*°▽°*)╯"
            output2 = output2.replace(" 좋은음식"," "+ random_key)
            output2 = output2.replace("@@시의성좋은음식@@", sentence)
        
        if "@@시의성나쁜음식@@" in output2:
            random_key, random_value = random.choice(list(bad_food.items()))
            sentence = f"\n{random_key}은(는) {random_value} `(*>﹏<*)′"
            output2 = output2.replace("나쁜음식"," "+ random_key)
            output2 = output2.replace("@@시의성나쁜음식@@", sentence)
        
        if "@@일반성좋은음식@@" in output2:
            sentence = ""
            selected_items = random.sample(list(good_food.items()), 3)
            for food, description in selected_items:
                sentence += f"- {food}: {description}\n"
            output2 = output2.replace("@@일반성좋은음식@@", sentence + "╰(*°▽°*)╯")
        
        if "@@일반성나쁜음식@@" in output2:
            sentence = ""
            selected_items = random.sample(list(bad_food.items()), 3)
            for food, description in selected_items:
                sentence += f"- {food}: {description}\n"
            output2 = output2.replace("@일반성나쁜음식@@", sentence + "`(*>﹏<*)′")

        # 3차 옷 템플릿 적용
        if "@@온도별옷@@" in output2:
            if weather_data["temperature"] <= 15: output2 = output2.replace("@@온도별옷@@", "\n해당 기온에서의 추천 옷차림은 얇은 긴팔 티셔츠나, 가벼운 가디건이나 니트를 추천합니다. 때에 따라서 선선함을 느낄 수 있는 날씨입니다.")
            elif weather_data["temperature"] <= 25: output2 = output2.replace("@@온도별옷@@", "\n해당 기온에서의 추천 옷차림은 얇은 긴팔 셔츠, 면바지, 가벼운 재킷과 같이\n상대적으로 선선한 날씨에 대비해 약간의 보온 효과를 주면서도 통기성이 좋은 소재로 만든 옷을 입는 것이 좋습니다.")
            else: output2 = output2.replace("@@온도별옷@@", "\n해당 기온에서의 추천 옷차림은 반팔 셔츠, 반바지, 면이나 리넨 소재의 옷과 같이\n상대적으로 가볍고 통기성이 좋으며 땀을 흡수하고 빨리 마르는 소재의 옷이 좋습니다.")
        
        response = f"정보 기준: {weather_data['timestamp']}\n{output2}"

        if index==1:
            response = f"정보 기준: {weather_data['timestamp']}\n{output2}"
            index=0
    else:
        response = f"{output1.replace('{', '').replace('}', '')}"
        if "{강수확률}" in output1:
            output2 = output2.replace("{강수확률}",  str(weather_data["rain_pop"])+"%")
        if "좋고습니다." in output1:
            output1  = output1.replace("좋고습니다.","좋습니다.")
    return jsonify({'response': response})

@app.route('/convenience_stores_map')
def get_map():
    return send_from_directory('.', 'convenience_stores_map.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
 