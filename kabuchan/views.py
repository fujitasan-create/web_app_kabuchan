from django.shortcuts import render
from django.views import View as V
from django.conf import settings
import random
import os
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import datetime
import requests
import jaconv
from bs4 import BeautifulSoup


class IndexView(V):
    def get(self,request):
        return render(request,'kabuchan/index.html')
    
def about(request):
    return render(request, 'kabuchan/about.html')

class RecommendStockView(V):
    def get(self, request):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'data', 'nikkei_225_top70_stocks.csv')

        df = pd.read_csv(csv_path)
        stock = random.choice(df["銘柄名"].tolist())

        return render(request, 'kabuchan/recommend.html', {'stock': stock})

class PredictStockView(V):
    def get(self, request):
        stock_name = request.GET.get('stock_name')
        if stock_name:
            return render(request, 'kabuchan/predict_thinking.html', {'stock_name': stock_name})
        return render(request, 'kabuchan/predict.html')

    def post(self, request):
        stock_name = request.POST.get('stock_name')

        csv_path = 'kabuchan/data/codes.csv'
        code_df = pd.read_csv(csv_path)

        if stock_name not in code_df["銘柄名"].values:
            return render(request, "kabuchan/predict_result.html", {
                "stock_name": stock_name,
                "not_found": True
            })

        try:
            ticker = code_df.query("銘柄名 == @stock_name")["銘柄コード"].values[0]
        except IndexError:
            return render(request, 'kabuchan/predict.html', {
                'error': f"「{stock_name}」は見つかりませんでした。"
            })
        
        today = datetime.date.today().strftime("%Y-%m-%d")


        df = yf.download(ticker, start='2021-01-01', end=today)
        close = df['Close'].squeeze()
        df['target'] = ((df['Close'].shift(-1) / df['Close']) > 1.01).astype(int)

        def calc_macd(close, fast=12, slow=26, signal=9):
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line, macd - signal_line

        def calc_rsi(close, period=14):
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        def calc_bollinger_bands(close, period=25, num_std=2):
            ma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            return ma + num_std * std, ma, ma - num_std * std

        df['sma1'] = df['Close'].rolling(window=5).mean()
        df['sma2'] = df['Close'].rolling(window=25).mean()
        df['sma3'] = df['Close'].rolling(window=50).mean()
        df['macd'], df['macdsignal'], df['macdhist'] = calc_macd(close)
        df['RSI'] = calc_rsi(close, period=25)
        df['upper'], df['middle'], df['lower'] = calc_bollinger_bands(close, 25, 2)
        df['smax'] = df['sma1'] - df['sma2']
        df['BTY'] = df['High'] - df['Low']

        X = df[['Open', 'Close', 'BTY', 'smax', 'sma1', 'sma2', 'sma3',
                'macd', 'macdhist', 'macdsignal', 'RSI', 'upper', 'lower', 'Volume']]
        y = df['target']

        df_model = pd.concat([X, y], axis=1).dropna()
        X = df_model[X.columns]
        y = df_model['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
        smote = SMOTE(random_state=0)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        model = GradientBoostingClassifier(max_depth=5, n_estimators=200,
                                           subsample=0.8, learning_rate=0.10)
        model.fit(X_train_resampled, y_train_resampled)
        X_latest = X_scaled[-1].reshape(1, -1)  
        last_proba = model.predict_proba(X_latest)[0, 1]  

        threshold = 0.25
        pred = (last_proba >= threshold).astype(int)

        last_pred = "明日は上がるかも！📈" if last_proba >= threshold else "うーん…明日は厳しそう📉"

        return render(request, 'kabuchan/predict_result.html', {
            'stock_name': stock_name,
            'prediction': last_pred,
            'probability': f"{last_proba*100:.2f} %"
        })

def load_emotion_dict():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dict_path = os.path.join(base_dir, 'kabuchan', 'data', 'emotions')

    emotion_dict = {}
    for fname in os.listdir(dict_path):
        if fname.endswith("_uncoded.txt"):
            label = fname.replace("_uncoded.txt", "")
            with open(os.path.join(dict_path, fname), "r", encoding="utf-8") as f:
                words = [line.strip() for line in f if line.strip()]
                emotion_dict[label] = words
    return emotion_dict

emotion_dict = load_emotion_dict()

# --- 感情分析（テキスト1つに対して） ---
def analyze_emotion(text):
    text = jaconv.kata2hira(jaconv.z2h(text, kana=True, digit=True, ascii=True))
    result = {}
    for label, words in emotion_dict.items():
        hits = [w for w in words if w in text]
        if hits:
            result[label] = hits
    return {"text": text, "emotion": result if result else None}

# --- Yahooニュースから見出しを取得 ---
def fetch_yahoo_titles():
    url = 'https://news.yahoo.co.jp/categories/business'
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    titles = []
    articles = soup.find_all('a')

    for a in articles:
        href = a.get('href')
        if href and '/articles/' in href:
            title = a.get_text(strip=True)
            if title and len(title) > 5:
                titles.append(title)
        if len(titles) >= 10:  # 上位10件だけ使う
            break

    return titles

# --- Djangoビュー ---
class MarketMoodView(V):
    def get(self, request):
        titles = fetch_yahoo_titles()
        emotion_counter = {}

        for title in titles:
            result = analyze_emotion(title)
            if result["emotion"]:
                for k in result["emotion"]:
                    emotion_counter[k] = emotion_counter.get(k, 0) + 1

        # --- スコアを計算 ---
        pos = emotion_counter.get("喜", 0) + emotion_counter.get("好", 0)
        neg = emotion_counter.get("哀", 0) + emotion_counter.get("怒", 0) + emotion_counter.get("嫌", 0)

        if pos >= 5:
            score = 5
        elif pos >= 3:
            score = 4
        elif neg >= 5:
            score = 1
        elif neg >= 3:
            score = 2
        else:
            score = 3

        mood_dict = {
            1: ("とっても悲しい気分…📉", "very-sad-face"),
            2: ("ちょっと元気ないかも…😢", "sad-face"),
            3: ("ふつうかな〜🤔", "neutral-face"),
            4: ("いい感じかも！✨", "happy-face"),
            5: ("絶好調〜！📈🎉", "very-happy-face"),
        }

        mood, face_class = mood_dict[score]

        return render(request, "kabuchan/market_result.html", {
            "score": score,
            "mood": mood,
            "face_class": face_class,
        })


index=IndexView.as_view()
recommend=RecommendStockView.as_view()
predict=PredictStockView.as_view()
market_result = MarketMoodView.as_view()