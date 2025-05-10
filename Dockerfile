# Python 3.12.4 をベースに
FROM python:3.12.4-slim

# 環境変数（.pyc作らない、ログ即出力）
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 作業ディレクトリ作成
WORKDIR /app

# OSパッケージのインストール（最低限でOK）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt をコピーしてインストール
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# アプリ全体をコピー
COPY . .

# 静的ファイル収集（必要なら）
RUN python manage.py collectstatic --noinput || true

# ポート8000を開ける
EXPOSE 8000

# 開発用サーバー起動
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]