# Используем официальный образ Python в качестве базового
FROM python:3.9-slim

# Устанавливаем зависимости для работы с Tkinter
RUN apt-get update && apt-get install -y \
    python3-tk \
    xvfb \
    && apt-get clean

# Устанавливаем необходимые Python библиотеки
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ваше приложение в контейнер
COPY . /app

# Указываем рабочую директорию внутри контейнера
WORKDIR /app

# Указываем права на выполнение для скрипта
RUN chmod +x app.py

# Указываем команду для запуска приложения
CMD ["xvfb-run", "python", "app.py"]



