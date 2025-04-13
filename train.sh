#!/bin/bash

# Скрипт запуска обучения MNIST на чистом numpy и pandas с проверкой хэша зависимостей

VENV_DIR="venv"
HASH_FILE=".requirements.hash"
REQ_FILE="requirements.txt"

echo "=============================="
echo "  🧩 MNIST PROJECT BOOTSTRAP "
echo "=============================="

# 1. Проверяем и создаём виртуальное окружение, если нужно
if [ -d "$VENV_DIR" ]; then
    echo "✅ Виртуальное окружение '$VENV_DIR' уже существует."
else
    echo "🚀 Создаю виртуальное окружение '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
fi

# 2. Активируем виртуальное окружение
echo "🔑 Активирую виртуальное окружение..."
source $VENV_DIR/bin/activate

# 3. Проверяем, изменился ли requirements.txt
if [ -f "$REQ_FILE" ]; then
    NEW_HASH=$(sha256sum "$REQ_FILE" | awk '{print $1}')
    OLD_HASH=$(cat "$HASH_FILE" 2>/dev/null)

    if [ "$NEW_HASH" != "$OLD_HASH" ]; then
        echo "🔄 requirements.txt изменился или hash-файл отсутствует."
        echo "📦 Переустанавливаю зависимости..."

        pip install --upgrade pip
        pip install -r "$REQ_FILE"

        echo "$NEW_HASH" > "$HASH_FILE"
        echo "✅ Зависимости установлены и hash обновлён."
    else
        echo "✅ requirements.txt не изменился. Пропускаю установку зависимостей."
    fi
else
    echo "❌ Файл $REQ_FILE не найден. Пожалуйста, создайте его."
    deactivate
    exit 1
fi

# 4. Запуск main.py
echo "🏁 Запускаю процесс обучения модели на MNIST с numpy + pandas..."
python3 main.py

# 5. Деактивация окружения
echo "🛑 Завершаю сессию виртуального окружения."
deactivate

echo "✅ Готово! Скрипт завершён."
echo "Посмотри сохранённые веса модели и графики в директории проекта."
