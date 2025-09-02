import whisper
import os

# 🔧 ВСТАВ СЮДИ ШЛЯХ ДО СВОГО ФАЙЛУ
file_path = r"D:\PROJECTs\MY\ScientificCalculations\SC\ScientificCalculations\useful_python_scripts\2025-05-19-12-01-19.mp4"

def transcribe_to_text(file_path):
    if not os.path.isfile(file_path):
        print(f"Файл не знайдено: {file_path}")
        return

    model = whisper.load_model("medium")  # можна змінити на "small" або "large"

    print("Триває транскрипція...")
    result = model.transcribe(file_path, language="uk")

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(os.path.dirname(file_path), f"{base_name}_ua.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Готово! Текст збережено тут: {output_path}")

transcribe_to_text(file_path)
