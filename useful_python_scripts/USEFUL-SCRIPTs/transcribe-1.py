import os
import time
import ffmpeg
import whisper

# 🔧 ВСТАВ СЮДИ ШЛЯХ ДО СВОГО ФАЙЛУ
file_path = r"report_plus.mp4"

def format_timestamp_simple(seconds):
    """Форматує секунди у формат HH:MM:SS (без мілісекунд)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def transcribe_to_text(file_path):
    if not os.path.isfile(file_path):
        print(f"Файл не знайдено: {file_path}")
        return

    start_time = time.time()

    print("Завантаження моделі...")
    model = whisper.load_model("medium", device="cuda")

    print("Триває транскрипція...")
    result = model.transcribe(file_path, language='uk', verbose=True)

    segments = result.get("segments", [])
    text_only = []
    text_with_timestamps = []

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        text_only.append(text)

        start_ts = format_timestamp_simple(start)
        end_ts = format_timestamp_simple(end)
        line = f"[{start_ts} --> {end_ts}]  {text}"
        text_with_timestamps.append(line)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_txt_path = os.path.join(".", f"{base_name}_ua.txt")
    output_log_path = os.path.join(".", f"{base_name}_with_timestamps.txt")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_only))

    with open(output_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_with_timestamps))

    print(f"\n✅ Текст без таймінгу збережено тут: {os.path.abspath(output_txt_path)}")
    print(f"✅ Текст з таймінгом збережено тут: {os.path.abspath(output_log_path)}")

    # ➕ Виведення додаткової інформації
    try:
        probe = ffmpeg.probe(file_path)
        duration = float(probe['format']['duration'])
        duration_fmt = format_timestamp_simple(duration)
        print(f"⏱️ Тривалість відео: {duration_fmt}")
    except Exception as e:
        print(f"⚠️ Не вдалося визначити тривалість відео: {e}")

    elapsed = time.time() - start_time
    elapsed_fmt = format_timestamp_simple(elapsed)
    print(f"🧠 Час транскрипції: {elapsed_fmt}")

transcribe_to_text(file_path)
