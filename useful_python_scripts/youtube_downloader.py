import yt_dlp
import os


def download_youtube_video():
    """
    Автоматично завантажує відео з YouTube у найкращій якості, завантажуючи відео та
    аудіо окремо, а потім з'єднуючи їх
    """
    # Фіксований URL та шлях для збереження
    video_url = "https://www.youtube.com/watch?v=yh71SZcReKI"
    save_path = "D://PROJECTs//MY//ScientificCalculations//SC//ScientificCalculations//useful_python_scripts"

    try:
        # Перевіряємо чи існує шлях для збереження
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Створено директорію {save_path}")

        print(f"Отримання інформації про відео з URL: {video_url}")

        # Отримуємо інформацію про всі доступні формати
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            video_title = info.get('title', 'відео')

        # Налаштування для завантаження найкращої якості відео та аудіо окремо і з'єднання їх
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',  # Спочатку спробує завантажити найкраще відео+найкраще аудіо, якщо не вдасться - найкращий комбінований формат
            'merge_output_format': 'mp4',  # Формат виходу після об'єднання
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
            'progress_hooks': [
                lambda d: print(f"Завантажено: {d['_percent_str']}" if d["status"] == "downloading" else
                                (f"Завершено завантаження {d['filename']}" if d["status"] == "finished" else ""))
            ],
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',  # Конвертуємо в mp4 для кращої сумісності
            }],
        }

        # Завантаження відео
        print("\nПочинаю завантаження відео у найкращій якості...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        print(f"\nВідео успішно завантажено в {save_path}")

        # Показуємо файли в директорії
        print("\nФайли в директорії:")
        try:
            all_files = os.listdir(save_path)
            downloaded_videos = [f for f in all_files if os.path.isfile(os.path.join(save_path, f)) and
                                 'найкраща_якість' in f]

            if downloaded_videos:
                for file in downloaded_videos:
                    file_path = os.path.join(save_path, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"Файл: {file} (Розмір: {file_size:.2f} МБ)")
                    print("Це відео найвищої якості з об'єднаним аудіо та відео потоками.")
            else:
                # Показуємо останні файли за датою модифікації
                files_with_time = [(f, os.path.getmtime(os.path.join(save_path, f)))
                                   for f in all_files if os.path.isfile(os.path.join(save_path, f))]
                files_with_time.sort(key=lambda x: x[1], reverse=True)

                print("Недавно створені/модифіковані файли:")
                for file, mtime in files_with_time[:5]:  # Показуємо тільки 5 останніх файлів
                    file_path = os.path.join(save_path, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"Файл: {file} (Розмір: {file_size:.2f} МБ)")
        except Exception as e:
            print(f"Помилка при перегляді файлів: {str(e)}")

    except Exception as e:
        print(f"Сталася помилка: {str(e)}")
        print("Деталі помилки:", e)


if __name__ == "__main__":
    download_youtube_video()
