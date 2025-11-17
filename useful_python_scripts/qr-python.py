import qrcode
from PIL import Image

# Посилання на вашу Google форму
google_form_url = "https://docs.google.com/forms/d/e/1FAIpQLScgY2lfK3BwB7WGRkX7cKU3bZybDv2r8rRnkiYf-DhEWu1Stw/viewform?usp=publish-editor"

# Створення QR-коду
qr = qrcode.QRCode(
    version=1,  # Розмір QR-коду (1-40, де 1 = найменший)
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # Рівень корекції помилок
    box_size=10,  # Розмір кожного квадратика
    border=4,  # Товщина рамки (мінімум 4)
)

# Додаємо дані (посилання) до QR-коду
qr.add_data(google_form_url)
qr.make(fit=True)

# Створюємо зображення
img = qr.make_image(fill_color="black", back_color="white")

# Зберігаємо QR-код як зображення
img.save("google_form_qr.png")

print("QR-код успішно створено та збережено як 'google_form_qr.png'")

# Опціонально: відкриваємо зображення для перегляду
img.show()