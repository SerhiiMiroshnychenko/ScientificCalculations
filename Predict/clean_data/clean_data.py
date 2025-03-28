import pandas as pd
import os


def replace_negative_values(input_file, output_file=None):
    """
    Прочитує CSV файл, замінює від'ємні значення на 0 і зберігає результат у новому файлі

    Args:
        input_file (str): Шлях до вхідного файлу
        output_file (str, optional): Шлях до вихідного файлу. Якщо не вказано,
                                     створюється автоматично з префіксом 'no_neg_'

    Returns:
        str: Шлях до вихідного файлу
    """
    # Якщо вихідний файл не вказано, створюємо автоматично
    if output_file is None:
        dir_path = os.path.dirname(input_file)
        file_name = os.path.basename(input_file)
        output_file = os.path.join(dir_path, f"no_neg_{file_name}")

    # Читаємо файл
    print(f"Читаємо файл: {input_file}")
    df = pd.read_csv(input_file)

    # Отримуємо список числових колонок
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    print(f"Знайдено {len(numeric_columns)} числових колонок")

    # Рахуємо загальну кількість від'ємних значень
    total_negative = 0

    # Замінюємо від'ємні значення на 0
    for col in numeric_columns:
        negative_count = (df[col] < 0).sum()
        total_negative += negative_count

        if negative_count > 0:
            print(f"Знайдено {negative_count} від'ємних значень у колонці '{col}'. Замінюємо їх на 0.")
            df[col] = df[col].apply(lambda x: max(0, x) if not pd.isna(x) else x)

    # Зберігаємо результат
    df.to_csv(output_file, index=False)
    print(f"\nВсього замінено {total_negative} від'ємних значень")
    print(f"Результат збережено у файл: {output_file}")

    return output_file


if __name__ == "__main__":
    # Шлях до вхідного файлу
    input_file = "cleaned_result_origin.csv"

    output_file = "cleaned_result.csv"

    # Запускаємо обробку
    replace_negative_values(input_file, output_file)