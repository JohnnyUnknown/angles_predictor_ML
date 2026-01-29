"""
Скрипт для извлечения признаков сопоставления изображений методом фазовой корреляции на сетке тайлов.

Назначение:
- Обрабатывает набор изображений с известными смещениями и поворотами из директории 'angles/images'
- Для каждой пары изображений (эталонное vs трансформированное):
  * Разбивает изображения на сетку 3×3 тайлов
  * Вычисляет фазовую корреляцию для каждого тайла
  * Определяет субпиксельные смещения методом взвешенного центроида
  * Рассчитывает метрики качества сопоставления (PCE, отклик)
- Сохраняет результаты в CSV-файлы с истинными и измеренными параметрами

Особенности обработки:
- Используется окно Ханна для подавления граничных эффектов в Фурье-преобразовании
- Субпиксельная точность достигается методом взвешенного центроида вокруг пика корреляции
- Метрика PCE (Peak to Correlation Energy) оценивает надёжность сопоставления
- Система координат: положительное смещение по X → вправо, по Y → вниз
- Важно: истинные смещения инвертируются в выводе (см. примечание в коде)

Структура выходных данных:
Каждая строка CSV содержит:
  - Истинные параметры: угол поворота, истинные смещения (с инверсией знака)
  - Для каждого из 9 тайлов (3×3 сетка):
      * Измеренное смещение по X/Y
      * PCE (отношение энергии пика к фону)
      * Максимальный отклик корреляции
"""

import os
import cv2
import numpy as np
import re
from sys import path


# Директории с данными
ANGLES_DIR = path[0] + "\\angles\\images"      # Входные изображения с трансформациями
PARAMS_DIR = path[0] + "\\angles\\parameters"  # Выходные CSV-файлы с параметрами

# Параметры обработки
img_size = (300, 300)   # Ожидаемый размер входных изображений (не используется напрямую)
grid_size = (3, 3)      # Размер сетки тайлов для локального анализа


def parse_shift_angle_from_filename(filename):
    """
    Извлекает истинные параметры трансформации из имени файла.
    
    Формат имени: "dx_dy_angle.jpg"
    Примеры: 
        "3p5_m2p0_1p50.jpg" → dx=+3.5, dy=-2.0, angle=1.50°
        "0p0_0p0_0p00.jpg"   → dx=0, dy=0, angle=0.00°
    
    Кодировка:
        'p' заменяет десятичную точку ('.')
        'm' заменяет минус ('-')
        Положительные значения без знака '+'
    
    :param filename: Имя файла (например, "3p5_m2p0_1p50.jpg")
    :return: Кортеж (dx, dy, angle) в пикселях/градусах или (None, None, None) при ошибке
    """
    name, _ = os.path.splitext(filename)
    parts = name.split('_')
    
    # Проверка структуры имени (должно быть ровно 3 компонента: dx, dy, angle)
    if len(parts) != 3:
        return None, None, None

    dx_str, dy_str, angle_str = parts


    def parse_component(s):
        """
        Преобразует закодированную строку в число с плавающей точкой.
        
        Поддерживаемые шаблоны:
          - "123"    → 123.0
          - "123p45" → 123.45
          - "m123"   → -123.0
          - "m123p45"→ -123.45
        
        :param s: Закодированная строка
        :return: Число с плавающей точкой или None при ошибке
        """
        # Валидация формата регулярным выражением
        if not re.fullmatch(r'm?\d+(p\d+)?', s):
            return None
        try:
            # Декодирование: замена 'p' → '.', 'm' → '-'
            s_clean = s.replace('p', '.')
            if s_clean.startswith('m'):
                s_clean = '-' + s_clean[1:]
            return float(s_clean)
        except Exception:
            return None

    # Парсинг всех трёх компонентов
    angle = parse_component(angle_str)
    dx = parse_component(dx_str)
    dy = parse_component(dy_str)

    # Проверка корректности всех значений
    if angle is None or dx is None or dy is None:
        return None, None, None

    return dx, dy, angle


def calculate_phase_correlation(img1, img2):
    """
    Вычисляет карту фазовой корреляции между двумя изображениями.
    
    Алгоритм:
      1. Применение окна Ханна для подавления граничных артефактов
      2. Вычисление 2D Фурье-образов обоих изображений
      3. Расчёт кросс-спектра с нормализацией фазы
      4. Обратное преобразование для получения карты корреляции
    
    Особенности:
      - Используется нормализация фазы (phase-only correlation) для устойчивости к изменениям освещения
      - Окно Ханна минимизирует утечку спектра
      - Результат сдвинут (fftshift) для центрирования нулевого сдвига в центре карты
    
    :param img1: Первое изображение (градации серого, 2D массив)
    :param img2: Второе изображение (того же размера)
    :return: Карта корреляции (2D массив, максимум соответствует сдвигу)
    """
    if img1.shape != img2.shape:
        raise ValueError("Изображения должны иметь одинаковые размеры")
    
    # Создание 2D окна Ханна для подавления граничных эффектов
    hann_window = np.outer(
        np.hanning(img1.shape[0]),
        np.hanning(img1.shape[1])
    )
    
    # Применение окна и преобразование в float32 для точности вычислений
    img1_float = img1.astype(np.float32) * hann_window
    img2_float = img2.astype(np.float32) * hann_window
    
    # Прямое Фурье-преобразование
    fft1 = np.fft.fft2(img1_float)
    fft2 = np.fft.fft2(img2_float)
    
    # Расчёт нормализованного кросс-спектра (phase-only correlation)
    cross_spectrum = np.conjugate(fft2) * fft1
    cross_spectrum_normalized = cross_spectrum / (np.abs(cross_spectrum) + 1e-10)
    
    # Обратное преобразование и центрирование нулевой частоты
    bfft = np.fft.ifft2(cross_spectrum_normalized)
    correlation = np.abs(np.fft.ifftshift(bfft))
    
    return correlation


def split_images_into_tiles(img1, img2, grid_size=(3, 3)):
    """
    Разбивает два изображения на одинаковые тайлы по сетке.
    
    :param img1: Первое изображение
    :param img2: Второе изображение
    :param grid_size: Размер сетки (строки, столбцы)
    :return: Кортеж (тайлы_изображения1, тайлы_изображения2, размер_тайла)
             Тайлы возвращаются как двумерные списки [ряд][столбец]
    """
    h, w = img1.shape[:2]
    
    # Расчёт размера тайла с учётом возможного неравномерного деления
    tile_h = h // grid_size[0]
    tile_w = w // grid_size[1]
    
    tiles1 = []
    tiles2 = []
    
    # Разбиение на тайлы с обработкой границ изображения
    for i in range(grid_size[0]):
        row_tiles1 = []
        row_tiles2 = []
        for j in range(grid_size[1]):
            # Расчёт координат тайла с коррекцией для последнего тайла в ряду/столбце
            y_start = i * tile_h
            y_end = (i + 1) * tile_h if i < grid_size[0] - 1 else h
            x_start = j * tile_w
            x_end = (j + 1) * tile_w if j < grid_size[1] - 1 else w
            
            # Извлечение тайлов из обоих изображений
            tile1 = img1[y_start:y_end, x_start:x_end]
            tile2 = img2[y_start:y_end, x_start:x_end]
            
            row_tiles1.append(tile1)
            row_tiles2.append(tile2)
        
        tiles1.append(row_tiles1)
        tiles2.append(row_tiles2)
    
    return tiles1, tiles2, (tile_h, tile_w)


def find_shift(correlation_map):
    """
    Находит позицию пика корреляции (грубая оценка сдвига).
    
    :param correlation_map: Карта корреляции (2D массив)
    :return: Кортеж (y, x) - координаты пика в пикселях
    """
    ind = np.unravel_index(np.argmax(correlation_map, axis=None), correlation_map.shape)
    return ind


def weighted_centroid(correlation, peak_loc, window_size=5):
    """
    Уточняет позицию пика методом взвешенного центроида (субпиксельная точность).
    
    Алгоритм:
      - Берёт окно вокруг грубого пика
      - Вычисляет центр масс значений корреляции в окне
      - Возвращает уточнённые координаты с дробной частью
    
    :param correlation: Карта корреляции
    :param peak_loc: Грубая позиция пика (y, x)
    :param window_size: Размер окна для уточнения (нечётное число)
    :return: Уточнённые координаты (x, y) с субпиксельной точностью
    """
    rows, cols = correlation.shape
    y_peak, x_peak = peak_loc
    
    # Расчёт границ окна с защитой от выхода за границы изображения
    half_window = window_size // 2
    y_start = max(0, y_peak - half_window)
    y_end = min(rows, y_peak + half_window + 1)
    x_start = max(0, x_peak - half_window)
    x_end = min(cols, x_peak + half_window + 1)
    
    # Извлечение окна и создание координатной сетки
    window = correlation[y_start:y_end, x_start:x_end]
    y_coords, x_coords = np.mgrid[y_start:y_end, x_start:x_end]
    
    # Вычисление центра масс
    total_weight = np.sum(window)
    if total_weight > 0:
        refined_y = np.sum(y_coords * window) / total_weight
        refined_x = np.sum(x_coords * window) / total_weight
    else:
        refined_y, refined_x = y_peak, x_peak
    
    return refined_x, refined_y  # Возвращаем в порядке (x, y) для удобства


def calculate_pce(correlation):
    """
    Рассчитывает метрику PCE (Peak to Correlation Energy). (Не имеет важности для синтетических данных).
    
    Формула:
        PCE = (peak_value²) / (средняя энергия фона)
    
    Где фон = все значения корреляции за исключением окрестности пика (5×5 пикселей).
    
    Интерпретация:
      - Высокий PCE (>100) → надёжное сопоставление
      - Низкий PCE (<10) → неуверенное сопоставление или шум
    
    :param correlation: Карта корреляции
    :return: Значение PCE (безразмерная величина)
    """
    # Поиск глобального максимума
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak_value = correlation[peak_idx]
    
    # Создание маски для выделения фона (всё кроме окрестности 5×5 вокруг пика)
    mask = np.ones_like(correlation, dtype=bool)
    h, w = correlation.shape
    r, c = peak_idx
    r_min, r_max = max(0, r - 2), min(h, r + 3)
    c_min, c_max = max(0, c - 2), min(w, c + 3)
    mask[r_min:r_max, c_min:c_max] = False
    
    # Расчёт энергии фона
    background = correlation[mask]
    background_energy = np.mean(background ** 2) + 1e-10  # Защита от деления на ноль
    
    # Вычисление PCE
    pce = (peak_value ** 2) / background_energy
    return pce


def calculate_correlation_for_tiles(tiles1, tiles2):
    """
    Вычисляет фазовую корреляцию для всех тайлов в сетке.
    
    :param tiles1: Двумерный список тайлов первого изображения [ряд][столбец]
    :param tiles2: Двумерный список тайлов второго изображения
    :return: Кортеж (объединённая_карта_корреляции, список_сдвигов_по_тайлам)
             Список сдвигов: двумерный список словарей с данными по каждому тайлу
    """
    grid_rows = len(tiles1)
    grid_cols = len(tiles1[0])
    
    # Хранение результатов для каждого тайла
    all_shifts = []
    
    # Получение размера корреляционной карты для одного тайла (для объединения)
    sample_tile1 = tiles1[0][0]
    sample_tile2 = tiles2[0][0]
    sample_correlation = calculate_phase_correlation(sample_tile1, sample_tile2)
    tile_corr_h, tile_corr_w = sample_correlation.shape
    
    # Создание объединённой карты для визуализации (опционально)
    combined_correlation = np.zeros((grid_rows * tile_corr_h, grid_cols * tile_corr_w))
    
    # Обработка каждого тайла в сетке
    for i in range(grid_rows):
        row_shifts = []
        for j in range(grid_cols):
            tile1 = tiles1[i][j]
            tile2 = tiles2[i][j]
            
            # Расчёт корреляции для текущего тайла
            correlation = calculate_phase_correlation(tile1, tile2)
            
            # Грубая оценка пика
            peak_location = find_shift(correlation)
            
            # Субпиксельное уточнение
            refined_peak = weighted_centroid(correlation, peak_location, 5)
            
            # Расчёт метрик качества
            response = np.max(correlation)
            pce = calculate_pce(correlation)
            
            # Сохранение результатов для тайла
            tile_shifts = {
                'tile_coords': (i, j),
                'custom_shift': refined_peak,      # Уточнённый сдвиг (x, y)
                'peak_location': peak_location,    # Грубый пик (y, x)
                'response': response,              # Максимальный отклик
                'PCE': pce                         # Peak to Correlation Energy
            }
            row_shifts.append(tile_shifts)
            
            # Заполнение объединённой карты (для отладки/визуализации)
            y_start = i * tile_corr_h
            y_end = (i + 1) * tile_corr_h
            x_start = j * tile_corr_w
            x_end = (j + 1) * tile_corr_w
            combined_correlation[y_start:y_end, x_start:x_end] = correlation
        
        all_shifts.append(row_shifts)
    
    return combined_correlation, all_shifts


def extract_tails_info(shifts):
    """
    Преобразует структуру данных о сдвигах тайлов в плоский словарь для CSV.
    
    Формат выходных ключей:
        "0_0_dx", "0_0_dy", "0_0_PCE", "0_0_resp", ... "2_2_dx", "2_2_dy", ...
    
    :param shifts: Двумерный список результатов по тайлам [ряд][столбец]
    :return: Плоский словарь с признаками для всех тайлов
    """
    tails_info = {}
    for i in range(len(shifts)):
        for j in range(len(shifts[i])):
            num = shifts[i][j]['tile_coords']
            tail_name = f'{num[0]}_{num[1]}'
            tails_info[tail_name + "_dx"] = shifts[i][j]['custom_shift'][0]
            tails_info[tail_name + "_dy"] = shifts[i][j]['custom_shift'][1]
            tails_info[tail_name + "_PCE"] = shifts[i][j]['PCE']
            tails_info[tail_name + "_resp"] = shifts[i][j]['response']
    return tails_info


def main():
    """
    Основной цикл обработки изображений.
    
    Алгоритм:
      1. Создание выходной директории для параметров
      2. Обход всех поддиректорий в 'angles/images':
          - Каждая поддиректория содержит кропы одного исходного изображения
      3. Для каждой поддиректории:
          - Загрузка эталонного изображения (0p0_0p0_0p00.jpg)
          - Нормализация яркости до диапазона [0, 255]
          - Обработка всех трансформированных изображений в папке:
              * Загрузка и нормализация
              * Разбиение на тайлы 3×3
              * Расчёт фазовой корреляции для каждого тайла
              * Извлечение признаков (сдвиги, PCE, отклик)
              * Парсинг истинных параметров из имени файла
          - Сохранение результатов в CSV с именем поддиректории
      4. Сортировка результатов по углу поворота для удобства анализа
    
    Важное примечание по знакам смещений:
      - В системе координат изображений положительное смещение по Y направлено ВНИЗ
      - При повороте изображения против часовой стрелки объект смещается ВВЕРХ на изображении
      - Поэтому истинные смещения инвертируются при сохранении (кроме нулевых):
          true_dx = -dx (если dx ≠ 0)
          true_dy = -dy (если dy ≠ 0)
      - Это обеспечивает соответствие между направлением поворота и измеренным смещением
    """
    # Создание директории для выходных параметров
    os.makedirs(PARAMS_DIR, exist_ok=True)
    
    # Обход всех поддиректорий в ANGLES_DIR (пропускаем корневую директорию)
    for root, dirs, files in os.walk(ANGLES_DIR):
        if root == ANGLES_DIR:
            continue

        # Поиск эталонного изображения (центральный кроп без трансформации)
        ref_path = os.path.join(root, "0p0_0p0_0p00.jpg")
        if not os.path.isfile(ref_path):
            print(f"Пропущена папка {root}: не найдено эталонное изображение 0p0_0p0_0p00.jpg")
            continue

        # Загрузка и нормализация эталонного изображения
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            print(f"Не удалось загрузить эталонное изображение: {ref_path}")
            continue
            
        # Нормализация яркости для стабильности вычислений
        ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX)

        results = []

        # Обработка всех изображений в текущей поддиректории (кроме эталона)
        for file in files:
            # Пропускаем эталонное изображение (обрабатываем только трансформированные)
            if file == "0p0_0p0_0p00.jpg":
                continue
                
            img_path = os.path.join(root, file)
            curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if curr_img is None:
                print(f"Не удалось загрузить {img_path}")
                continue

            # Нормализация текущего изображения
            curr_img = cv2.normalize(curr_img, None, 0, 255, cv2.NORM_MINMAX)

            try:
                # Разбиение изображений на тайлы
                tile1, tile2, _ = split_images_into_tiles(ref_img, curr_img, grid_size)
                
                # Расчёт корреляции и извлечение сдвигов для всех тайлов
                _, shifts = calculate_correlation_for_tiles(tile1, tile2)
                
                # Извлечение истинных параметров из имени файла
                true_dx, true_dy, angle = parse_shift_angle_from_filename(file)
                if true_dx is None:
                    print(f"Не удалось распарсить параметры из имени: {file}")
                    continue
                    
            except Exception as e:
                print(f"Ошибка обработки {file}: {e}")
                continue

            # Формирование записи результата
            res = extract_tails_info(shifts)  # Добавление признаков от всех тайлов
            res['angle'] = angle

            results.append(res)

        # Сортировка результатов по углу поворота для упорядоченного вывода
        results.sort(key=lambda x: x['angle'])

        # Формирование заголовка CSV с явным указанием всех столбцов
        header = (
            "0_0_dx,0_0_dy,0_0_PCE,0_0_resp,0_1_dx,0_1_dy,0_1_PCE,0_1_resp,0_2_dx,0_2_dy,0_2_PCE,0_2_resp,"
            "1_0_dx,1_0_dy,1_0_PCE,1_0_resp,1_1_dx,1_1_dy,1_1_PCE,1_1_resp,1_2_dx,1_2_dy,1_2_PCE,1_2_resp,"
            "2_0_dx,2_0_dy,2_0_PCE,2_0_resp,2_1_dx,2_1_dy,2_1_PCE,2_1_resp,2_2_dx,2_2_dy,2_2_PCE,2_2_resp,"
            "angle"
        )

        # Сохранение результатов в CSV-файл
        output_csv = os.path.join(PARAMS_DIR, f"{os.path.basename(root)}.csv")
        with open(output_csv, 'w', encoding='utf-8') as f:
            f.write(f"{header}\n")
            for r in results:
                # Форматирование числовых значений с фиксированной точностью
                f.write(
                    f"{r['0_0_dx']:.3f},{r['0_0_dy']:.3f},{r['0_0_PCE']:.3f},{r['0_0_resp']:.6f},"
                    f"{r['0_1_dx']:.3f},{r['0_1_dy']:.3f},{r['0_1_PCE']:.3f},{r['0_1_resp']:.6f},"
                    f"{r['0_2_dx']:.3f},{r['0_2_dy']:.3f},{r['0_2_PCE']:.3f},{r['0_2_resp']:.6f},"
                    f"{r['1_0_dx']:.3f},{r['1_0_dy']:.3f},{r['1_0_PCE']:.3f},{r['1_0_resp']:.6f},"
                    f"{r['1_1_dx']:.3f},{r['1_1_dy']:.3f},{r['1_1_PCE']:.3f},{r['1_1_resp']:.6f},"
                    f"{r['1_2_dx']:.3f},{r['1_2_dy']:.3f},{r['1_2_PCE']:.3f},{r['1_2_resp']:.6f},"
                    f"{r['2_0_dx']:.3f},{r['2_0_dy']:.3f},{r['2_0_PCE']:.3f},{r['2_0_resp']:.6f},"
                    f"{r['2_1_dx']:.3f},{r['2_1_dy']:.3f},{r['2_1_PCE']:.3f},{r['2_1_resp']:.6f},"
                    f"{r['2_2_dx']:.3f},{r['2_2_dy']:.3f},{r['2_2_PCE']:.3f},{r['2_2_resp']:.6f},"
                    f"{r['angle']:.2f}\n"
                )
                
        print(f"Обработано: {os.path.basename(root)} → {len(results)} изображений, сохранено в {os.path.basename(output_csv)}")

    print("\nВсе папки успешно обработаны!")


if __name__ == "__main__":
    main()