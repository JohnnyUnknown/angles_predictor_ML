"""
Скрипт для извлечения признаков сопоставления изображений методом фазовой корреляции на сетке тайлов.

ОСНОВНАЯ ЗАДАЧА:
Генерация обучающего датасета для предсказания угла поворота между изображениями на основе локальных
смещений, измеренных методом фазовой корреляции в разных регионах изображения.

АЛГОРИТМ ОБРАБОТКИ:
1. Для каждой пары изображений (эталонное vs трансформированное):
   - Разбиение на сетку 3×3 тайлов для локального анализа
   - Вычисление фазовой корреляции для каждого тайла
   - Определение субпиксельных смещений методом взвешенного центроида
   - Расчёт метрик качества (PCE, максимальный отклик)

2. Формирование признакового вектора:
   - 9 тайлов × 2 признака = 18 признаков на изображение:
     * Измеренное смещение по X (dx)
     * Измеренное смещение по Y (dy)
     * PCE (Peak to Correlation Energy) - убрано, не несет значимой информации
     * Максимальный отклик корреляции   - убрано, не несет значимой информации
   - Целевая переменная: угол поворота (градусы)

ОСОБЕННОСТИ РЕАЛИЗАЦИИ:
- Используется окно Ханна для подавления граничных артефактов в Фурье-пространстве
- Субпиксельная точность достигается методом взвешенного центроида (окно 5×5)
- Нормализация яркости изображений для устойчивости к вариациям освещения
- Все данные агрегируются в единый CSV-файл для удобства обучения моделей

ВАЖНОЕ ПРИМЕЧАНИЕ:
Для синтетических данных (идеальные трансформации без шума) метрика PCE имеет ограниченную
дискриминативную способность, так как корреляция всегда близка к идеальному пику. Основную
информацию для предсказания угла несут именно измеренные смещения (dx, dy) в разных регионах.
"""
import os
import cv2
import numpy as np
import pandas as pd
import re
from sys import path

# Директории с данными
ANGLES_DIR = path[0] + "\\angles\\images"      # Входные изображения с трансформациями

img_size = (300, 300)   # Размер входных кропов (пиксели)
grid_size = (3, 3)      # Сетка тайлов для локального анализа сопоставления


def parse_shift_angle_from_filename(filename):
    """
    Декодирует истинные параметры трансформации из имени файла.
    
    Синтаксис имени: "смещение_X_смещение_Y_угол.jpg"
    Примеры корректных имён:
      - "3p5_m2p0_1p50.jpg" → dx=+3.5, dy=-2.0, угол=+1.50°
      - "0p0_0p0_0p00.jpg"  → dx=0, dy=0, угол=0.00°
    
    Алгоритм декодирования:
      1. Разделение имени на 3 компонента по символу '_'
      2. Замена 'p' → '.' для восстановления десятичной точки
      3. Замена 'm' → '-' для восстановления отрицательного знака
      4. Преобразование в числовой формат
    
    :param filename: Строка с именем файла (с расширением)
    :return: Кортеж (dx: float, dy: float, angle: float) или (None, None, None) при ошибке
    """
    name, _ = os.path.splitext(filename)
    parts = name.split('_')
    
    # Валидация структуры имени (ровно 3 компонента)
    if len(parts) != 3:
        return None, None, None

    dx_str, dy_str, angle_str = parts

    def parse_component(s):
        """
        Преобразует закодированную строку в число с плавающей точкой.
        
        Поддерживаемые паттерны:
          "123"     → 123.0
          "123p45"  → 123.45
          "m123"    → -123.0
          "m123p45" → -123.45
        
        :param s: Закодированная строка
        :return: float или None при невалидном формате
        """
        # Проверка формата регулярным выражением: опциональный 'm', цифры, опциональная дробная часть
        if not re.fullmatch(r'm?\d+(p\d+)?', s):
            return None
        try:
            # Декодирование символов
            s_clean = s.replace('p', '.')
            if s_clean.startswith('m'):
                s_clean = '-' + s_clean[1:]
            return float(s_clean)
        except (ValueError, TypeError):
            return None

    # Парсинг всех трёх компонентов с валидацией
    angle = parse_component(angle_str)
    dx = parse_component(dx_str)
    dy = parse_component(dy_str)

    if None in (angle, dx, dy):
        return None, None, None

    return dx, dy, angle


def calculate_phase_correlation(img1, img2):
    """
    Вычисляет карту фазовой корреляции (Phase-Only Correlation) для измерения сдвига.
    
    :param img1: Первое изображение (2D массив, градации серого)
    :param img2: Второе изображение (того же размера)
    :return: Карта корреляции (2D массив), где максимум соответствует вектору сдвига
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Несовпадение размеров изображений: {img1.shape} != {img2.shape}")
    
    # Генерация 2D окна Ханна для минимизации граничных артефактов
    hann_window = np.outer(
        np.hanning(img1.shape[0]),  # Горизонтальная компонента
        np.hanning(img1.shape[1])   # Вертикальная компонента
    )
    
    # Применение окна и преобразование в float32 для точных вычислений
    img1_windowed = img1.astype(np.float32) * hann_window
    img2_windowed = img2.astype(np.float32) * hann_window
    
    # Прямое 2D Фурье-преобразование
    fft1 = np.fft.fft2(img1_windowed)
    fft2 = np.fft.fft2(img2_windowed)
    
    # Фазово-нормализованный кросс-спектр (ключевая операция метода)
    cross_spectrum = np.conjugate(fft2) * fft1
    cross_spectrum_normalized = cross_spectrum / (np.abs(cross_spectrum) + 1e-10)  # Защита от деления на ноль
    
    # Обратное преобразование и центрирование пика нулевого сдвига
    correlation = np.abs(np.fft.ifftshift(np.fft.ifft2(cross_spectrum_normalized)))
    
    return correlation


def split_images_into_tiles(img1, img2, grid_size=(3, 3)):
    """
    Разбивает пару изображений на идентичные тайлы по прямоугольной сетке.
    
    ОСОБЕННОСТИ РЕАЛИЗАЦИИ:
      - Равномерное разделение с коррекцией для последнего тайла в ряду/столбце
      - Сохранение полной площади изображения (без потерь на границах)
      - Возврат двумерных списков для удобной адресации по координатам сетки
    
    :param img1: Первое изображение (H×W)
    :param img2: Второе изображение (того же размера)
    :param grid_size: Кортеж (высота_сетки, ширина_сетки), по умолчанию (3, 3)
    :return: Кортеж:
             - tiles1: двумерный список тайлов первого изображения [[ряд0], [ряд1], ...]
             - tiles2: двумерный список тайлов второго изображения
             - (tile_h, tile_w): размер одного тайла в пикселях
    """
    h, w = img1.shape[:2]
    
    # Расчёт базового размера тайла (целочисленное деление)
    tile_h = h // grid_size[0]
    tile_w = w // grid_size[1]
    
    tiles1, tiles2 = [], []
    
    # Итерация по сетке с обработкой границ
    for i in range(grid_size[0]):
        row_tiles1, row_tiles2 = [], []
        for j in range(grid_size[1]):
            # Расчёт координат тайла с коррекцией для последних тайлов
            y_start = i * tile_h
            y_end = h if i == grid_size[0] - 1 else (i + 1) * tile_h
            x_start = j * tile_w
            x_end = w if j == grid_size[1] - 1 else (j + 1) * tile_w
            
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
    Находит координаты глобального максимума на карте корреляции (грубая оценка сдвига).
    
    :param correlation_map: 2D массив значений корреляции
    :return: Кортеж (y, x) — координаты пика в пикселях
    """
    return np.unravel_index(np.argmax(correlation_map), correlation_map.shape)


def weighted_centroid(correlation, peak_loc, window_size=5):
    """
    Уточняет позицию пика методом взвешенного центроида для достижения субпиксельной точности.
    
    ПРИНЦИП РАБОТЫ:
      Центр масс = Σ(координата × вес) / Σ(вес)
      где вес = значение корреляции в данной точке
    
    ПРЕИМУЩЕСТВА:
      - Повышает точность измерения сдвига до 0.1 пикселя
      - Устойчив к шуму за счёт усреднения в окрестности пика
      - Вычислительно эффективен (локальная операция)
    
    :param correlation: Карта корреляции (2D массив)
    :param peak_loc: Грубая позиция пика (y, x)
    :param window_size: Размер окна уточнения (нечётное число, по умолчанию 5)
    :return: Кортеж (x, y) — уточнённые координаты с дробной частью
    """
    rows, cols = correlation.shape
    y_peak, x_peak = peak_loc
    
    # Расчёт границ окна с защитой от выхода за границы изображения
    half = window_size // 2
    y1 = max(0, y_peak - half)
    y2 = min(rows, y_peak + half + 1)
    x1 = max(0, x_peak - half)
    x2 = min(cols, x_peak + half + 1)
    
    # Извлечение окна и создание координатной сетки
    window = correlation[y1:y2, x1:x2]
    y_grid, x_grid = np.mgrid[y1:y2, x1:x2]
    
    # Вычисление центра масс
    total = np.sum(window)
    if total > 0:
        y_refined = np.sum(y_grid * window) / total
        x_refined = np.sum(x_grid * window) / total
    else:
        y_refined, x_refined = y_peak, x_peak
    
    return x_refined, y_refined


def calculate_pce(correlation):
    """
    Вычисляет метрику PCE (Peak to Correlation Energy) — отношение энергии пика к энергии фона.
    
    ИНТЕРПРЕТАЦИЯ:
      - PCE > 100: очень надёжное сопоставление
      - PCE 10–100: надёжное сопоставление
      - PCE < 10: неуверенное сопоставление (риск ложного пика)
    
    ВАЖНО ДЛЯ СИНТЕТИЧЕСКИХ ДАННЫХ:
      При идеальных трансформациях (без шума) PCE всегда высокий, поэтому метрика имеет
      ограниченную дискриминативную способность. Основную информацию несут измеренные смещения.
    
    :param correlation: Карта корреляции (2D массив)
    :return: Значение PCE (безразмерная величина, float)
    """
    # Поиск глобального максимума
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak_val = correlation[peak_idx]
    
    # Создание маски фона (всё кроме окрестности 5×5 вокруг пика)
    mask = np.ones_like(correlation, dtype=bool)
    h, w = correlation.shape
    r, c = peak_idx
    r1, r2 = max(0, r - 2), min(h, r + 3)
    c1, c2 = max(0, c - 2), min(w, c + 3)
    mask[r1:r2, c1:c2] = False
    
    # Расчёт энергии фона с защитой от деления на ноль
    bg_energy = np.mean(correlation[mask] ** 2) + 1e-10
    pce = (peak_val ** 2) / bg_energy
    
    return pce


def calculate_correlation_for_tiles(tiles1, tiles2):
    """
    Выполняет фазовую корреляцию для всех тайлов в сетке и извлекает признаки сопоставления.
    
    :param tiles1: Двумерный список тайлов первого изображения [ряд][столбец]
    :param tiles2: Двумерный список тайлов второго изображения
    :return: Кортеж:
             - combined_correlation: объединённая карта корреляции (для визуализации)
             - all_shifts: двумерный список словарей с результатами для каждого тайла
    """
    rows, cols = len(tiles1), len(tiles1[0])
    all_shifts = []
    
    # Получение размера корреляционной карты одного тайла
    sample_corr = calculate_phase_correlation(tiles1[0][0], tiles2[0][0])
    tile_h, tile_w = sample_corr.shape
    
    # Инициализация объединённой карты (для отладки)
    combined = np.zeros((rows * tile_h, cols * tile_w), dtype=np.float32)
    
    # Обработка каждого тайла в сетке
    for i in range(rows):
        row_results = []
        for j in range(cols):
            # Вычисление корреляции для текущего тайла
            corr = calculate_phase_correlation(tiles1[i][j], tiles2[i][j])
            
            # Обнаружение и уточнение пика
            peak = find_shift(corr)
            refined = weighted_centroid(corr, peak, window_size=5)
            
            # # Расчёт метрик
            # response = np.max(corr)
            # pce = calculate_pce(corr)
            
            # Сохранение результатов
            row_results.append({
                'tile_coords': (i, j),
                'custom_shift': refined,   # (x, y) с субпиксельной точностью
                'peak_location': peak,     # (y, x) грубая оценка
                # 'response': response,      # Максимальное значение корреляции
                # 'PCE': pce                 # Peak to Correlation Energy
            })
            
            # Заполнение объединённой карты (опционально для визуализации)
            combined[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = corr
        
        all_shifts.append(row_results)
    
    return combined, all_shifts


def extract_tails_info(shifts):
    """
    Преобразует результаты обработки тайлов в плоский словарь признаков для экспорта в CSV.
    
    СТРУКТУРА ВЫХОДНЫХ ДАННЫХ:
      Для каждого тайла (3×3 = 9 тайлов) сохраняются 4 признака:
        - {i}_{j}_dx: смещение по оси X (пиксели)
        - {i}_{j}_dy: смещение по оси Y (пиксели)
        - {i}_{j}_PCE: метрика качества сопоставления
        - {i}_{j}_resp: максимальный отклик корреляции
    
    ПРИМЕР КЛЮЧЕЙ:
      "0_0_dx", "0_0_dy", "0_0_PCE", "0_0_resp", ..., "2_2_dx", "2_2_dy", "2_2_PCE", "2_2_resp"
    
    :param shifts: Двумерный список результатов [ряд][столбец]
    :return: Словарь с 36 признаками (9 тайлов × 4 признака)
    """
    features = {}
    for i in range(len(shifts)):
        for j in range(len(shifts[i])):
            tile = shifts[i][j]
            prefix = f"{i}_{j}"
            # Округление для компактности хранения и уменьшения шума в данных
            features[f"{prefix}_dx"] = round(tile['custom_shift'][0], 4)
            features[f"{prefix}_dy"] = round(tile['custom_shift'][1], 4)
            # features[f"{prefix}_PCE"] = round(tile['PCE'], 3)
            # features[f"{prefix}_resp"] = round(tile['response'], 3)
    return features


def main():
    """
    Основной цикл генерации обучающего датасета для предсказания угла поворота.
    
    ЭТАПЫ ОБРАБОТКИ:
      1. Обход всех поддиректорий в ANGLES_DIR
      2. Для каждой поддиректории:
          a. Загрузка эталонного изображения (0p0_0p0_0p00.jpg)
          b. Нормализация яркости до диапазона [0, 255]
          c. Обработка всех трансформированных изображений:
               - Загрузка и нормализация
               - Разбиение на тайлы 3×3
               - Вычисление фазовой корреляции для каждого тайла
               - Извлечение 36 признаков (смещения, PCE, отклик)
               - Парсинг истинного угла из имени файла
          d. Формирование записи для датасета
      3. Агрегация всех записей в единый DataFrame
      4. Сохранение в CSV-файл для последующего обучения моделей
    
    СТРУКТУРА ВЫХОДНОГО ДАТАСЕТА:
      - 36 признаков: измеренные смещения и метрики для 9 тайлов
      - 1 целевая переменная: угол поворота (градусы)
      - Столбцы упорядочены: все признаки → угол (удобно для разделения X/y)
    """
    
    # Определение структуры столбцов выходного датасета
    # columns = [
    #     "0_0_dx", "0_0_dy", "0_0_PCE", "0_0_resp",
    #     "0_1_dx", "0_1_dy", "0_1_PCE", "0_1_resp",
    #     "0_2_dx", "0_2_dy", "0_2_PCE", "0_2_resp",
    #     "1_0_dx", "1_0_dy", "1_0_PCE", "1_0_resp",
    #     "1_1_dx", "1_1_dy", "1_1_PCE", "1_1_resp",
    #     "1_2_dx", "1_2_dy", "1_2_PCE", "1_2_resp",
    #     "2_0_dx", "2_0_dy", "2_0_PCE", "2_0_resp",
    #     "2_1_dx", "2_1_dy", "2_1_PCE", "2_1_resp",
    #     "2_2_dx", "2_2_dy", "2_2_PCE", "2_2_resp",
    #     "angle"  # Целевая переменная в конце для удобства разделения признаков и таргета
    # ]
    columns = [
        "0_0_dx", "0_0_dy", "0_1_dx", "0_1_dy", "0_2_dx", "0_2_dy", 
        "1_0_dx", "1_0_dy", "1_1_dx", "1_1_dy", "1_2_dx", "1_2_dy", 
        "2_0_dx", "2_0_dy", "2_1_dx", "2_1_dy", "2_2_dx", "2_2_dy", 
        "angle"  # Целевая переменная в конце для удобства разделения признаков и таргета
    ]
    
    # Инициализация пустого DataFrame для агрегации результатов
    all_data = pd.DataFrame(columns=columns)
    
    # Обход всех поддиректорий с изображениями
    for root, _, files in os.walk(ANGLES_DIR):
        # Пропуск корневой директории (обрабатываем только поддиректории)
        if root == ANGLES_DIR:
            continue
        
        # Поиск эталонного изображения (центральный кроп без трансформаций)
        ref_path = os.path.join(root, "0p0_0p0_0p00.jpg")
        if not os.path.isfile(ref_path):
            print(f"Пропущена папка '{os.path.basename(root)}': отсутствует эталонное изображение")
            continue
        
        # Загрузка и нормализация эталонного изображения
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            print(f"Ошибка загрузки эталона: {ref_path}")
            continue
        
        ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Список для накопления результатов текущей поддиректории
        results = []
        
        # Обработка всех трансформированных изображений
        for filename in files:
            # Пропуск эталонного изображения (обрабатываем только трансформированные)
            if filename == "0p0_0p0_0p00.jpg":
                continue
            
            img_path = os.path.join(root, filename)
            curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if curr_img is None:
                print(f"Ошибка загрузки: {filename}")
                continue
            
            # Нормализация текущего изображения
            curr_img = cv2.normalize(curr_img, None, 0, 255, cv2.NORM_MINMAX)
            
            try:
                # Разбиение на тайлы и вычисление корреляции
                tiles_ref, tiles_curr, _ = split_images_into_tiles(ref_img, curr_img, grid_size)
                _, tile_shifts = calculate_correlation_for_tiles(tiles_ref, tiles_curr)
                
                # Извлечение истинного угла из имени файла
                _, _, angle = parse_shift_angle_from_filename(filename)
                if angle is None:
                    print(f"Не удалось распарсить угол из имени: {filename}")
                    continue
                
                # Формирование записи с признаками и целевой переменной
                record = extract_tails_info(tile_shifts)
                record['angle'] = angle
                results.append(record)
                
            except Exception as e:
                print(f"Ошибка обработки {filename}: {str(e)}")
                continue
        
        # Преобразование результатов в DataFrame и добавление к общему датасету
        if results:
            df_batch = pd.DataFrame(results, columns=columns)
            df_batch.fillna(0, inplace=True)  # Замена пропущенных значений нулями
            all_data = pd.concat([all_data, df_batch], ignore_index=True)
            print(f"Обработано: '{os.path.basename(root)}' → {len(df_batch)} записей")
        else:
            print(f"Нет валидных изображений в папке: {os.path.basename(root)}")
    
    # Вывод сводной информации о датасете
    print(f"\nПример данных (первые 3 записи):")
    print(all_data.head(3))
    
    # Сохранение датасета в CSV
    output_path = path[0] + "\\combined_data_angle.csv"
    all_data.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nДатасет сохранён: {output_path}")
    print(f"   Размер: {os.path.getsize(output_path) / 1024:.1f} КБ")


if __name__ == "__main__":
    main()