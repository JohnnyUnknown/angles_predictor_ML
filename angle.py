import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression, RANSACRegressor, QuantileRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sys import path
import optuna
from xgboost import XGBRegressor
from EDA_angles import get_selected_params
from time import perf_counter


def bayes_opt(X_train, y_train):
    def multioutput_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    mae_scorer = make_scorer(multioutput_mae, greater_is_better=False)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 40, 300),
            "eval_metric": "mae",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        
        model = XGBRegressor(**params)
        scores = -cross_val_score(model, X_train, y_train, cv=3, scoring=mae_scorer, n_jobs=-1)
        return scores.mean()

    print("\nЗапуск Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Лучшие параметры (Optuna):")
    print(study.best_params)
    print(f"Лучший MAE: {study.best_value:.3f} px")


def angular_error_deg(y_true, y_pred):
    """
    Вычисляет ошибку между истинными и предсказанными углами (в градусах)
    с учётом цикличности. Для малых углов (-3..3) почти совпадает с простой разностью.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = y_pred - y_true
    # Нормализуем к [-180, 180)
    errors = (diff + 180) % 360 - 180
    return errors


def evaluate_angle_predictions(y_true, y_pred, title="Оценка качества предсказания угла"):
    """
    Полная оценка качества предсказания угла с визуализацией.
    
    Параметры:
        y_true : array-like — истинные углы (градусы)
        y_pred : array-like — предсказанные углы (градусы)
        title : str — заголовок графиков
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 1. Ошибки
    errors = angular_error_deg(y_true, y_pred)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    std_error = np.std(errors)
    
    # 2. Статистика
    print(f"🔍 Статистика ошибок (в градусах):")
    print(f"  MAE (средняя абсолютная ошибка): {mae:.4f}°")
    print(f"  RMSE (среднеквадратичная ошибка): {rmse:.4f}°")
    print(f"  Стандартное отклонение ошибки:  {std_error:.4f}°")
    print(f"  Максимальная ошибка:           {np.max(np.abs(errors)):.4f}°")
    print(f"  95-й перцентиль ошибки:        {np.percentile(np.abs(errors), 95):.4f}°")
    
    # 3. Визуализация
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # a) Scatter plot: истинный vs предсказанный
    axs[0].scatter(y_true, y_pred, alpha=0.6, s=20)
    axs[0].plot([-3, 3], [-3, 3], 'r--', label='Идеальное совпадение')
    axs[0].set_xlabel('Истинный угол (°)')
    axs[0].set_ylabel('Предсказанный угол (°)')
    axs[0].set_title('Истинный vs Предсказанный')
    axs[0].legend()
    axs[0].grid(True)
    
    # b) Гистограмма ошибок
    axs[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axs[1].axvline(0, color='red', linestyle='--')
    axs[1].set_xlabel('Ошибка (°)')
    axs[1].set_ylabel('Частота')
    axs[1].set_title(f'Распределение ошибок\n(MAE = {mae:.4f}°)')
    axs[1].grid(True)
    
    # c) Ошибки по величине истинного угла
    axs[2].scatter(y_true, errors, alpha=0.6, s=20)
    axs[2].axhline(0, color='red', linestyle='--')
    axs[2].set_xlabel('Истинный угол (°)')
    axs[2].set_ylabel('Ошибка (°)')
    axs[2].set_title('Ошибка в зависимости от истинного угла')
    axs[2].grid(True)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'std_error': std_error,
        'max_error': np.max(np.abs(errors)),
        'percentile_95': np.percentile(np.abs(errors), 95),
        'errors': errors
    }


path_dir = Path(path[0])

all_data = pd.read_csv((path_dir / "angles\\combined_data_angle.csv"))

delta = 1

X, y = get_selected_params(method=None, num_of_params=8, show_img=False, save_img=False)
print(len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
test_index = list(y_test.index)


# # Оптимизация гиперпараметров моделей
# bayes_opt(X, y)


model = LinearRegression()
# model = QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs')
# model = XGBRegressor(n_estimators=249, max_depth=10, learning_rate=0.037, eval_metric=mean_absolute_error,
#                       random_state=1, subsample=0.64, colsample_bytree=0.7)


print("Значение кросс-валидации модели (MAE):", np.mean(cross_val_score(model, X, y, cv=5,
                                                                  scoring='neg_mean_absolute_error') * -1), "\n")
print("Значение кросс-валидации модели (MSE):", np.mean(cross_val_score(model, X, y, cv=5,
                                                                  scoring='neg_mean_squared_error') * -1), "\n")

model.fit(X_train, y_train)

start = perf_counter()
y_pred = model.predict(X_test)
finish = perf_counter()
print("Время инференса модели:", round((finish - start) / X_test.shape[0] * 1000000, 5), "мкр сек.\n")

# # Сохранение модели (обучение на полном наборе данных)
# model.fit(np.array(X), y)
# dump(model, 'angles_calc.joblib')


evaluate_angle_predictions(y_test, y_pred)

