import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sys import path
import seaborn as sns


def corr_matrix(all_data):    
    corr_matrix = all_data.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляционная матрица')
    plt.show()


def PCA_analysis(X, n_components=0.95, quantity=None, show_img=True, save_img=False):
    """
    Анализ важности признаков через PCA.
    """
    # 1. Стандартизация (обязательно для PCA!)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # 2. Применение PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    
    num_params = quantity if quantity else pca.n_components_

    # 3. Анализ объяснённой дисперсии
    print(f"Выбрано компонент: {pca.n_components_}")
    print(f"Объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"Дисперсия по компонентам: {[f'{v:.1%}' for v in pca.explained_variance_ratio_]}...")
    
    # 4. Важность признаков
    loadings = pca.components_.T  # shape: (n_features, n_components)
    feature_importance = np.abs(loadings).sum(axis=1)
    
    # 5. Формируем рейтинг
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # 6. Визуализация
    if show_img:
        plt.figure(figsize=(12, 6))
        plt.barh(importance_df['feature'][:num_params], 
                 importance_df['importance'][:num_params])
        plt.xlabel('Суммарная абсолютная нагрузка')
        plt.title(f'Важность признаков по PCA\n(объяснено {pca.explained_variance_ratio_.sum():.1%} дисперсии)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_img:
            plt.savefig(path_dir / "graphics/importance_PCA.jpg", dpi=300, bbox_inches='tight')
        plt.show()
    
    # 7. Возврат топ-признаков
    selected = importance_df['feature'].values[:num_params]
    print("\nТоп признаков по PCA:")
    print(selected)
    
    return selected


def AVG_analysis(X, y, quantity=10, show_img=True, save_img=False):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Получаем важности
    importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    if show_img:
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Средняя важность признака')
        plt.title('Влияние признаков на ошибку измерения ???.')
        plt.gca().invert_yaxis()  # Самый важный — сверху
        plt.tight_layout()
        plt.show()
        if save_img:
            plt.savefig((path_dir / "graphics\\importance_avg.jpg"), dpi=500)

    if quantity > len(X.columns): 
        quantity = len(X.columns)
    top_features = feature_importance_df['feature'].values[:quantity]
    print("AVG:", top_features)
    return top_features


def SBS_analysis(X, y, quantity=10, show_img=True, save_img=False):
    quantity = min(quantity, X.shape[1])
    model = LinearRegression().fit(X, y)
    selector = SelectFromModel(model, max_features=quantity, threshold=-np.inf).fit(X, y)
    
    mask = selector.get_support()
    print(mask)
    idx = np.argsort(np.abs(model.coef_)[mask])[::-1]
    features, importances = X.columns[mask][idx], np.abs(model.coef_)[mask][idx]
    
    if show_img:
        plt.figure(figsize=(10, max(4, len(features) * 0.45)))
        plt.barh(features, importances, color='steelblue')
        plt.xlabel('Абсолютное значение коэффициента')
        plt.title(f'Топ-{len(features)} признаков')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_img:
            plt.savefig(path_dir / "graphics/importance_SBS.jpg", dpi=500, bbox_inches='tight')
        plt.show()
    
    print("Топ признаков по SelectFromModel:\n", features.values)
    return features.values


def analysis_data(all_data):    
    print(round(all_data.describe().transpose(), 2))

    # Поиск нулевых значений
    print(all_data.isnull().sum())

    # # Проверка на дисбаланс примеров с нулевыми смещениями
    # print("dx=0, dy=0:", len(all_data.loc[(all_data["true_dx"] == 0) & (all_data["true_dy"] == 0)]), 
    #       "раз из", len(all_data), "\n")

    corr_matrix(all_data)


def get_selected_params(method=None, num_of_params=18, show_img=False, save_img=False):
    all_data = pd.read_csv((path_dir / "combined_data_angle.csv"))

    # analysis_data(all_data)
    features_columns = [col for col in all_data.columns[:-1]]

    y = all_data.loc[:, all_data.columns[-1]]
    X = all_data.loc[:, features_columns]

    if method in ['AVG', 'PCA', 'SBS']:
        if method == 'AVG':
            params = AVG_analysis(X, y, quantity=num_of_params, show_img=show_img, save_img=save_img)
        if method == 'PCA':
            params = PCA_analysis(X, quantity=num_of_params, show_img=show_img, save_img=save_img)
        if method == 'SBS':
            params = SBS_analysis(X, y, quantity=num_of_params, show_img=show_img, save_img=save_img)
    else:
        print("Выбраны признаки по умолчанию!")
        return X, y

    # Оптимальные параметры, определенные эмпирически
    # params = ['angle', 'dx', 'dy', 'sharpness', 'entropy', 'snr', 'mean_brightness']
    # print("Выбраны параметры:", params)

    # Отбор и возврат наиболее значимых параметров
    return X.loc[:, params], y


path_dir = Path(path[0])
get_selected_params(method='SBS', num_of_params=18, show_img=True)