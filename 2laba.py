import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Генерування випадкових значень
np.random.seed(42)
X = np.sort(5 * np.random.rand(1000, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 1000)

# Нормалізування даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Розділення існуючих записів на навчальну і тестову вибірки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Навчання KNN-регресора з різними значеннями K
k_values = list(range(1, 31))
mse_scores = []
r2_scores = []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse_scores.append(mean_squared_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

# Обрано величину K для найкращих показників якості регресії у тестовій вибірці
best_k = k_values[np.argmin(mse_scores)]
best_mse = min(mse_scores)
best_r2 = r2_scores[np.argmin(mse_scores)]

print(f"Найкраще значення K: {best_k}")
print(f"Найменша помилка (MSE): {best_mse:.4f}")
print(f"Коефіцієнт детермінації (R^2): {best_r2:.4f}")

# Візуалізація результатів

# Графік 1: Залежність MSE від параметра K
plt.figure(figsize=(10, 5))
plt.plot(k_values, mse_scores, marker='o', color='b', linestyle='--')
plt.axvline(x=best_k, color='r', linestyle=':', label=f'Найкраще K = {best_k}')
plt.title('Залежність середньоквадратичної помилки (MSE) від кількості сусідів (K)')
plt.xlabel('Кількість сусідів (K)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.legend()
plt.show()

# Графік 2: Візуалізація результатів регресії для найкращого K
best_model = KNeighborsRegressor(n_neighbors=best_k)
best_model.fit(X_train, y_train)

X_grid = np.linspace(X_scaled.min(), X_scaled.max(), 500).reshape(-1, 1)
y_grid_pred = best_model.predict(X_grid)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='gray', alpha=0.6, label='Тестові дані')
plt.plot(X_grid, y_grid_pred, color='red', linewidth=2, label=f'KNN Регресія (K={best_k})')
plt.title(f'Апроксимація функції за допомогою KNN-регресора (K={best_k})')
plt.xlabel('Нормалізована ознака (X)')
plt.ylabel('Цільове значення (Y)')
plt.grid(True)
plt.legend()
plt.show()