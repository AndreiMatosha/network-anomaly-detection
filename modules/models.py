"""
models.py — Модели для детекции аномалий сетевого трафика
Включает: Isolation Forest, Autoencoder, DBSCAN, One-Class SVM, Local Outlier Factor
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# sklearn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

# TensorFlow / Keras для Autoencoder
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow не установлен. Autoencoder недоступен.")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ISOLATION FOREST
# ══════════════════════════════════════════════════════════════════════════════

class IsolationForestDetector:
    """
    Isolation Forest: изолирует аномалии, разбивая данные случайными разрезами.
    Аномалии изолируются быстрее → меньшая глубина дерева → меньший score.
    Лучший выбор для старта — быстрый, эффективный, интерпретируемый.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42):
        """
        Args:
            contamination: ожидаемая доля аномалий (0.01–0.20)
            n_estimators: количество деревьев (больше = стабильнее)
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1
        )
        self.name = "Isolation Forest"

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        print(f"Обучение {self.name} на {X.shape[0]} записях...")
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает 1 (аномалия) или 0 (норма)."""
        raw = self.model.predict(X)          # sklearn: 1 норм, -1 аномалия
        return np.where(raw == -1, 1, 0)    # переводим: 1 = аномалия

    def score(self, X: np.ndarray) -> np.ndarray:
        """Возвращает anomaly score (чем ниже — тем аномальнее)."""
        return -self.model.score_samples(X)  # инвертируем: выше = аномальнее

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Модель сохранена: {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 2. AUTOENCODER (Neural Network)
# ══════════════════════════════════════════════════════════════════════════════

class AutoencoderDetector:
    """
    Autoencoder: обучается сжимать и восстанавливать нормальный трафик.
    Аномалии плохо восстанавливаются → высокая ошибка реконструкции.
    Хорошо улавливает сложные паттерны, требует больше данных.
    """

    def __init__(self, input_dim: int = 30, encoding_dim: int = 8,
                 threshold_percentile: float = 95):
        """
        Args:
            input_dim: размерность входных признаков
            encoding_dim: размер bottleneck (сжатое представление)
            threshold_percentile: процентиль ошибки для порога аномалии
        """
        if not TF_AVAILABLE:
            raise ImportError("Установи tensorflow: pip install tensorflow")

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.name = "Autoencoder"
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """Строит симметричный Autoencoder: encoder → bottleneck → decoder."""
        inputs = Input(shape=(self.input_dim,), name="input")

        # Encoder
        x = Dense(64, activation="relu", name="enc_1")(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation="relu", name="enc_2")(x)
        x = Dropout(0.2)(x)
        encoded = Dense(self.encoding_dim, activation="relu", name="bottleneck")(x)

        # Decoder
        x = Dense(32, activation="relu", name="dec_1")(encoded)
        x = Dropout(0.2)(x)
        x = Dense(64, activation="relu", name="dec_2")(x)
        outputs = Dense(self.input_dim, activation="linear", name="output")(x)

        model = Model(inputs, outputs, name="autoencoder")
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X: np.ndarray, epochs: int = 50, batch_size: int = 256,
            validation_split: float = 0.1) -> "AutoencoderDetector":
        print(f"Обучение {self.name} на {X.shape[0]} записях...")

        # Пересоздаём модель с правильной размерностью
        self.input_dim = X.shape[1]
        self.model = self._build_model()

        early_stop = EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        )

        self.model.fit(
            X, X,                        # Autoencoder: вход = выход
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )

        # Вычисляем порог на обучающих данных
        recon_errors = self._reconstruction_error(X)
        self.threshold = np.percentile(recon_errors, self.threshold_percentile)
        print(f"Порог аномалии (p{self.threshold_percentile}): {self.threshold:.4f}")
        return self

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Вычисляет MSE ошибку реконструкции для каждой записи."""
        X_pred = self.model.predict(X, verbose=0)
        return np.mean(np.power(X - X_pred, 2), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает 1 (аномалия) или 0 (норма)."""
        errors = self._reconstruction_error(X)
        return (errors > self.threshold).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Возвращает ошибку реконструкции как anomaly score."""
        return self._reconstruction_error(X)

    def save(self, path: str):
        model_path = path.replace(".pkl", "_keras.keras")
        self.model.save(model_path)
        meta = {"threshold": self.threshold, "input_dim": self.input_dim,
                "encoding_dim": self.encoding_dim,
                "threshold_percentile": self.threshold_percentile}
        with open(path, "wb") as f:
            pickle.dump(meta, f)
        print(f"Autoencoder сохранён: {model_path}, мета: {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            meta = pickle.load(f)
        self.threshold = meta["threshold"]
        self.input_dim = meta["input_dim"]
        model_path = path.replace(".pkl", "_keras.keras")
        self.model = load_model(model_path)
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 3. ONE-CLASS SVM
# ══════════════════════════════════════════════════════════════════════════════

class OneClassSVMDetector:
    """
    One-Class SVM: строит гиперплоскость, разделяющую нормальный трафик.
    Хорошая теоретическая основа, но медленный на больших данных.
    Рекомендуется на выборке <= 50,000 записей.
    """

    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale"):
        """
        Args:
            nu: верхняя граница доли аномалий (аналог contamination)
            kernel: тип ядра (rbf, linear, poly)
        """
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.name = "One-Class SVM"

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        print(f"Обучение {self.name} на {X.shape[0]} записях...")
        # Для больших датасетов обучаем на подвыборке
        if X.shape[0] > 50000:
            idx = np.random.choice(X.shape[0], 50000, replace=False)
            print(f"Датасет большой, обучаем на подвыборке: 50,000 записей")
            self.model.fit(X[idx])
        else:
            self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict(X)
        return np.where(raw == -1, 1, 0)

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOCAL OUTLIER FACTOR
# ══════════════════════════════════════════════════════════════════════════════

class LOFDetector:
    """
    Local Outlier Factor: сравнивает плотность точки с её соседями.
    Хорошо находит локальные аномалии (аномалии внутри кластеров нормального трафика).
    Не поддерживает predict на новых данных без novelty=True.
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,           # novelty=True позволяет predict на новых данных
            n_jobs=-1
        )
        self.name = "Local Outlier Factor"

    def fit(self, X: np.ndarray) -> "LOFDetector":
        print(f"Обучение {self.name} на {X.shape[0]} записях...")
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict(X)
        return np.where(raw == -1, 1, 0)

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 5. DBSCAN (кластеризация)
# ══════════════════════════════════════════════════════════════════════════════

class DBSCANDetector:
    """
    DBSCAN: кластеризует трафик; точки вне кластеров = аномалии (label = -1).
    Не требует задавать число кластеров, сам определяет их форму.
    Чувствителен к параметрам eps и min_samples.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 10):
        """
        Args:
            eps: максимальное расстояние между точками одного кластера
            min_samples: минимум точек для формирования кластера
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels_ = None
        self.name = "DBSCAN"

    def fit(self, X: np.ndarray) -> "DBSCANDetector":
        print(f"Обучение {self.name} на {X.shape[0]} записях...")
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        self.labels_ = self.model.fit_predict(X)
        n_anomalies = np.sum(self.labels_ == -1)
        print(f"DBSCAN нашёл кластеров: {len(set(self.labels_)) - 1}, "
              f"аномалий (шум): {n_anomalies} ({n_anomalies/len(X)*100:.1f}%)")
        return self

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """DBSCAN transductive: возвращает метки обучающих данных."""
        return np.where(self.labels_ == -1, 1, 0)

    def score(self, X: np.ndarray = None) -> np.ndarray:
        """Условный score: аномалии получают 1, нормальные — 0."""
        return self.predict().astype(float)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "labels": self.labels_}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.labels_ = data["labels"]
        return self


# ══════════════════════════════════════════════════════════════════════════════
# Фабрика моделей
# ══════════════════════════════════════════════════════════════════════════════

def get_all_models(contamination: float = 0.05, input_dim: int = 30) -> dict:
    """Возвращает словарь всех доступных детекторов."""
    models = {
        "isolation_forest": IsolationForestDetector(contamination=contamination),
        "one_class_svm":    OneClassSVMDetector(nu=contamination),
        "lof":              LOFDetector(contamination=contamination),
        "dbscan":           DBSCANDetector(),
    }
    if TF_AVAILABLE:
        models["autoencoder"] = AutoencoderDetector(
            input_dim=input_dim,
            threshold_percentile=int((1 - contamination) * 100)
        )
    return models
