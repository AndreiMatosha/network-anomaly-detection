"""
evaluate.py — Метрики, визуализация и сравнение моделей
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.decomposition import PCA

# Стиль графиков
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "normal": "#2196F3",
    "anomaly": "#F44336",
    "accent": "#FF9800",
    "green": "#4CAF50",
}

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   scores: np.ndarray, model_name: str) -> dict:
    """
    Вычисляет полный набор метрик для одной модели.
    Работает даже без y_true (unsupervised) — тогда только базовые статистики.
    """
    results = {"model": model_name}

    n_anomalies = y_pred.sum()
    results["n_anomalies"] = int(n_anomalies)
    results["anomaly_rate"] = float(n_anomalies / len(y_pred) * 100)

    if y_true is not None:
        # Supervised метрики
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])

        results["precision"] = float(tp / (tp + fp + 1e-9))
        results["recall"]    = float(tp / (tp + fn + 1e-9))
        results["f1"]        = float(2 * results["precision"] * results["recall"] /
                                     (results["precision"] + results["recall"] + 1e-9))
        results["roc_auc"]   = float(roc_auc_score(y_true, scores))
        results["avg_precision"] = float(average_precision_score(y_true, scores))

        print(f"\n{'─'*50}")
        print(f"Модель: {model_name}")
        print(f"Найдено аномалий: {n_anomalies} ({results['anomaly_rate']:.1f}%)")
        print(f"Precision:        {results['precision']:.4f}")
        print(f"Recall:           {results['recall']:.4f}")
        print(f"F1-score:         {results['f1']:.4f}")
        print(f"ROC-AUC:          {results['roc_auc']:.4f}")
        print(f"Avg Precision:    {results['avg_precision']:.4f}")
        print(classification_report(y_true, y_pred,
                                    target_names=["Нормальный", "Аномалия"]))
    else:
        print(f"\n{model_name}: найдено {n_anomalies} аномалий "
              f"({results['anomaly_rate']:.1f}%) — без меток для оценки")

    return results


def plot_comparison(all_results: list, save: bool = True):
    """Сравнительный bar-chart всех моделей по метрикам."""
    df = pd.DataFrame(all_results)
    if "f1" not in df.columns:
        print("Нет меток — пропускаем сравнительный график")
        return

    metrics = ["precision", "recall", "f1", "roc_auc", "avg_precision"]
    df_plot = df[["model"] + [m for m in metrics if m in df.columns]].set_index("model")

    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.7)
    ax.set_title("Сравнение моделей детекции аномалий", fontsize=14, fontweight="bold")
    ax.set_xlabel("Модель")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    if save:
        path = REPORT_DIR / "model_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {path}")
    plt.show()


def plot_roc_curves(models_scores: dict, y_true: np.ndarray, save: bool = True):
    """ROC-кривые всех моделей на одном графике."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, scores in models_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        axes[1].plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")

    # ROC
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("ROC Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", fontsize=9)

    # Precision-Recall
    axes[1].set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if save:
        path = REPORT_DIR / "roc_pr_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"ROC/PR кривые сохранены: {path}")
    plt.show()


def plot_pca_anomalies(X: np.ndarray, y_pred: np.ndarray,
                       model_name: str, y_true: np.ndarray = None, save: bool = True):
    """
    PCA-визуализация: проецируем многомерные данные в 2D
    и окрашиваем аномалии красным.
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2 if y_true is not None else 1,
                             figsize=(14 if y_true is not None else 7, 6))
    if y_true is None:
        axes = [axes]

    # Предсказания модели
    ax = axes[0]
    colors = [COLORS["anomaly"] if p == 1 else COLORS["normal"] for p in y_pred]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.4, s=10, linewidths=0)
    ax.set_title(f"{model_name} — Предсказания\n"
                 f"(красный=аномалия, синий=норма)", fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # Истинные метки (если есть)
    if y_true is not None:
        ax = axes[1]
        colors_true = [COLORS["anomaly"] if t == 1 else COLORS["normal"] for t in y_true]
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_true, alpha=0.4, s=10, linewidths=0)
        ax.set_title("Истинные метки\n(красный=аномалия, синий=норма)", fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    plt.tight_layout()
    if save:
        path = REPORT_DIR / f"pca_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"PCA-визуализация сохранена: {path}")
    plt.show()


def plot_anomaly_scores_distribution(scores: np.ndarray, threshold: float,
                                     model_name: str, y_true: np.ndarray = None,
                                     save: bool = True):
    """Распределение anomaly scores с порогом."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if y_true is not None:
        ax.hist(scores[y_true == 0], bins=80, alpha=0.6,
                color=COLORS["normal"], label="Нормальный трафик", density=True)
        ax.hist(scores[y_true == 1], bins=80, alpha=0.6,
                color=COLORS["anomaly"], label="Аномалии", density=True)
    else:
        ax.hist(scores, bins=100, alpha=0.7, color=COLORS["normal"], density=True)

    ax.axvline(threshold, color=COLORS["accent"], lw=2, linestyle="--",
               label=f"Порог = {threshold:.3f}")
    ax.set_title(f"{model_name} — Распределение Anomaly Scores", fontsize=12, fontweight="bold")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Плотность")
    ax.legend()

    if save:
        path = REPORT_DIR / f"scores_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(X: np.ndarray, feature_names: list,
                             anomaly_idx: np.ndarray, normal_idx: np.ndarray,
                             top_n: int = 15, save: bool = True):
    """
    Сравнение средних значений признаков у аномалий и нормальных записей.
    Показывает, какие признаки сильнее всего отличают аномалии от нормы.
    """
    means_normal  = np.mean(X[normal_idx],  axis=0)
    means_anomaly = np.mean(X[anomaly_idx], axis=0)

    # Нормализуем разницу
    diff = np.abs(means_anomaly - means_normal)
    top_idx = np.argsort(diff)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(top_n)
    ax.barh(y_pos - 0.2, means_normal[top_idx],  height=0.4,
            color=COLORS["normal"],  alpha=0.8, label="Норма")
    ax.barh(y_pos + 0.2, means_anomaly[top_idx], height=0.4,
            color=COLORS["anomaly"], alpha=0.8, label="Аномалия")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
    ax.set_title(f"Топ-{top_n} признаков (разница аномалия vs норма)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Среднее нормализованное значение")
    ax.legend()
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "feature_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"График признаков сохранён: {path}")
    plt.show()


def plot_traffic_timeline(df: pd.DataFrame, anomaly_mask: np.ndarray, save: bool = True):
    """Временной ряд трафика с отмеченными аномалиями (если есть timestamp)."""
    if "timestamp" not in df.columns:
        print("Нет колонки timestamp — временной ряд недоступен")
        return

    df_plot = df.copy()
    df_plot["is_anomaly"] = anomaly_mask
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
    df_plot = df_plot.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Агрегируем по минутам
    df_plot = df_plot.set_index("timestamp")
    traffic_per_min = df_plot.resample("1T").size()
    anomalies_per_min = df_plot[df_plot["is_anomaly"] == 1].resample("1T").size()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(traffic_per_min.index, traffic_per_min.values,
            color=COLORS["normal"], lw=1, alpha=0.8, label="Всего событий/мин")
    ax.fill_between(anomalies_per_min.index, anomalies_per_min.values,
                    color=COLORS["anomaly"], alpha=0.5, label="Аномалии/мин")
    ax.set_title("Временной ряд трафика с аномалиями", fontsize=12, fontweight="bold")
    ax.set_xlabel("Время")
    ax.set_ylabel("Событий в минуту")
    ax.legend()
    plt.tight_layout()

    if save:
        path = REPORT_DIR / "timeline.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Timeline сохранён: {path}")
    plt.show()
