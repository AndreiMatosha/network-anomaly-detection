"""
export_anomalies.py — Экспорт аномалий для ручной проверки (unsupervised сценарий)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def export_top_anomalies(df_original: pd.DataFrame,
                         anomaly_mask: np.ndarray,
                         scores: np.ndarray,
                         model_name: str,
                         top_n: int = 100,
                         output_dir: str = "reports") -> str:
    """
    Экспортирует топ-N самых аномальных записей в CSV для ручной проверки.
    
    Args:
        df_original: исходный DataFrame с логами (до feature engineering)
        anomaly_mask: бинарная маска [1=аномалия, 0=норма]
        scores: anomaly scores (выше = аномальнее)
        model_name: название модели для имени файла
        top_n: сколько записей экспортировать
        output_dir: директория для сохранения
        
    Returns:
        Путь к созданному файлу
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Копируем датафрейм и добавляем результаты модели
    df_export = df_original.copy()
    df_export["is_anomaly"] = anomaly_mask
    df_export["anomaly_score"] = scores
    
    # Берём только аномалии
    df_anomalies = df_export[df_export["is_anomaly"] == 1].copy()
    
    # Сортируем по score (самые аномальные сверху)
    df_anomalies = df_anomalies.sort_values("anomaly_score", ascending=False)
    
    # Топ-N
    df_top = df_anomalies.head(top_n)
    
    # Выбираем колонки для экспорта (самые информативные)
    export_cols = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "protocol", "application", "policy",
        "bytes_sent", "bytes_rcvd", "pkts_sent", "pkts_rcvd",
        "duration", "action",
        "anomaly_score"
    ]
    
    # Берём только те колонки, которые есть в df
    available_cols = [c for c in export_cols if c in df_top.columns]
    df_export_final = df_top[available_cols]
    
    # Добавляем колонку для ручной разметки
    df_export_final["manual_label"] = ""  # analyst заполнит: "port_scan", "ddos", "benign", etc.
    df_export_final["notes"] = ""         # analyst может добавить комментарии
    
    # Сохраняем
    filename = f"anomalies_{model_name.lower().replace(' ', '_')}_top{top_n}.csv"
    filepath = Path(output_dir) / filename
    df_export_final.to_csv(filepath, index=False)
    
    print(f"✓ Экспортировано {len(df_export_final)} аномалий: {filepath}")
    print(f"  Для ручной проверки: заполните колонки 'manual_label' и 'notes'")
    
    return str(filepath)


def print_anomaly_summary(df_original: pd.DataFrame,
                         anomaly_mask: np.ndarray,
                         scores: np.ndarray,
                         model_name: str):
    """
    Печатает сводку по найденным аномалиям (топ src_ip, dst_port, etc.)
    """
    df = df_original.copy()
    df["is_anomaly"] = anomaly_mask
    df["anomaly_score"] = scores
    
    anomalies = df[df["is_anomaly"] == 1]
    
    print(f"\n{'='*70}")
    print(f"СВОДКА АНОМАЛИЙ: {model_name}")
    print(f"{'='*70}")
    print(f"Всего найдено аномалий: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)")
    print(f"Средний anomaly score:  {scores[anomaly_mask==1].mean():.4f}")
    print(f"Макс anomaly score:     {scores.max():.4f}")
    
    # Топ src_ip по количеству аномалий
    if "src_ip" in anomalies.columns:
        print(f"\nТоп-5 src_ip (по количеству аномальных сессий):")
        top_src = anomalies["src_ip"].value_counts().head(5)
        for ip, count in top_src.items():
            pct = count / len(anomalies) * 100
            print(f"  {ip:20s}  {count:5d} сессий ({pct:.1f}% аномалий)")
    
    # Топ dst_port среди аномалий
    if "dst_port" in anomalies.columns:
        print(f"\nТоп-5 dst_port (среди аномалий):")
        top_ports = anomalies["dst_port"].value_counts().head(5)
        for port, count in top_ports.items():
            pct = count / len(anomalies) * 100
            print(f"  {port:6d}  {count:5d} сессий ({pct:.1f}% аномалий)")
    
    # Топ applications среди аномалий
    if "application" in anomalies.columns:
        print(f"\nТоп-5 applications (среди аномалий):")
        top_apps = anomalies["application"].value_counts().head(5)
        for app, count in top_apps.items():
            pct = count / len(anomalies) * 100
            print(f"  {app:25s}  {count:5d} сессий ({pct:.1f}% аномалий)")
    
    # Распределение по времени суток
    if "hour" in anomalies.columns:
        print(f"\nРаспределение аномалий по времени суток:")
        night = len(anomalies[anomalies["hour"].isin([22,23,0,1,2,3,4,5,6])])
        day   = len(anomalies) - night
        print(f"  Ночь (22:00-06:00): {night:5d} ({night/len(anomalies)*100:.1f}%)")
        print(f"  День (07:00-21:00): {day:5d} ({day/len(anomalies)*100:.1f}%)")
    
    print(f"{'='*70}\n")


def compare_models_unsupervised(all_results: list) -> pd.DataFrame:
    """
    Сравнивает модели по unsupervised метрикам (без y_true).
    
    Returns:
        DataFrame с метриками каждой модели
    """
    df = pd.DataFrame(all_results)
    
    # Сортируем по anomaly_rate (предпочитаем 3-7%)
    df["rate_deviation"] = abs(df["anomaly_rate"] - 5.0)  # идеал = 5%
    df = df.sort_values("rate_deviation")
    
    print("\n" + "="*70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ (unsupervised метрики)")
    print("="*70)
    print(df[["model", "n_anomalies", "anomaly_rate"]].to_string(index=False))
    print(f"\nРекомендация: модели с anomaly_rate 3-7% обычно наиболее адекватны")
    print(f"              слишком высокий % → модель слишком чувствительна")
    print(f"              слишком низкий %  → модель пропускает аномалии")
    print("="*70)
    
    return df
