"""
eda.py — Exploratory Data Analysis для network traffic логов
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Настройки графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path("reports/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_basic_stats(df):
    """1. Базовая статистика датасета"""
    print("\n" + "="*70)
    print("БАЗОВАЯ СТАТИСТИКА ДАТАСЕТА")
    print("="*70)

    print(f"\n📊 Размер датасета: {len(df):,} сессий")
    print(f"   Размер в памяти: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        print(f"\n📅 Временной диапазон:")
        print(f"   От:  {df['timestamp'].min()}")
        print(f"   До:  {df['timestamp'].max()}")
        duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
        print(f"   Период: {duration_days} дней")

    print(f"\n🌐 Уникальные значения:")
    if 'src_ip' in df.columns:
        print(f"   src_ip:      {df['src_ip'].nunique():,}")
    if 'dst_ip' in df.columns:
        print(f"   dst_ip:      {df['dst_ip'].nunique():,}")
    if 'application' in df.columns:
        print(f"   application: {df['application'].nunique():,}")
    
    if 'protocol' in df.columns:
        print(f"\n🔌 Протоколы:")
        for proto, count in df['protocol'].value_counts().head(5).items():
            pct = count / len(df) * 100
            print(f"   {proto:8s}  {count:8,} ({pct:5.1f}%)")
    
    if 'action' in df.columns:
        print(f"\n⚡ Actions:")
        for action, count in df['action'].value_counts().items():
            pct = count / len(df) * 100
            print(f"   {action:8s}  {count:8,} ({pct:5.1f}%)")
    
    # Статистика трафика
    if 'bytes_sent' in df.columns and 'bytes_rcvd' in df.columns:
        total_bytes = df['bytes_sent'].sum() + df['bytes_rcvd'].sum()
        print(f"\n📦 Суммарный трафик: {total_bytes / 1024**3:.2f} GB")
        print(f"   Средний размер сессии: {total_bytes / len(df) / 1024:.1f} KB")


def plot_traffic_timeline(df):
    """2. Временной ряд трафика"""
    if 'timestamp' not in df.columns:
        return
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df_time = df.set_index('timestamp').resample('1h').size()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Сессии по времени
    ax1.plot(df_time.index, df_time.values, linewidth=1.5, color='#2196F3')
    ax1.fill_between(df_time.index, df_time.values, alpha=0.3, color='#2196F3')
    ax1.set_title('Количество сессий по времени (hourly)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Количество сессий')
    ax1.grid(True, alpha=0.3)
    
    # Трафик по времени
    if 'bytes_sent' in df.columns:
        df_traffic = df.set_index('timestamp').resample('1h')[['bytes_sent', 'bytes_rcvd']].sum()
        df_traffic_gb = df_traffic / 1024**3
        
        ax2.plot(df_traffic_gb.index, df_traffic_gb['bytes_sent'], 
                label='Sent', linewidth=1.5, color='#FF5722')
        ax2.plot(df_traffic_gb.index, df_traffic_gb['bytes_rcvd'], 
                label='Received', linewidth=1.5, color='#4CAF50')
        ax2.fill_between(df_traffic_gb.index, df_traffic_gb['bytes_sent'], alpha=0.2, color='#FF5722')
        ax2.fill_between(df_traffic_gb.index, df_traffic_gb['bytes_rcvd'], alpha=0.2, color='#4CAF50')
        ax2.set_title('Объём трафика по времени (hourly)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('GB')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_traffic_timeline.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Сохранён: {OUTPUT_DIR / '01_traffic_timeline.png'}")
    plt.close()


def plot_activity_heatmap(df):
    """2.2 Heatmap активности по времени суток"""
    if 'timestamp' not in df.columns:
        return
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    
    # Pivot table: hour x day_of_week
    activity = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    
    # Правильный порядок дней недели
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    activity = activity.reindex([d for d in days_order if d in activity.index])
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(activity, cmap='YlOrRd', annot=False, fmt='d', 
                cbar_kws={'label': 'Количество сессий'})
    plt.title('Heatmap активности: день недели x час', fontsize=14, fontweight='bold')
    plt.xlabel('Час дня')
    plt.ylabel('День недели')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_activity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Сохранён: {OUTPUT_DIR / '02_activity_heatmap.png'}")
    plt.close()


def plot_feature_distributions(df):
    """3. Распределения числовых признаков"""
    numeric_features = ['bytes_sent', 'bytes_rcvd', 'pkts_sent', 'pkts_rcvd']
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    if not numeric_features:
        return
    
    fig, axes = plt.subplots(2, len(numeric_features), figsize=(16, 8))
    
    for i, feature in enumerate(numeric_features):
        # Histogram (normal scale)
        ax = axes[0, i] if len(numeric_features) > 1 else axes[0]
        ax.hist(df[feature].dropna(), bins=50, color='#2196F3', alpha=0.7, edgecolor='black')
        ax.set_title(f'{feature}', fontweight='bold')
        ax.set_ylabel('Частота')
        ax.grid(True, alpha=0.3)
        
        # Histogram (log scale)
        ax = axes[1, i] if len(numeric_features) > 1 else axes[1]
        log_values = np.log1p(df[feature].dropna())
        ax.hist(log_values, bins=50, color='#FF9800', alpha=0.7, edgecolor='black')
        ax.set_title(f'log1p({feature})', fontweight='bold')
        ax.set_ylabel('Частота')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Сохранён: {OUTPUT_DIR / '03_feature_distributions.png'}")
    plt.close()


def plot_correlation_heatmap(df):
    """4. Correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Исключаем ID колонки
    numeric_cols = [c for c in numeric_cols if c not in ['session_id', 'protocol_num']]
    
    if len(numeric_cols) < 2:
        return
    
    # Вычисляем корреляции
    corr = df[numeric_cols].corr()
    
    # Heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # маскируем верхний треугольник
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix (числовые признаки)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Сохранён: {OUTPUT_DIR / '04_correlation_heatmap.png'}")
    plt.close()


def plot_top_ips_and_ports(df):
    """5. Топ IP-адресов и портов"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Топ src_ip по количеству сессий
    if 'src_ip' in df.columns:
        top_src = df['src_ip'].value_counts().head(15)
        axes[0, 0].barh(range(len(top_src)), top_src.values, color='#2196F3')
        axes[0, 0].set_yticks(range(len(top_src)))
        axes[0, 0].set_yticklabels(top_src.index, fontsize=9)
        axes[0, 0].set_xlabel('Количество сессий')
        axes[0, 0].set_title('Топ-15 src_ip (по количеству сессий)', fontweight='bold')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Топ dst_ip по количеству сессий
    if 'dst_ip' in df.columns:
        top_dst = df['dst_ip'].value_counts().head(15)
        axes[0, 1].barh(range(len(top_dst)), top_dst.values, color='#4CAF50')
        axes[0, 1].set_yticks(range(len(top_dst)))
        axes[0, 1].set_yticklabels(top_dst.index, fontsize=9)
        axes[0, 1].set_xlabel('Количество сессий')
        axes[0, 1].set_title('Топ-15 dst_ip (по количеству сессий)', fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Топ dst_port
    if 'dst_port' in df.columns:
        top_ports = df['dst_port'].value_counts().head(15)
        axes[1, 0].barh(range(len(top_ports)), top_ports.values, color='#FF5722')
        axes[1, 0].set_yticks(range(len(top_ports)))
        axes[1, 0].set_yticklabels(top_ports.index, fontsize=9)
        axes[1, 0].set_xlabel('Количество сессий')
        axes[1, 0].set_title('Топ-15 dst_port', fontweight='bold')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Топ applications
    if 'application' in df.columns:
        top_apps = df['application'].value_counts().head(15)
        axes[1, 1].barh(range(len(top_apps)), top_apps.values, color='#9C27B0')
        axes[1, 1].set_yticks(range(len(top_apps)))
        axes[1, 1].set_yticklabels(top_apps.index, fontsize=9)
        axes[1, 1].set_xlabel('Количество сессий')
        axes[1, 1].set_title('Топ-15 applications', fontweight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_top_ips_ports_apps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Сохранён: {OUTPUT_DIR / '05_top_ips_ports_apps.png'}")
    plt.close()


def plot_outliers_boxplot(df):
    """7. Box plots для выявления выбросов"""
    features = ['bytes_sent', 'bytes_rcvd', 'pkts_sent', 'pkts_rcvd']
    features = [f for f in features if f in df.columns]
    
    if not features:
        return
    
    fig, axes = plt.subplots(1, len(features), figsize=(16, 5))
    
    for i, feature in enumerate(features):
        ax = axes[i] if len(features) > 1 else axes
        
        # Box plot
        bp = ax.boxplot(df[feature].dropna(), vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#2196F3')
        bp['boxes'][0].set_alpha(0.7)
        
        ax.set_ylabel('Значение')
        ax.set_title(feature, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Показываем количество выбросов
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[feature] < Q1 - 1.5*IQR) | (df[feature] > Q3 + 1.5*IQR)]
        ax.text(0.5, 0.95, f'Выбросов: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)',
                transform=ax.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_outliers_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Сохранён: {OUTPUT_DIR / '06_outliers_boxplot.png'}")
    plt.close()


def plot_class_distribution(df):
    """9. Распределение классов (если есть метки)"""
    if 'is_anomaly' not in df.columns:
        print("\n⚠️  Колонка 'is_anomaly' не найдена — пропускаем class distribution")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    counts = df['is_anomaly'].value_counts()
    labels = ['Нормальные', 'Аномалии']
    colors = ['#4CAF50', '#F44336']
    
    ax1.pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Распределение классов', fontsize=14, fontweight='bold')
    
    # Bar chart
    ax2.bar(['Нормальные', 'Аномалии'], counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Количество сессий')
    ax2.set_title('Дисбаланс классов', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Добавляем числа на столбики
    for i, v in enumerate(counts.values):
        ax2.text(i, v + max(counts.values)*0.02, f'{v:,}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Сохранён: {OUTPUT_DIR / '07_class_distribution.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='EDA для network traffic данных')
    parser.add_argument('--data', type=str, required=True, 
                       help='Путь к CSV файлу с данными')
    args = parser.parse_args()
    
    # Загружаем данные
    print(f"\n📂 Загрузка данных из: {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    
    print(f"✓ Загружено: {len(df):,} строк, {len(df.columns)} колонок")
    
    # Запускаем EDA
    print("\n" + "="*70)
    print("ЗАПУСК EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    print_basic_stats(df)
    
    print(f"\n📊 Создание графиков...")
    plot_traffic_timeline(df)
    plot_activity_heatmap(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_top_ips_and_ports(df)
    plot_outliers_boxplot(df)
    plot_class_distribution(df)
    
    print(f"\n{'='*70}")
    print(f"✅ EDA ЗАВЕРШЁН")
    print(f"{'='*70}")
    print(f"Все графики сохранены в: {OUTPUT_DIR}/")
    print(f"Создано графиков: {len(list(OUTPUT_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
