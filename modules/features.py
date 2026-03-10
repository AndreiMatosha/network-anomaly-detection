"""
features.py — Feature Engineering для сетевого трафика
Извлекаем признаки из сырых логов SRX для обучения моделей
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Хорошо известные порты — признак легитимности соединения
WELL_KNOWN_PORTS = {80, 443, 22, 53, 25, 110, 143, 993, 995,
                    21, 23, 3306, 5432, 6379, 27017, 8080, 8443}

# Приватные IP-диапазоны (RFC 1918)
PRIVATE_RANGES = [
    ("10.0.0.0", "10.255.255.255"),
    ("172.16.0.0", "172.31.255.255"),
    ("192.168.0.0", "192.168.255.255"),
]


def ip_to_int(ip: str) -> int:
    """Конвертирует IP-адрес в число."""
    try:
        parts = str(ip).split(".")
        return int(parts[0]) * 16777216 + int(parts[1]) * 65536 + \
               int(parts[2]) * 256 + int(parts[3])
    except Exception:
        return 0


def is_private_ip(ip: str) -> int:
    """Возвращает 1, если IP адрес приватный."""
    try:
        ip_int = ip_to_int(ip)
        for start, end in PRIVATE_RANGES:
            if ip_to_int(start) <= ip_int <= ip_to_int(end):
                return 1
        return 0
    except Exception:
        return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Основная функция feature engineering.
    Принимает сырой DataFrame, возвращает DataFrame с новыми признаками.
    """
    df = df.copy()

    # Regex-парсер возвращает все поля как str — приводим числовые к int/float.
    # Это нужно только для данных из parse_srx_syslog(); generate_sample_data()
    # и CSV уже возвращают правильные типы, но pd.to_numeric безопасен в обоих случаях.
    _numeric_cols = ["src_port", "dst_port", "bytes_sent", "bytes_rcvd",
                     "pkts_sent", "pkts_rcvd", "duration", "protocol_num"]
    for _col in _numeric_cols:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce").fillna(0)

    # ── 1. Базовые числовые признаки ──────────────────────────────────────────
    df["bytes_total"] = df["bytes_sent"].fillna(0) + df["bytes_rcvd"].fillna(0)
    df["pkts_total"] = df["pkts_sent"].fillna(0) + df["pkts_rcvd"].fillna(0)

    # Соотношения (добавляем +1 чтобы избежать деления на 0)
    df["bytes_ratio"] = df["bytes_sent"] / (df["bytes_rcvd"] + 1)
    df["pkts_ratio"] = df["pkts_sent"] / (df["pkts_rcvd"] + 1)

    # duration: длительность сессии в секундах.
    # При парсинге syslog (RT_FLOW) колонка создаётся в parser.py со значением 0,
    # так как Juniper SRX не включает elapsed-time в этот формат лога.
    # Если у тебя есть реальные данные о длительности — подставь их здесь до вызова
    # engineer_features(), например через join SESSION_CREATE/SESSION_CLOSE по session_id.
    if "duration" not in df.columns:
        df["duration"] = 0  # защита: колонка не пришла из парсера

    # Скорость передачи байт/пакетов в секунду
    # (+1 в знаменателе защищает от деления на 0 когда duration == 0)
    df["bytes_per_sec"] = df["bytes_total"] / (df["duration"].fillna(0) + 1)
    df["pkts_per_sec"] = df["pkts_total"] / (df["duration"].fillna(0) + 1)

    # Средний размер пакета
    df["avg_pkt_size"] = df["bytes_total"] / (df["pkts_total"] + 1)

    # ── 2. Признаки портов ────────────────────────────────────────────────────
    df["dst_port"] = df["dst_port"].fillna(0).astype(int)
    df["src_port"] = df["src_port"].fillna(0).astype(int)

    df["is_well_known_port"] = df["dst_port"].apply(
        lambda p: 1 if p in WELL_KNOWN_PORTS else 0
    )
    df["is_high_port"] = (df["dst_port"] > 1024).astype(int)
    df["is_privileged_port"] = (df["dst_port"] < 1024).astype(int)

    # Признак: необычный порт назначения (не встречается часто)
    port_counts = df["dst_port"].value_counts()
    rare_threshold = port_counts.quantile(0.10)
    df["is_rare_dst_port"] = df["dst_port"].apply(
        lambda p: 1 if port_counts.get(p, 0) <= rare_threshold else 0
    )

    # ── 3. Признаки IP-адресов ────────────────────────────────────────────────
    df["src_is_private"] = df["src_ip"].apply(is_private_ip)
    df["dst_is_private"] = df["dst_ip"].apply(is_private_ip)

    # Внешний трафик: internal -> external
    df["is_outbound"] = ((df["src_is_private"] == 1) & (df["dst_is_private"] == 0)).astype(int)
    df["is_inbound"] = ((df["src_is_private"] == 0) & (df["dst_is_private"] == 1)).astype(int)
    df["is_internal"] = ((df["src_is_private"] == 1) & (df["dst_is_private"] == 1)).astype(int)

    # ── 4. Частотные признаки (поведенческие) ────────────────────────────────
    # Сколько раз один src_ip инициировал соединения (высокий = потенциальный скан)
    src_conn_count = df.groupby("src_ip")["dst_port"].transform("count")
    df["src_connection_count"] = src_conn_count

    # Количество уникальных dst_port на один src_ip (признак port scan)
    src_unique_ports = df.groupby("src_ip")["dst_port"].transform("nunique")
    df["src_unique_dst_ports"] = src_unique_ports

    # Количество уникальных dst_ip на один src_ip (признак network scan)
    src_unique_ips = df.groupby("src_ip")["dst_ip"].transform("nunique")
    df["src_unique_dst_ips"] = src_unique_ips

    # ── 5. Признаки протокола ─────────────────────────────────────────────────
    proto_encoder = LabelEncoder()
    df["protocol_enc"] = proto_encoder.fit_transform(df["protocol"].fillna("UNKNOWN"))

    # ── 6. Признаки действия (action) ────────────────────────────────────────
    df["is_denied"] = (df["action"].str.lower() == "deny").astype(int)

    # ── 7. Временные признаки ─────────────────────────────────────────────────
    if "timestamp" in df.columns:
        # SRX syslog формат: "Jan 12 12:00:06" (без года)
        # Проверяем первую строку — если это syslog формат, добавляем год явно
        first_ts = str(df["timestamp"].iloc[0]).strip()
        if len(first_ts.split()) == 3:  # "Jan 12 12:00:06" = 3 слова
            # Добавляем текущий год для корректного парсинга
            import datetime
            year = datetime.datetime.now().year
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(str).str.strip() + f" {year}",
                format="%b %d %H:%M:%S %Y",
                errors="coerce"
            )
        else:
            # Уже содержит полную дату (из CSV или другого источника)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        # Ночное время: возможно нестандартная активность
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    else:
        df["hour"] = 12
        df["day_of_week"] = 0
        df["is_night"] = 0

    # ── 8. Логарифмические признаки (нормализация скошенных распределений) ────
    for col in ["bytes_sent", "bytes_rcvd", "bytes_total", "pkts_sent",
                "pkts_rcvd", "duration", "bytes_per_sec", "pkts_per_sec"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].fillna(0))

    return df


def get_feature_columns() -> list:
    """Возвращает список колонок, которые пойдут в модель."""
    return [
        # Сетевые метрики
        "log_bytes_sent", "log_bytes_rcvd", "log_bytes_total",
        "log_pkts_sent", "log_pkts_rcvd",
        "log_duration", "log_bytes_per_sec", "log_pkts_per_sec",
        "bytes_ratio", "pkts_ratio", "avg_pkt_size",
        # Порты
        "dst_port", "src_port",
        "is_well_known_port", "is_high_port", "is_privileged_port", "is_rare_dst_port",
        # IP
        "src_is_private", "dst_is_private",
        "is_outbound", "is_inbound", "is_internal",
        # Поведенческие
        "src_connection_count", "src_unique_dst_ports", "src_unique_dst_ips",
        # Протокол и action
        "protocol_enc", "is_denied",
        # Временные
        "hour", "day_of_week", "is_night",
    ]


def prepare_features(df: pd.DataFrame, scaler: StandardScaler = None,
                     fit: bool = True) -> tuple:
    """
    Финальная подготовка признаков для модели.

    Args:
        df: DataFrame с инженерными признаками
        scaler: sklearn StandardScaler (передать None для создания нового)
        fit: True — обучить scaler, False — только трансформировать

    Returns:
        X (np.ndarray), scaler
    """
    feature_cols = get_feature_columns()
    # Берём только те колонки, которые есть в датасете
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0).values

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    print(f"Признаков для модели: {len(available)}")
    return X_scaled, scaler, available
