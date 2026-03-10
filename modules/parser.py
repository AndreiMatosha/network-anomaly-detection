"""
parser.py — Парсинг логов Juniper SRX firewall
Поддерживает форматы: syslog (RT_FLOW), structured CSV
"""

import re
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex для логов Juniper SRX RT_FLOW (syslog формат)
#
# Поддерживает ДВА варианта формата:
#
# ФОРМАТ 1 (с "source rule"):
#   Jan 12 12:00:06 FW7-Gi1 RT_FLOW: RT_FLOW_SESSION_CLOSE: session closed
#   TCP CLIENT RST: 10.28.245.77/49706->142.250.120.156/443 0x0 junos-https
#   94.139.153.77/47698->142.250.120.156/443 0x0 source rule internet_offload2
#   N/A N/A 6 internet offload2_ssr offload2_bgw 1357210026769
#   10(2358) 7(5379) ...
#
# ФОРМАТ 2 (БЕЗ "source rule"):
#   Mar  2 18:00:25 FW7-Gi1 RT_FLOW: RT_FLOW_SESSION_CREATE: session created
#   162.142.125.232/64647->94.139.151.254/30426 0x0 None
#   162.142.125.232/64647->94.139.151.254/30426 0x0 N/A N/A N/A N/A 6
#   internet_static untrust SGi 1297096660675 N/A(N/A) reth1.947 ...
#
# Ключевые отличия:
#   - Формат 1: после NAT flow идёт "source rule policy_name"
#   - Формат 2: после NAT flow сразу идут 4 поля N/A без "source rule"
# ---------------------------------------------------------------------------

SRX_PATTERN = re.compile(
    # timestamp: "Jan 12 12:00:06" / "Mar  2 18:00:25" (два пробела если день < 10)
    r'(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'
    r'\s+(?P<hostname>\S+)'
    r'\s+RT_FLOW:\s+'
    r'(?P<event_type>RT_FLOW_SESSION_\w+):\s+'
    # reason_text — всё между event_type и первым IP-адресом
    # CLOSE: "session closed TCP CLIENT RST: "  (может быть двоеточие)
    # CREATE: "session created "                (без двоеточия перед IP)
    # Lookahead (?!\d{1,3}\.\d) останавливает захват перед IP-адресом
    r'(?P<reason_text>(?:(?!\d{1,3}\.\d).)+?)\s*'
    # ── Оригинальный flow (до NAT) ──────────────────────────────────────
    r'(?P<src_ip>\d{1,3}(?:\.\d{1,3}){3})/(?P<src_port>\d+)'
    r'->(?P<dst_ip>\d{1,3}(?:\.\d{1,3}){3})/(?P<dst_port>\d+)'
    r'\s+\S+\s+'                           # тег (0x0)
    r'(?P<application>\S+)\s+'             # application: junos-https, None, UNKNOWN
    # ── NAT flow ────────────────────────────────────────────────────────
    r'(?P<nat_src_ip>\d{1,3}(?:\.\d{1,3}){3})/(?P<nat_src_port>\d+)'
    r'->(?P<nat_dst_ip>\d{1,3}(?:\.\d{1,3}){3})/(?P<nat_dst_port>\d+)'
    r'\s+\S+\s+'                           # тег (0x0)
    # ── Policy блок (ОПЦИОНАЛЬНЫЙ "source rule") ────────────────────────
    # Формат 1: "source rule policy_name N/A N/A"
    # Формат 2: "N/A N/A N/A N/A" (без "source rule")
    r'(?:source\s+rule\s+)?'               # опциональный блок "source rule"
    r'(?P<policy>\S+)\s+(?P<src_zone>\S+)\s+(?P<dst_zone>\S+)'
    r'(?:\s+\S+)*?'                        # любое количество дополнительных полей (N/A, extra)
    r'\s+'
    # ── Protocol и interfaces ───────────────────────────────────────────
    r'(?P<protocol_num>\d+)\s+'
    r'(?P<src_interface>\S+)\s+(?P<dst_interface>\S+)\s+\S+\s+'
    # ── Session и traffic ───────────────────────────────────────────────
    r'(?P<session_id>\d+)\s+'
    # pkts_sent(bytes_sent): CLOSE → "10(2358)", CREATE → "N/A(N/A)"
    r'(?P<pkts_sent>\d+|N/A)\((?P<bytes_sent>\d+|N/A)\)'
    # pkts_rcvd(bytes_rcvd): есть только в CLOSE, отсутствует в CREATE
    r'(?:\s+(?P<pkts_rcvd>\d+|N/A)\((?P<bytes_rcvd>\d+|N/A)\))?',
    re.IGNORECASE
)

# Маппинг protocol_num → название протокола (RFC)
PROTO_MAP = {
    "1":  "ICMP",
    "6":  "TCP",
    "17": "UDP",
    "47": "GRE",
    "50": "ESP",
    "51": "AH",
    "89": "OSPF",
}

# Маппинг event_type → action
ACTION_MAP = {
    "RT_FLOW_SESSION_CLOSE":  "close",
    "RT_FLOW_SESSION_CREATE": "create",
    "RT_FLOW_SESSION_DENY":   "deny",
}


def parse_srx_syslog(filepath: str) -> pd.DataFrame:
    """
    Парсит syslog-файл Juniper SRX и возвращает DataFrame.
    Подходит для файлов формата RT_FLOW_SESSION_CLOSE / CREATE / DENY.
    
    Поддерживает оба варианта:
    - С блоком "source rule" (старые конфигурации SRX)
    - Без блока "source rule" (новые конфигурации SRX)
    """
    records = []
    filepath = Path(filepath)
    skipped = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            if "RT_FLOW" not in line:
                skipped += 1
                continue

            match = SRX_PATTERN.search(line)
            if match:
                data = match.groupdict()
                data["protocol"] = PROTO_MAP.get(data["protocol_num"], data["protocol_num"])
                data["action"] = ACTION_MAP.get(data["event_type"].upper(), "unknown")

                # duration отсутствует в RT_FLOW syslog формате Juniper SRX.
                # Формат логирует сессию одной строкой при закрытии (SESSION_CLOSE),
                # но elapsed-time в этом syslog-формате не включается.
                # Если нужна длительность — можно вычислить постфактум:
                #   duration = timestamp(CLOSE) - timestamp(CREATE) для одного session_id,
                #   но это требует join двух событий по session_id.
                # Пока ставим 0, features.py использует (duration + 1) и не упадёт.
                data["duration"] = 0

                records.append(data)
            else:
                skipped += 1

    df = pd.DataFrame(records)

    if df.empty:
        print("WARNING: не найдено ни одной записи.")
        print("Используй debug_line() для диагностики одной строки.")
        return df

    # Числовые колонки: CREATE пишет "N/A" в pkts/bytes — coerce превращает в NaN,
    # fillna(0) заменяет NaN нулём. Это корректно: в момент CREATE трафика ещё нет.
    int_cols = ["src_port", "dst_port", "nat_src_port", "nat_dst_port",
                "bytes_sent", "bytes_rcvd", "pkts_sent", "pkts_rcvd", "protocol_num"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # pkts_rcvd и bytes_rcvd отсутствуют в CREATE (None из regex) — заполняем 0
    for col in ["pkts_rcvd", "bytes_rcvd"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df.replace("N/A", pd.NA, inplace=True)

    print(f"Загружено записей:  {len(df)}")
    print(f"Пропущено строк:    {skipped}")
    print(f"Event types: {df['event_type'].value_counts().to_dict()}")
    print(f"Протоколы:   {df['protocol'].value_counts().to_dict()}")

    return df


def parse_srx_csv(filepath: str) -> pd.DataFrame:
    """Загружает логи SRX, экспортированные в CSV."""
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Загружено записей из CSV: {len(df)}")
    return df


def debug_line(line: str) -> None:
    """
    Отладочная функция: показывает что распарсилось из одной строки.
    Используй когда regex не матчится на новом формате лога.

    Пример:
        from src.parser import debug_line
        debug_line("Jan 12 12:00:06 FW7-Gi1 RT_FLOW: ...")
    """
    print("-" * 60)
    m = SRX_PATTERN.search(line)
    if m:
        print("MATCH! Разобранные поля:")
        for k, v in m.groupdict().items():
            print(f"  {k:20s} = {v}")
        print(f"  {'protocol':20s} = {PROTO_MAP.get(m.group('protocol_num'), '?')}")
        print(f"  {'action':20s} = {ACTION_MAP.get(m.group('event_type').upper(), '?')}")
    else:
        print("NO MATCH. Поэтапная диагностика:")
        steps = [
            ("timestamp",   r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}'),
            ("RT_FLOW",     r'RT_FLOW:\s+RT_FLOW_SESSION_\w+'),
            ("orig flow",   r'\d{1,3}(?:\.\d{1,3}){3}/\d+->\d{1,3}(?:\.\d{1,3}){3}/\d+'),
            ("policy zone", r'(?:source\s+rule\s+)?\S+\s+\S+\s+\S+'),
            ("pkts(bytes)", r'(?:\d+|N/A)\((?:\d+|N/A)\)'),
        ]
        for name, pat in steps:
            found = re.search(pat, line, re.IGNORECASE)
            status = "OK" if found else "FAIL"
            val = repr(found.group()) if found else "NOT FOUND"
            print(f"  [{status}] {name:15s}: {val}")
    print("-" * 60)


def generate_sample_data(n_samples: int = 10000, anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """
    Генерирует синтетические данные трафика для тестирования.
    Используй, если у тебя ещё нет реальных логов.
    """
    import numpy as np
    np.random.seed(42)

    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomaly = n_samples - n_normal

    normal = pd.DataFrame({
        "src_ip": [f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,254)}"
                   for _ in range(n_normal)],
        "dst_ip": [f"172.16.{np.random.randint(0,255)}.{np.random.randint(1,254)}"
                   for _ in range(n_normal)],
        "src_port": np.random.randint(1024, 65535, n_normal),
        "dst_port": np.random.choice([80, 443, 22, 53, 8080, 3306], n_normal,
                                      p=[0.35, 0.40, 0.10, 0.08, 0.04, 0.03]),
        "protocol": np.random.choice(["TCP", "UDP"], n_normal, p=[0.85, 0.15]),
        "bytes_sent": np.random.lognormal(7, 1.5, n_normal).astype(int),
        "bytes_rcvd": np.random.lognormal(8, 2, n_normal).astype(int),
        "pkts_sent": np.random.randint(1, 100, n_normal),
        "pkts_rcvd": np.random.randint(1, 150, n_normal),
        "action": np.random.choice(["close", "create"], n_normal, p=[0.98, 0.02]),
        "application": np.random.choice(["junos-https","junos-http","junos-ssh","UNKNOWN"],
                                         n_normal, p=[0.50, 0.25, 0.15, 0.10]),
        "policy": "internet_offload2",
    })

    anomaly = pd.DataFrame({
        "src_ip": [f"192.168.{np.random.randint(0,10)}.{np.random.randint(1,10)}"
                   for _ in range(n_anomaly)],
        "dst_ip": [f"8.8.{np.random.randint(0,255)}.{np.random.randint(1,254)}"
                   for _ in range(n_anomaly)],
        "src_port": np.random.randint(1024, 65535, n_anomaly),
        "dst_port": np.random.randint(1, 65535, n_anomaly),
        "protocol": np.random.choice(["TCP", "UDP", "ICMP"], n_anomaly, p=[0.5, 0.3, 0.2]),
        "bytes_sent": np.random.lognormal(12, 2, n_anomaly).astype(int),
        "bytes_rcvd": np.random.randint(0, 100, n_anomaly),
        "pkts_sent": np.random.randint(500, 10000, n_anomaly),
        "pkts_rcvd": np.random.randint(0, 10, n_anomaly),
        "action": np.random.choice(["deny", "close"], n_anomaly, p=[0.7, 0.3]),
        "application": np.random.choice(["UNKNOWN", "junos-https"], n_anomaly, p=[0.8, 0.2]),
        "policy": "internet_offload2",
    })

    df = pd.concat([normal, anomaly], ignore_index=True)
    df["is_anomaly"] = [0] * n_normal + [1] * n_anomaly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Синтетический датасет: {n_normal} норм. + {n_anomaly} аномалий")
    return df


def compute_session_duration(df: pd.DataFrame, year: int = None) -> pd.DataFrame:
    """
    Вычисляет duration (секунды) для каждой SESSION_CLOSE строки,
    делая join с соответствующим SESSION_CREATE по session_id.

    Почему так, а не через session_id напрямую:
        session_id в Juniper SRX — это монотонный счётчик ядра Junos,
        НЕ unix timestamp. Единственный способ узнать длительность сессии —
        найти пару событий SESSION_CREATE + SESSION_CLOSE с одинаковым
        session_id и взять разницу их timestamp.

    Juniper SRX syslog timestamp НЕ содержит год ("Jan 12 12:00:06"),
    поэтому год передаётся явно через параметр year.

    Что делаем с неполными данными:
        - CLOSE без CREATE (сессия началась до начала лог-файла) →
          duration = медиана всех найденных сессий.
        - Отрицательный duration (ротация логов через полночь) →
          прибавляем 86400 секунд (одни сутки).

    Args:
        df:   DataFrame из parse_srx_syslog() — все события вместе
              (SESSION_CREATE + SESSION_CLOSE + SESSION_DENY).
        year: год для парсинга timestamp. По умолчанию — текущий год.

    Returns:
        DataFrame только с SESSION_CLOSE строками + колонка duration (float, секунды).
        SESSION_CREATE строки отбрасываются — их байты/пакеты всегда 0,
        они нужны только как источник ts_create.
    """
    if year is None:
        year = __import__("datetime").datetime.now().year

    df = df.copy()

    # Шаг 1: парсим timestamp — добавляем год, которого нет в syslog
    df["ts"] = pd.to_datetime(
        df["timestamp"].str.strip() + f" {year}",
        format="%b %d %H:%M:%S %Y",
        errors="coerce"
    )

    # Шаг 2: разделяем CREATE и CLOSE
    creates = (
        df[df["event_type"] == "RT_FLOW_SESSION_CREATE"]
        [["session_id", "ts"]]
        .rename(columns={"ts": "ts_create"})
        .drop_duplicates(subset="session_id", keep="first")
    )

    closes = df[df["event_type"] == "RT_FLOW_SESSION_CLOSE"].copy()

    if closes.empty:
        print("Предупреждение: нет событий SESSION_CLOSE в датасете.")
        return closes

    # Шаг 3: left join по session_id
    # left — чтобы сохранить все CLOSE даже без пары CREATE
    merged = closes.merge(creates, on="session_id", how="left")

    # Шаг 4: duration = ts_close - ts_create в секундах
    merged["duration"] = (
        (merged["ts"] - merged["ts_create"])
        .dt.total_seconds()
        .round(1)
    )

    # Шаг 5: корректировка отрицательного duration
    # Возникает когда CREATE был до полуночи, CLOSE — после (ротация логов)
    negative_mask = merged["duration"] < 0
    if negative_mask.any():
        print(f"  {negative_mask.sum()} сессий с отрицательным duration "
              f"(ротация через полночь) — прибавляем 86400 сек")
        merged.loc[negative_mask, "duration"] += 86400

    # Шаг 6: CLOSE без CREATE → duration = медиана найденных сессий
    nan_count = merged["duration"].isna().sum()
    if nan_count > 0:
        median_dur = merged["duration"].median()
        print(f"  {nan_count} сессий без CREATE (начались до лог-файла) → "
              f"duration = median ({median_dur:.1f} сек)")
        merged["duration"] = merged["duration"].fillna(median_dur)

    # Убираем вспомогательные колонки
    result = merged.drop(columns=["ts", "ts_create"], errors="ignore")

    total = len(closes)
    matched = total - nan_count
    print(f"Session duration: {matched}/{total} сессий имеют пару CREATE+CLOSE "
          f"({matched/total*100:.1f}%)")

    return result
