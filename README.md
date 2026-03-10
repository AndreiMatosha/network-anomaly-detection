# Детекция аномалий сетевого трафика
### Проектная работа по Machine Learning | Juniper SRX Firewall Logs

---

## Описание

Система автоматической детекции аномалий в сетевом трафике на основе логов
Juniper SRX firewall. Реализует 4 алгоритмf unsupervised anomaly detection.

---

## Структура проекта
```
├── anomalies_results
│   ├── anomalies_autoencoderdetector_top100.csv
│   ├── anomalies_isolation_forest_top100.csv
│   ├── anomalies_lofdetector_top100.csv
│   └── anomalies_oneclasssvmdetector_top100.csv
├── data
│   ├── data_raw
│   │   ├── attack_filtered_static.log
│   │   └── fw_prefix_filtered.log
│   ├── data_tmp
│   │   ├── feature_eng_done.csv
│   │   ├── parsed_df.csv
│   │   ├── pickle_scaler.pkl
│   │   ├── processed_session_closed_df.csv
│   │   └── X.npy
│   └── prediction_tmp
│       ├── anomalies_autoencoderdetector_top100.csv
│       ├── anomalies_isolation_forest_top100.csv
│       ├── anomalies_lofdetector_top100.csv
│       ├── anomalies_oneclasssvmdetector_top100.csv
│       ├── feature_eng_done.csv
│       ├── parsed_df.csv
│       ├── pickle_scaler.pkl
│       ├── processed_session_closed_df.csv
│       └── X.npy
├── models
│   ├── autoencoder_keras.keras
│   ├── autoencoder.pkl
│   ├── isolation_forest.pkl
│   ├── lofmodel_keras.keras
│   ├── lofmodel.pkl
│   └── oneclasssvm_model.pkl
├── modules
│   ├── eda.py
│   ├── evaluate.py
│   ├── export_anomalies.py
│   ├── features.py
│   ├── models.py
│   └── parser.py
│  
├── otus_main.ipynb
├── prediction.ipynb
├── README.md
└── reports
    └── eda
        ├── 01_traffic_timeline.png
        ├── 02_activity_heatmap.png
        ├── 03_feature_distributions.png
        ├── 04_correlation_heatmap.png
        ├── 05_top_ips_ports_apps.png
        └── 06_outliers_boxplot.png

```

---

## Модели

| Модель | Принцип | Скорость | Лучшие сценарии |
|--------|---------|----------|-----------------|
| **Isolation Forest** | Изолирует аномалии деревьями | ★★★★★ | Старт, prod |
| **Autoencoder** | Ошибка реконструкции NN | ★★★☆☆ | Сложные паттерны |
| **One-Class SVM** | Гиперплоскость для нормы | ★★☆☆☆ | Малые датасеты |
| **Local Outlier Factor** | Плотность соседей | ★★★☆☆ | Локальные аномалии |
| **DBSCAN** | Кластеры + шум | ★★★☆☆ | Разведка данных |
| **Ensemble** | Голосование 5 моделей | ★★★☆☆ | Финал, точность |

---

## Типы аномалий, которые обнаруживает система

- **DDoS**: аномально высокое число пакетов/соединений с одного IP
- **Port Scan**: один src_ip → много разных dst_port за короткое время
- **Network Scan**: один src_ip → много разных dst_ip
- **Data Exfiltration**: аномально большой объём исходящих данных
- **Brute Force**: много отказанных соединений (deny) к одному порту
- **Beaconing**: регулярные короткие соединения к внешним IP (C2)

---

## Признаки (Feature Engineering)

**Сетевые метрики**: bytes_sent/rcvd, pkts_sent/rcvd, duration, bytes/pkts per sec  
**Порты**: is_well_known_port, is_rare_dst_port, is_privileged_port  
**IP-адреса**: is_private, is_outbound, is_inbound, is_internal  
**Поведенческие**: src_connection_count, src_unique_dst_ports, src_unique_dst_ips  
**Временные**: hour, day_of_week, is_night  

---
## Результаты работы
Достигнутые результаты

Разработана система детекции аномалий в сетевом трафике оператора связи на базе логов Juniper SRX firewall
Реализован  ML-pipeline: Парсинг сырых syslog → Feature engineering (60+ признаков) → Обучение моделей → Детекция аномалий
Обучено и сравнено 4 unsupervised моделей
Экспортировано топ-100 подозрительных сессий для ручной проверки 

Текущие ограничения: 

Без меток невозможно точно измерить Precision/Recall 
️Требуется ручная проверка топ-100 для валидации
️Contamination (5%) выбран эмпирически, требует тонкой настройки


Научная и образовательная ценность:

✔️ Освоены современные методы unsupervised anomaly detection
✔️ Получен практический опыт работы с реальными сетевыми логами промышленного оборудования 
✔️ Углублённое понимание feature engineering для кибербезопасности
✔️ Навыки работы с несбалансированными данными и отсутствием меток


## Выводы
Unsupervised anomaly detection — не идеальное, но единственно возможное решение для старта в условиях отсутствия размеченных данных. Итеративный процесс "детекция → ручная проверка → корректировка → переобучение" позволяет постепенно улучшать систему и накапливать экспертизу.

---

## Автор

Проектная работа по курсу Machine Learning  
Тема: детекция аномалий сетевого трафика (мобильный оператор, Juniper SRX)

---



