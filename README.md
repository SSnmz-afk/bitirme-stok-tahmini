# Makine Öğrenmesi ile Seyrek Talep Tahmini

Endüstriyel bir boru bağlantı parçasına (110-90° PN 10 Dirsek) ait günlük satış
verisi üzerinde geleneksel tahmin yöntemleri, seyrek talep modelleri ve makine
öğrenmesi modellerinin karşılaştırmalı analizi.

Serhat Sönmez,
Murat Erkahraman,
İshak Parlak

---

## Proje Yapısı

```
├── pipeline.py                        # Ana pipeline
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    ├── README.md                      # Sütun açıklamaları
    └── sample_gunluk_ml_dataset.csv   # Örnek veri (anonim, 365 satır)
```

Çıktılar çalıştırma sonrasında `outputs/` klasörüne yazılır (`.gitignore` kapsamında).

---

## Kurulum

```bash
git clone https://github.com/SSnmz-afk/bitirme-stok-tahmini.git
cd bitirme-stok-tahmini
pip install -r requirements.txt
```

---

## Çalıştırma

**Gerçek veriyle:**

1. Veri dosyasını `data/gunluk_ml_dataset.csv` olarak koyun
2. Pipeline'ı başlatın:

```bash
python pipeline.py
```

**Örnek veriyle hızlı test:**

```bash
cp data/sample_gunluk_ml_dataset.csv data/gunluk_ml_dataset.csv
python pipeline.py
```

## Veri Formatı

Pipeline `date` ve `sales_qty` sütunlarını zorunlu tutar; diğerleri opsiyoneldir.
Detaylı sütun açıklaması için `data/README.md` dosyasına bakın.
Örnek veri yapısı için `data/sample_gunluk_ml_dataset.csv` kullanılabilir.

---

## Modeller

| Grup | Model |
|---|---|
| Benchmark | Naive, Seasonal Naive, Hareketli Ortalama (7/14/30 gün), SES |
| Seyrek Talep | TSB, Croston-SBA |
| Makine Öğrenmesi | Random Forest, LightGBM (Tweedie / QBlend / TwoStage / ThreeStage), XGBoost, Ridge |
| Derin Öğrenme | LSTM |
| Ensemble | Ridge Meta-Learner Stacking |

Her model üç geçmiş penceresiyle denenir: `full`, `recent_only` (son 730 gün),
`recency_weighted`.

---

## Değerlendirme

**Walk-forward validasyon** — 2023, 2024 ve 2025 yılları sırasıyla test dönemi.

| Metrik | Açıklama |
|---|---|
| hWAPE | Haftalık toplanmış WAPE |
| WAPE | Ağırlıklı mutlak yüzde hata |
| Bias % | Sistematik sapma |
| Demand F1 | Talep var/yok sınıflandırma skoru |
| focused_score | Model seçiminde kullanılan bileşik skor (düşük = iyi) |

---

## Senaryolar

**Senaryo A — Toplam Talep:** Ham satış serisi üzerinde modelleme.

**Senaryo B — Operasyonel Talep:** P95 eşiği üzerindeki proje/ihale günleri
seriden ayrıştırılır; rutin talep ayrıca modellenir. Eşik veriden otomatik hesaplanır.

---

## Gereksinimler

```
Python >= 3.8
numpy · pandas · matplotlib · scikit-learn · openpyxl · xlsxwriter
lightgbm · xgboost · tensorflow   (opsiyonel)
```
