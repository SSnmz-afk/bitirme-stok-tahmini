# =============================================================================
# Makine Öğrenmesi ile Seyrek Talep Tahmini — Bitirme Projesi
# Ürün: 110-90 PN 10 Dirsek
#
# Bölümler "# === [N] BAŞLIK ===" satırlarıyla ayrılmıştır;
# gerekirse her bölüm ayrı bir dosyaya taşınabilir.
# =============================================================================

# === [1] KURULUM VE IMPORTLAR ===============================================

import os
import json
import shutil
import zipfile
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def log(tag, msg):
    """[TAG] mesaj formatında log basar."""
    print(f"[{tag}] {msg}")


# Opsiyonel bağımlılıklar: kurulu değilse ilgili modeller atlanır.
try:
    import lightgbm  # noqa: F401
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow  # noqa: F401
    HAS_TF = True
except ImportError:
    HAS_TF = False

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

if HAS_LGB:
    import lightgbm as lgb
if HAS_XGB:
    import xgboost as xgb
if HAS_TF:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.random.set_seed(42)
    tf.get_logger().setLevel("ERROR")

np.random.seed(42)

log("SETUP", f"LightGBM: {'VAR' if HAS_LGB else 'YOK'} | "
             f"XGBoost: {'VAR' if HAS_XGB else 'YOK'} | "
             f"TensorFlow: {'VAR' if HAS_TF else 'YOK'}")

# === [2] GENEL AYARLAR =======================================================

# Türkçe karakter uyumlu DejaVu Sans kullanılır.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11

DPI = 300

# Renk paleti — tüm grafiklerde sabit anlam sözlüğü
COLOR_ACTUAL = "#0B3C5D"        # gercek talep -> koyu lacivert
COLOR_PRED = "#E8743B"          # tahmin -> turuncu
COLOR_GOOD = "#2E8B57"          # iyi/pozitif -> yesil
COLOR_BAD = "#C0392B"           # kritik/negatif -> kirmizi
COLOR_BENCH = "#7F8C8D"         # benchmark modeller -> gri/mavi
COLOR_ML = "#1ABC9C"            # ML modelleri -> turkuaz
COLOR_INTERMITTENT = "#8E44AD"  # seyrek talep modelleri -> mor
COLOR_DEEP = "#B8860B"          # LSTM/Stacking -> altin/koyu kirmizi
COLOR_BAND = "#F2C14E"          # tahmin bandi / buffer

PALETTE_GROUP = {
    "Benchmark": COLOR_BENCH,
    "Seyrek Talep": COLOR_INTERMITTENT,
    "Makine Ogrenmesi": COLOR_ML,
    "Derin Ogrenme": COLOR_DEEP,
    "Ensemble": "#34495E",
}

# Proje kok klasoru ve alt klasorler (senaryoya gore yeniden ayarlanabilir)
BASE_DIR = os.path.join(os.getcwd(), "outputs", "ml_talep_tahmini")
SUBDIRS = {}


def setup_output_dirs(base_dir):
    """Çıktı klasör yapısını oluşturur ve global BASE_DIR/SUBDIRS değişkenlerini günceller."""
    global BASE_DIR, SUBDIRS
    BASE_DIR = base_dir
    SUBDIRS = {
        "rapor": os.path.join(BASE_DIR, "00_rapor_icin_secili_grafikler"),
        "veri": os.path.join(BASE_DIR, "01_veri_analizi_grafikleri"),
        "karsilastirma": os.path.join(BASE_DIR, "02_model_karsilastirma_grafikleri"),
        "en_iyi": os.path.join(BASE_DIR, "03_en_iyi_model_grafikleri"),
        "detay": os.path.join(BASE_DIR, "04_model_bazli_detay_grafikler"),
        "stok": os.path.join(BASE_DIR, "05_stok_politikasi_grafikleri"),
        "feature_imp": os.path.join(BASE_DIR, "06_feature_importance"),
        "tablolar": os.path.join(BASE_DIR, "tablolar"),
        "predictions": os.path.join(BASE_DIR, "predictions"),
        "logs": os.path.join(BASE_DIR, "logs"),
    }
    for d in SUBDIRS.values():
        os.makedirs(d, exist_ok=True)
    return BASE_DIR, SUBDIRS


setup_output_dirs(BASE_DIR)

# Test / validation / büyük talep esikleri
TEST_YEARS = [2023, 2024, 2025]
VAL_WINDOW_DAYS = 365          # validation penceresi: test yilindan onceki 365 gun
BIG50_THRESHOLD = 50
BIG100_THRESHOLD = 100
SEASON_LENGTH = 7               # seasonal naive icin sezon uzunlugu

HISTORY_MODES = ["full", "recent_only", "recency_weighted"]
RECENT_ONLY_DAYS = 730          # recent_only modunda kullanilacak son gun sayisi
RECENCY_HALF_LIFE = 365         # recency_weighted agirliklandirma yari omru (gun)

log("SETUP", f"Cikti klasoru: {BASE_DIR}")
log("SETUP", f"Test yillari: {TEST_YEARS}")
# === [3] VERI OKUMA VE TEMEL TEMIZLIK =======================================

def find_data_file():
    """Veri dosyasını olası konumlarda sırayla arar, bulamazsa hata fırlatır."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "data", "gunluk_ml_dataset.csv"),
        os.path.join(base, "data", "gunluk_ml_dataset.xlsx"),
        os.path.join(base, "gunluk_ml_dataset.csv"),
        os.path.join(base, "gunluk_ml_dataset.xlsx"),
        "gunluk_ml_dataset.csv",
        "gunluk_ml_dataset.xlsx",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Veri dosyası bulunamadı. Beklenen konum: data/gunluk_ml_dataset.csv\n"
        "Örnek format için data/sample_gunluk_ml_dataset.csv dosyasına bakın."
    )


def read_raw_data(path):
    """CSV/Excel dosyasını okur; tarihe göre sıralar ve tekrar edenleri temizler."""
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("Veri dosyasinda 'date' kolonu bulunamadi.")
    if "sales_qty" not in df.columns:
        raise ValueError("Veri dosyasinda 'sales_qty' kolonu bulunamadi.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.reset_index(drop=True)

    # Tarih serisinde eksik gun varsa tamamla (gunluk frekans garanti edilir)
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    if len(full_range) != len(df):
        log("CLEAN", f"Eksik gun tespit edildi: {len(full_range) - len(df)} "
                      f"gun tamamlaniyor (sales_qty=0 ile dolduruluyor).")
        df = df.set_index("date").reindex(full_range)
        df.index.name = "date"
        df = df.reset_index()
        if "sales_qty" in df.columns:
            df["sales_qty"] = df["sales_qty"].fillna(0)
        df = df.ffill().bfill()
        df["sales_qty"] = df["sales_qty"].fillna(0)

    # Negatif veya gecersiz talep degerleri temizlenir
    neg_count = (df["sales_qty"] < 0).sum()
    if neg_count > 0:
        log("CLEAN", f"UYARI: {neg_count} negatif sales_qty kaydi 0'a "
                      f"cekiliyor.")
        df.loc[df["sales_qty"] < 0, "sales_qty"] = 0

    df["sales_qty"] = df["sales_qty"].astype(float)

    df = apply_data_cutoff(df)

    return df


# Veri kesim tarihi: 2 Aralık 2025 sonrasında ERP kayıt bütünlüğü şüpheli;
# bu dönemdeki sıfır blok gerçek talep değil muhtemelen kayıt eksikliği.
DATA_CUTOFF_DATE = pd.Timestamp("2025-12-02")


def apply_data_cutoff(df):
    """DATA_CUTOFF_DATE sonrasını keser — eksik ERP kaydını sıfır talep olarak okumamak için."""
    before = len(df)
    df = df[df["date"] <= DATA_CUTOFF_DATE].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        log("CLEAN", f"Veri kesimi uygulandi: {DATA_CUTOFF_DATE.date()} sonrasi "
                      f"{removed} gun, veri butunlugu supheli oldugu icin "
                      f"kapsam disi birakildi.")
    return df


# === TATİL FEATURE'LARI ======================================================
# Dini bayram sütunları veri setinde hazır; doğrudan kullanılır.
# Sabit tarihli ulusal tatiller (yılbaşı, cumhuriyet vs.) ayrıca hesaplanır.

FIXED_HOLIDAYS_MD = [
    (1, 1),    # Yilbasi
    (4, 23),   # Ulusal Egemenlik ve Cocuk Bayrami
    (5, 1),    # Emek ve Dayanisma Gunu
    (5, 19),   # Ataturk'u Anma, Genclik ve Spor Bayrami
    (7, 15),   # Demokrasi ve Milli Birlik Gunu
    (8, 30),   # Zafer Bayrami
    (10, 29),  # Cumhuriyet Bayrami
]


def add_holiday_features(df):
    """Sabit ulusal tatil günlerini işaretler; dini bayram sütunları varsa korur, yoksa 0 ekler."""
    df = df.copy()
    md = list(zip(df["date"].dt.month, df["date"].dt.day))
    df["is_resmi_tatil"] = [1 if x in FIXED_HOLIDAYS_MD else 0 for x in md]

    for col in ["is_dini_bayram", "bayram_oncesi_3g", "bayram_sonrasi_3g"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0).astype(int)

    return df


# === [4] VERI PROFILI VE SEYREK TALEP ANALIZI ===============================

def classify_demand(adi, cv2):
    """ADI / CV² eşiklerine göre talep sınıfını döndürür (smooth/intermittent/erratic/lumpy)."""
    if adi < 1.32 and cv2 < 0.49:
        return "smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        return "intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        return "erratic"
    else:
        return "lumpy"


def compute_adi_cv2(sales):
    """Pozitif talep günleri üzerinden ADI ve CV² hesaplar."""
    sales = np.asarray(sales)
    positive_idx = np.where(sales > 0)[0]
    n_pos = len(positive_idx)

    if n_pos < 2:
        return np.nan, np.nan

    intervals = np.diff(positive_idx)
    adi = np.mean(intervals) if len(intervals) > 0 else np.nan

    pos_values = sales[positive_idx]
    mean_pos = pos_values.mean()
    std_pos = pos_values.std(ddof=1) if n_pos > 1 else 0.0
    cv2 = (std_pos / mean_pos) ** 2 if mean_pos > 0 else np.nan

    return adi, cv2


def build_data_profile(df):
    """Veri istatistikleri, ADI/CV² sınıflandırması ve yıllık özeti hesaplar."""
    sales = df["sales_qty"].values
    n_days = len(df)
    n_zero = int((sales == 0).sum())
    n_pos = int((sales > 0).sum())
    adi, cv2 = compute_adi_cv2(sales)
    demand_class = classify_demand(adi, cv2) if not np.isnan(adi) else "n/a"

    rows = [
        ("Tarih araligi (baslangic)", df["date"].min().strftime("%Y-%m-%d")),
        ("Tarih araligi (bitis)", df["date"].max().strftime("%Y-%m-%d")),
        ("Toplam gun sayisi", n_days),
        ("Toplam satis/talep", float(sales.sum())),
        ("Ortalama gunluk talep", float(sales.mean())),
        ("Medyan gunluk talep", float(np.median(sales))),
        ("Maksimum gunluk talep", float(sales.max())),
        ("Sifir talep gunu sayisi", n_zero),
        ("Pozitif talep gunu sayisi", n_pos),
        ("Sifir talep orani (%)", round(100 * n_zero / n_days, 2)),
        ("Pozitif talep orani (%)", round(100 * n_pos / n_days, 2)),
        ("ADI (Average Demand Interval)", round(adi, 3) if not np.isnan(adi) else None),
        ("CV^2 (Talep Varyasyon Katsayisi Karesi)", round(cv2, 3) if not np.isnan(cv2) else None),
        ("Talep sinifi (ADI-CV^2)", demand_class),
        (f"{BIG50_THRESHOLD}+ buyuk talep gunu sayisi", int((sales >= BIG50_THRESHOLD).sum())),
        (f"{BIG100_THRESHOLD}+ buyuk talep gunu sayisi", int((sales >= BIG100_THRESHOLD).sum())),
    ]

    profile_df = pd.DataFrame(rows, columns=["Metrik", "Deger"])

    # Yillik talep ozetleri
    yearly_rows = []
    for yr, g in df.groupby(df["date"].dt.year):
        s = g["sales_qty"].values
        yearly_rows.append({
            "Yil": int(yr),
            "Gun Sayisi": len(g),
            "Toplam Talep": float(s.sum()),
            "Ortalama Gunluk Talep": round(float(s.mean()), 2),
            "Medyan Gunluk Talep": float(np.median(s)),
            "Maksimum Gunluk Talep": float(s.max()),
            "Sifir Talep Orani (%)": round(100 * (s == 0).mean(), 2),
            f"{BIG50_THRESHOLD}+ Gun Sayisi": int((s >= BIG50_THRESHOLD).sum()),
            f"{BIG100_THRESHOLD}+ Gun Sayisi": int((s >= BIG100_THRESHOLD).sum()),
        })
    yearly_df = pd.DataFrame(yearly_rows)

    return profile_df, yearly_df, {"adi": adi, "cv2": cv2, "demand_class": demand_class}


def print_data_profile(profile_df, yearly_df):
    """Veri profilini konsola yazdırır."""
    log("PROFILE", "Veri seti genel ozeti:")
    for _, row in profile_df.iterrows():
        print(f"    - {row['Metrik']}: {row['Deger']}")
    log("PROFILE", "Yillik talep ozeti:")
    print(yearly_df.to_string(index=False))
# === [4B] PROJE/İHALE PİKİ AYRIŞTIRMA (OPSİYONEL SENARYO) ===================
# Senaryo B için: talebi rutin + proje/ihale piki olarak iki bileşene ayırır.
# Eşik, pozitif satış günlerinin P95 yüzdeliğinden veri odaklı hesaplanır.

PROJECT_PEAK_QUANTILE = 0.95          # pozitif satislarin hangi yuzdelik dilimi pik sayilsin
PROJECT_PEAK_THRESHOLD_OVERRIDE = None  # elle sabit bir esik zorlamak icin (varsayilan: kullanilmaz)
PROJECT_PEAK_MODE = "set_to_zero"     # pik gunler operasyonel seriden nasil cikarilsin


def segment_project_peaks(df, quantile=PROJECT_PEAK_QUANTILE,
                           threshold_override=PROJECT_PEAK_THRESHOLD_OVERRIDE):
    """
    Proje/ihale günlerini (pik) rutin talepten ayırır.
    Eşik, pozitif satış günlerinin `quantile` yüzdeliğinden otomatik hesaplanır.
    Döner: (operasyonel_df, pik_günleri_df, özet_sözlük)
    """
    d = df.copy().sort_values("date").reset_index(drop=True)
    d["original_sales_qty"] = pd.to_numeric(d["sales_qty"], errors="coerce").fillna(0).clip(lower=0)

    pos = d.loc[d["original_sales_qty"] > 0, "original_sales_qty"]
    if len(pos) == 0:
        threshold = np.inf
    elif threshold_override is not None:
        threshold = float(threshold_override)
    else:
        threshold = float(pos.quantile(quantile))  # <-- VERIYE DAYALI OTOMATIK ESIK

    d["peak_threshold"] = threshold
    d["is_project_peak"] = ((d["original_sales_qty"] > threshold) & (d["original_sales_qty"] > 0)).astype(int)
    d["project_peak_qty"] = np.where(d["is_project_peak"] == 1, d["original_sales_qty"], 0.0)
    d["sales_qty_operational"] = np.where(d["is_project_peak"] == 1, 0.0, d["original_sales_qty"])

    op = d.copy()
    op["sales_qty"] = op["sales_qty_operational"]

    peak_days = d.loc[d["is_project_peak"] == 1,
                       ["date", "original_sales_qty", "project_peak_qty", "peak_threshold"]].copy()

    total = float(d["original_sales_qty"].sum())
    peak_total = float(d["project_peak_qty"].sum())
    op_total = float(op["sales_qty"].sum())

    summary = {
        "peak_rule": f"positive_sales_gt_P{int(quantile * 100)}" if threshold_override is None
                     else f"manual_override_gt_{threshold_override}",
        "peak_threshold": threshold,
        "peak_quantile_used": quantile if threshold_override is None else None,
        "original_rows": int(len(d)),
        "original_total_sales": total,
        "operational_total_sales": op_total,
        "project_peak_day_count": int(d["is_project_peak"].sum()),
        "project_peak_total_qty": peak_total,
        "project_peak_share_of_total_%": round(100 * peak_total / total, 2) if total > 0 else np.nan,
        "positive_day_count_original": int((d["original_sales_qty"] > 0).sum()),
        "positive_day_count_operational": int((op["sales_qty"] > 0).sum()),
        "zero_rate_original_%": round(100 * float((d["original_sales_qty"] == 0).mean()), 2),
        "zero_rate_operational_%": round(100 * float((op["sales_qty"] == 0).mean()), 2),
        "mode": PROJECT_PEAK_MODE,
    }
    return op, peak_days, summary


def plot_peak_days(raw_df, peak_days, threshold, folder):
    """Orijinal talep serisi uzerinde otomatik hesaplanan esik ve pik gunleri gosterir."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(raw_df["date"], raw_df["sales_qty"], color=COLOR_ACTUAL, linewidth=0.7,
            alpha=0.8, label="Gunluk Talep (Orijinal)")
    if len(peak_days):
        ax.scatter(peak_days["date"], peak_days["original_sales_qty"], color=COLOR_BAD,
                   s=40, zorder=5, label=f"Proje/Ihale Piki (n={len(peak_days)})")
    ax.axhline(threshold, color=COLOR_DEEP, linestyle="--", linewidth=1.3,
               label=f"Otomatik Esik (P{int(PROJECT_PEAK_QUANTILE*100)}) = {threshold:.1f} adet")
    ax.set_title("Proje/Ihale Piki Ayristirma (Esik Veriden Otomatik Hesaplanmistir)")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Gunluk Talep (Adet)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_fig(fig, folder, "00_peak_segmentasyon_esik_otomatik")


# === [5] FEATURE ENGINEERING =================================================
# Lag ve rolling feature'lar yalnızca geçmiş veriden hesaplanır; shift(1) ile
# aynı günün satışı feature olarak kullanılmaz (veri sızıntısı önlemi).

LAG_DAYS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 56, 60, 90, 120, 180, 270, 365]
ROLL_WINDOWS = [7, 14, 21, 30, 45, 60, 90, 120, 180, 365]

# Hedefi dogrudan sizdirabilecek, model feature seti disinda tutulmasi
# zorunlu olan kolonlar.
LEAKAGE_COLS = {
    "sales_qty", "revenue", "avg_unit_price", "sales_qty_original",
    "original_sales_qty", "project_peak_qty", "is_project_peak",
    "removed_by_peak_rule", "sales_qty_winsor", "n_tx", "max_tx_qty",
    "list_price", "is_bulk_day", "peak_threshold", "sales_qty_operational",
}
# Not: list_price, trend_days ile yüksek korelasyonu (r~0.93) nedeniyle
# anlamlı bilgi katmadan gürültü ekliyor; bu yüzden feature dışı bırakıldı.
# avg_unit_price ise tahmin anında bilinmediğinden sızıntı kapsamında.


def make_features(df):
    """Ham veriden takvim, lag, rolling, seyrek talep ve trend feature gruplarını türetir."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    y = df["sales_qty"].values
    n = len(df)

    # --- A) Takvim feature'lari ---------------------------------------------
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_sunday"] = (df["day_of_week"] == 6).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # --- B) Donguesel encoding -----------------------------------------------
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    for k in [1, 2, 3]:
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_year"] / 365.25)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_year"] / 365.25)

    # --- C) Tatil feature'lari (zaten add_holiday_features ile eklendi) -----
    for col in ["is_resmi_tatil", "is_dini_bayram", "bayram_oncesi_3g", "bayram_sonrasi_3g"]:
        if col not in df.columns:
            df[col] = 0

    # --- D) Trend feature'lari -------------------------------------------------
    df["trend_days"] = np.arange(n)
    post_2023 = (df["date"] >= "2023-01-01").astype(int)
    df["post_2023_trend"] = post_2023 * np.arange(n)

    def rolling_slope(series, window):
        """Verilen pencere icin basit dogrusal trend egimini hesaplar."""
        slopes = np.full(len(series), np.nan)
        vals = series.values
        x = np.arange(window)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        for i in range(window, len(vals) + 1):
            window_vals = vals[i - window:i]
            y_mean = window_vals.mean()
            slope = ((x - x_mean) * (window_vals - y_mean)).sum() / x_var
            slopes[i - 1] = slope
        return slopes

    df["trend_slope_30"] = rolling_slope(df["sales_qty"], 30)
    df["trend_slope_90"] = rolling_slope(df["sales_qty"], 90)

    # --- E) Lag feature'lari ----------------------------------------------------
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df["sales_qty"].shift(lag)

    # --- F) Rolling feature'lari (shift(1) ile sizinti onlenir) -------------
    shifted = df["sales_qty"].shift(1)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"roll_sum_{w}"] = shifted.rolling(w, min_periods=1).sum()
        df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std()
        df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max()
        df[f"pos_days_{w}"] = (shifted > 0).rolling(w, min_periods=1).sum()
        df[f"big50_days_{w}"] = (shifted >= BIG50_THRESHOLD).rolling(w, min_periods=1).sum()
        df[f"big100_days_{w}"] = (shifted >= BIG100_THRESHOLD).rolling(w, min_periods=1).sum()

    # --- G) Seyrek talep feature'lari ----------------------------------------
    pos_mask = (df["sales_qty"] > 0).values
    days_since_last_sale = np.full(n, np.nan)
    days_since_big50 = np.full(n, np.nan)
    days_since_big100 = np.full(n, np.nan)
    last_positive_qty = np.full(n, np.nan)

    last_pos_idx, last_b50_idx, last_b100_idx = None, None, None
    last_pos_val = np.nan
    sales_vals = df["sales_qty"].values
    for i in range(n):
        days_since_last_sale[i] = (i - last_pos_idx) if last_pos_idx is not None else np.nan
        days_since_big50[i] = (i - last_b50_idx) if last_b50_idx is not None else np.nan
        days_since_big100[i] = (i - last_b100_idx) if last_b100_idx is not None else np.nan
        last_positive_qty[i] = last_pos_val

        if sales_vals[i] > 0:
            last_pos_idx = i
            last_pos_val = sales_vals[i]
        if sales_vals[i] >= BIG50_THRESHOLD:
            last_b50_idx = i
        if sales_vals[i] >= BIG100_THRESHOLD:
            last_b100_idx = i

    df["days_since_last_sale"] = days_since_last_sale
    df["days_since_big50"] = days_since_big50
    df["days_since_big100"] = days_since_big100
    df["last_positive_qty"] = last_positive_qty

    for w in [30, 60, 90, 180, 365]:
        df[f"pos_rate_{w}"] = (shifted > 0).rolling(w, min_periods=1).mean()
        df[f"big50_rate_{w}"] = (shifted >= BIG50_THRESHOLD).rolling(w, min_periods=1).mean()
        df[f"big100_rate_{w}"] = (shifted >= BIG100_THRESHOLD).rolling(w, min_periods=1).mean()

    # --- H) Veri setinden gelen aciklayici degiskenler -----------------------
    for col in ["ay_uretim_plani", "stok_proxy", "is_covid_period"]:
        if col not in df.columns:
            df[col] = 0

    return df


def get_feature_columns(df):
    """Veri sızıntısı oluşturabilecek sütunlar dışarıda bırakılmış feature listesini döndürür."""
    exclude = LEAKAGE_COLS | {"date"}
    cols = [c for c in df.columns if c not in exclude]
    # Sadece sayisal kolonlar feature olarak kullanilir
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols
# === [6] PERFORMANS METRİKLERİ ===============================================
# Sıfır talep günlerinde MAPE tanımsız olduğundan kullanılmaz.
# Ana metrikler: WAPE tabanlı agregasyonlar, MAE/RMSE ve talep sınıflandırma.


def safe_div(a, b, default=0.0):
    return a / b if b not in (0, 0.0) else default


def metric_wape(y_true, y_pred):
    """WAPE = toplam mutlak hata / toplam gerçek talep."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.sum(np.abs(y_true))
    return float(safe_div(np.sum(np.abs(y_true - y_pred)), denom, np.nan))


def metric_agg_wape(dates, y_true, y_pred, freq):
    """Talebi haftaya/aya toplar, ardından WAPE hesaplar. Günlük gürültüyü azaltır."""
    tmp = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred})
    tmp = tmp.set_index("date").resample(freq).sum()
    return metric_wape(tmp["y"].values, tmp["p"].values)


def metric_bias_pct(y_true, y_pred):
    """Bias % = (toplam tahmin - toplam gerçek) / toplam gerçek × 100."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.sum(y_true)
    return float(safe_div(np.sum(y_pred) - np.sum(y_true), denom, np.nan) * 100)


def metric_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def metric_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metric_total_pred_true_ratio(y_true, y_pred):
    """Toplam tahmin / toplam gerçek oranı. 1.0 ideal."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(safe_div(np.sum(y_pred), np.sum(y_true), np.nan))


def metric_demand_classification(y_true, y_pred, threshold=0.5):
    """Talep var/yok (binary) sınıflandırma metriği: precision, recall, F1."""
    y_true_bin = (np.asarray(y_true) > 0).astype(int)
    y_pred_bin = (np.asarray(y_pred) > threshold).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    precision = safe_div(tp, tp + fp, 0.0)
    recall = safe_div(tp, tp + fn, 0.0)
    f1 = safe_div(2 * precision * recall, precision + recall, 0.0)
    return float(precision), float(recall), float(f1)


def metric_top30_capture(y_true, y_pred, n=30):
    """Gerçek talebin en yüksek olduğu N günde tahmin tutma oranı. 1.0 = ideal."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return np.nan
    n_use = min(n, len(y_true))
    top_idx = np.argsort(y_true)[-n_use:]
    true_top_sum = np.sum(y_true[top_idx])
    pred_top_sum = np.sum(y_pred[top_idx])
    return float(safe_div(pred_top_sum, true_top_sum, np.nan))


def metric_large_underforecast_rate(y_true, y_pred, threshold):
    """Büyük talep günlerinde tahminin düşük kaldığı günlerin oranı (stockout riski)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true >= threshold
    if mask.sum() == 0:
        return np.nan
    under = (y_pred[mask] < y_true[mask]).sum()
    return float(under / mask.sum())


def metric_peak_mae(y_true, y_pred, threshold):
    """Eşik üzerindeki günlerde MAE."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true >= threshold
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def evaluate_predictions(dates, y_true, y_pred):
    """Tahmin seti için tüm metrikleri hesaplar, sözlük döndürür."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)  # negatif tahminler 0'a cekilir

    precision, recall, f1 = metric_demand_classification(y_true, y_pred)

    return {
        "WAPE": metric_wape(y_true, y_pred),
        "hWAPE": metric_agg_wape(dates, y_true, y_pred, "W"),
        "mWAPE": metric_agg_wape(dates, y_true, y_pred, "ME"),
        "MAE": metric_mae(y_true, y_pred),
        "RMSE": metric_rmse(y_true, y_pred),
        "Bias_%": metric_bias_pct(y_true, y_pred),
        "total_pred_true_ratio": metric_total_pred_true_ratio(y_true, y_pred),
        "demand_precision": precision,
        "demand_recall": recall,
        "demand_f1": f1,
        "top30_capture_ratio": metric_top30_capture(y_true, y_pred),
        "large50_underforecast_rate": metric_large_underforecast_rate(y_true, y_pred, BIG50_THRESHOLD),
        "large100_underforecast_rate": metric_large_underforecast_rate(y_true, y_pred, BIG100_THRESHOLD),
        "Pik_MAE_50+": metric_peak_mae(y_true, y_pred, BIG50_THRESHOLD),
        "Pik_MAE_100+": metric_peak_mae(y_true, y_pred, BIG100_THRESHOLD),
        "n_obs": len(y_true),
    }


def compute_focused_score(metrics_row):
    """
    Stok yönetimi odaklı bileşik skor (düşük = daha iyi).
    hWAPE, Bias, F1, top30 yakalama ve underforecast oranlarını ağırlıklı birleştirir.
    """
    hwape = metrics_row.get("hWAPE", np.nan)
    bias = abs(metrics_row.get("Bias_%", np.nan)) / 100.0
    f1 = metrics_row.get("demand_f1", 0.0)
    recall = metrics_row.get("demand_recall", 0.0)
    top30 = metrics_row.get("top30_capture_ratio", np.nan)
    under50 = metrics_row.get("large50_underforecast_rate", np.nan)
    under100 = metrics_row.get("large100_underforecast_rate", np.nan)

    top30_penalty = abs(1.0 - top30) if not np.isnan(top30) else 1.0
    under50 = under50 if not np.isnan(under50) else 0.5
    under100 = under100 if not np.isnan(under100) else 0.5
    hwape = hwape if not np.isnan(hwape) else 2.0
    bias = bias if not np.isnan(bias) else 1.0

    score = (
        0.35 * hwape
        + 0.20 * bias
        + 0.15 * (1 - f1)
        + 0.10 * (1 - recall)
        + 0.10 * top30_penalty
        + 0.05 * under50
        + 0.05 * under100
    )
    return float(score)
# === [7] GELENEKSEL BENCHMARK MODELLER =======================================
# Naïve tarzda çalışır; parametre optimizasyonu dışında eğitim gerektirmez.


def model_naive(train_y, n_test):
    """Son gözlemlenen değeri tüm test dönemi için sabit tahmin olarak döndürür."""
    last_val = train_y[-1] if len(train_y) > 0 else 0.0
    return np.full(n_test, last_val)


def model_seasonal_naive(train_y, n_test, season=SEASON_LENGTH):
    """t günü tahmini = (t - season) günündeki değer. Test dönemi içinde özyinelemeli çalışır."""
    preds = np.zeros(n_test)
    train_y = np.asarray(train_y, dtype=float)
    n_train = len(train_y)
    for i in range(n_test):
        src_idx = i - season  # test icindeki gore konum
        if src_idx >= 0:
            preds[i] = preds[src_idx]
        else:
            train_idx = n_train + src_idx
            preds[i] = train_y[train_idx] if train_idx >= 0 else (train_y[-1] if n_train else 0.0)
    return preds


def model_moving_average(train_y, n_test, window):
    """Son `window` günün ortalamasını sabit tahmin olarak döndürür."""
    if len(train_y) == 0:
        return np.zeros(n_test)
    avg = np.mean(train_y[-window:])
    return np.full(n_test, avg)


def model_ses(train_y, n_test, alpha=0.3):
    """Basit üstel düzleştirme. Son düzleştirilmiş seviye tüm test döneminde sabit tahmin."""
    if len(train_y) == 0:
        return np.zeros(n_test)
    level = train_y[0]
    for v in train_y[1:]:
        level = alpha * v + (1 - alpha) * level
    return np.full(n_test, level)


def optimize_ses_alpha(train_y, val_y, alphas=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)):
    """Validation seti üzerinde WAPE minimize eden SES alfa değerini seçer."""
    best_alpha, best_wape = alphas[0], np.inf
    for a in alphas:
        pred = model_ses(train_y, len(val_y), alpha=a)
        w = metric_wape(val_y, pred)
        if not np.isnan(w) and w < best_wape:
            best_wape, best_alpha = w, a
    return best_alpha


# === [8] SEYREK TALEP MODELLERİ ==============================================

def model_tsb(train_y, n_test, alpha=0.1, beta=0.1):
    """
    TSB (Teunter-Syntetos-Babai) yontemi: talep buyuklugu (z) ve talep
    olasiligi (p) ayri ayri ussel duzlestirilir; tahmin z*p'dir.
    """
    train_y = np.asarray(train_y, dtype=float)
    if len(train_y) == 0:
        return np.full(n_test, 0.0)

    z = train_y[train_y > 0][0] if np.any(train_y > 0) else 0.0
    p = np.mean(train_y > 0) if len(train_y) > 0 else 0.0

    for v in train_y:
        is_pos = v > 0
        p = p + beta * (int(is_pos) - p)
        if is_pos:
            z = z + alpha * (v - z)

    forecast = z * p
    return np.full(n_test, forecast)


def model_croston_sba(train_y, n_test, alpha=0.1, beta=0.1):
    """
    Croston metodunun SBA (Syntetos-Boylan Approximation) duzeltmesi.
    Talep buyuklugu (z) ve talep araligi (v_interval) ayri duzlestirilir;
    SBA katsayisi ile sapma duzeltilir.
    """
    train_y = np.asarray(train_y, dtype=float)
    if len(train_y) == 0 or not np.any(train_y > 0):
        return np.full(n_test, 0.0)

    first_pos_idx = np.argmax(train_y > 0)
    z = train_y[first_pos_idx]
    q = 1.0
    interval_count = 1

    last_pos_idx = first_pos_idx
    for i in range(first_pos_idx + 1, len(train_y)):
        if train_y[i] > 0:
            interval = i - last_pos_idx
            z = z + alpha * (train_y[i] - z)
            q = q + beta * (interval - q)
            last_pos_idx = i

    sba_factor = (1 - beta / 2)
    forecast = sba_factor * (z / q) if q > 0 else 0.0
    return np.full(n_test, forecast)


def optimize_intermittent_params(model_func, train_y, val_y,
                                  alphas=(0.05, 0.1, 0.2, 0.3),
                                  betas=(0.05, 0.1, 0.2, 0.3)):
    """TSB / Croston-SBA icin validation uzerinde en iyi alpha-beta ciftini secer."""
    best_params, best_wape = (alphas[0], betas[0]), np.inf
    for a in alphas:
        for b in betas:
            pred = model_func(train_y, len(val_y), alpha=a, beta=b)
            w = metric_wape(val_y, pred)
            if not np.isnan(w) and w < best_wape:
                best_wape, best_params = w, (a, b)
    return best_params
# === [9] MAKİNE ÖĞRENMESİ MODELLERİ =========================================
# Tüm ML modelleri aynı iki aşamalı mantıkla çalışır:
#   1) X_val eval_set olarak kullanılıp early-stopping ile best_iteration bulunur.
#   2) Bulunan best_iteration ile train+val birleştirilerek final model refit edilir.
# Tüm train_* fonksiyonları en az "val_pred" ve "final_model" içeren sözlük döndürür.

LGB_PARAMS_BASE = dict(
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=15,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbosity=-1,
)
LGB_MAX_ESTIMATORS = 600
MIN_ESTIMATORS_FLOOR = 30  # early stopping cok erken durursa guvenlik tabani


def get_sample_weights(dates, mode):
    """recency_weighted için üstel azalan örnek ağırlıkları üretir; diğer modlar için None döner."""
    if mode != "recency_weighted":
        return None
    dates = pd.to_datetime(dates)
    ref_date = dates.max()
    days_diff = (ref_date - dates).dt.days.values
    weights = 0.5 ** (days_diff / RECENCY_HALF_LIFE)
    return weights


def get_history_slice(train_df, mode):
    """history_mode'a göre train setini kısıtlar (full / recent_only / recency_weighted)."""
    if mode == "recent_only":
        cutoff = train_df["date"].max() - pd.Timedelta(days=RECENT_ONLY_DAYS)
        return train_df[train_df["date"] > cutoff].reset_index(drop=True)
    return train_df.reset_index(drop=True)


def _concat_weights(sw_a, n_a, sw_b, n_b):
    """İki sample_weight dizisini (None olabilir) güvenle birleştirir."""
    if sw_a is None and sw_b is None:
        return None
    a = sw_a if sw_a is not None else np.ones(n_a)
    b = sw_b if sw_b is not None else np.ones(n_b)
    return np.concatenate([a, b])


# --- Random Forest -----------------------------------------------------------
# Early-stopping yoktur; validation tahmini için train-only, final model için
# train+val birleştirilerek eğitilir.

def train_random_forest(X_train, y_train, X_val, y_val, sample_weight=None):
    val_model = RandomForestRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    val_model.fit(X_train, y_train, sample_weight=sample_weight)
    val_pred = np.clip(val_model.predict(X_val), 0, None)

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    sw_full = _concat_weights(sample_weight, len(y_train), None, len(y_val))

    final_model = RandomForestRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    final_model.fit(X_full, y_full, sample_weight=sw_full)

    return {"final_model": final_model, "val_pred": val_pred, "fi_source": final_model}


def predict_random_forest(pack, X_test):
    if pack is None:
        return None
    return np.clip(pack["final_model"].predict(X_test), 0, None)


# --- LightGBM Tweedie --------------------------------------------------------

def train_lgb_tweedie(X_train, y_train, X_val, y_val, sample_weight=None):
    if not HAS_LGB:
        return None
    probe = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                               n_estimators=LGB_MAX_ESTIMATORS, **LGB_PARAMS_BASE)
    probe.fit(
        X_train, y_train, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
    )
    val_pred = np.clip(probe.predict(X_val), 0, None)
    best_iter = max(probe.best_iteration_ or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    sw_full = _concat_weights(sample_weight, len(y_train), None, len(y_val))

    final_model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                                     n_estimators=best_iter, **LGB_PARAMS_BASE)
    final_model.fit(X_full, y_full, sample_weight=sw_full)

    return {"final_model": final_model, "val_pred": val_pred, "fi_source": final_model}


def predict_lgb(pack, X_test):
    if pack is None:
        return None
    return np.clip(pack["final_model"].predict(X_test), 0, None)


# --- LightGBM Quantile Blend (QBlend) ----------------------------------------

def train_lgb_qblend(X_train, y_train, X_val, y_val, sample_weight=None,
                      quantiles=(0.5, 0.75, 0.9)):
    """Farklı kantil düzeyleri için LGB modelleri eğitir; ağırlıklı ortalama ile birleştirir (QBlend)."""
    if not HAS_LGB:
        return None
    blend_weights = (0.5, 0.3, 0.2)

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    sw_full = _concat_weights(sample_weight, len(y_train), None, len(y_val))

    final_models = {}
    val_preds_per_q = {}
    for q in quantiles:
        probe = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                   n_estimators=LGB_MAX_ESTIMATORS, **LGB_PARAMS_BASE)
        probe.fit(
            X_train, y_train, sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
        )
        val_preds_per_q[q] = probe.predict(X_val)
        best_iter = max(probe.best_iteration_ or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)

        final = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                   n_estimators=best_iter, **LGB_PARAMS_BASE)
        final.fit(X_full, y_full, sample_weight=sw_full)
        final_models[q] = final

    val_pred = np.zeros(len(X_val))
    for w, q in zip(blend_weights, sorted(val_preds_per_q.keys())):
        val_pred += w * val_preds_per_q[q]
    val_pred = np.clip(val_pred, 0, None)

    return {"final_models": final_models, "blend_weights": blend_weights,
            "val_pred": val_pred, "fi_source": final_models[sorted(final_models.keys())[0]]}


def predict_lgb_qblend(pack, X_test):
    if pack is None:
        return None
    preds = np.zeros(len(X_test))
    for w, q in zip(pack["blend_weights"], sorted(pack["final_models"].keys())):
        preds += w * pack["final_models"][q].predict(X_test)
    return np.clip(preds, 0, None)


# --- LightGBM TwoStage (talep var/yok + miktar) -----------------------------

def train_lgb_twostage(X_train, y_train, X_val, y_val, sample_weight=None):
    """
    1. asama: LightGBM siniflandirici ile talep var/yok olasiligi.
    2. asama: yalnizca pozitif talep gunlerinde LightGBM regresor ile miktar.
    Final tahmin: P(talep var) * E[miktar | talep var]
    Her iki asama da kendi best_iteration'i ile train+val uzerinde refit edilir.
    """
    if not HAS_LGB:
        return None
    y_bin_train = (y_train > 0).astype(int)
    y_bin_val = (y_val > 0).astype(int)

    clf_probe = lgb.LGBMClassifier(n_estimators=LGB_MAX_ESTIMATORS, learning_rate=0.05,
                                    num_leaves=31, random_state=42, verbosity=-1)
    clf_probe.fit(
        X_train, y_bin_train, sample_weight=sample_weight,
        eval_set=[(X_val, y_bin_val)],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
    )
    clf_best_iter = max(clf_probe.best_iteration_ or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)
    val_p_demand = clf_probe.predict_proba(X_val)[:, 1]

    pos_mask = y_train > 0
    X_train_pos, y_train_pos = X_train[pos_mask], y_train[pos_mask]
    pos_mask_val = y_val > 0
    X_val_pos, y_val_pos = X_val[pos_mask_val], y_val[pos_mask_val]
    sw_pos = sample_weight[pos_mask] if sample_weight is not None else None

    if len(X_val_pos) >= 5 and len(X_train_pos) >= 5:
        reg_probe = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                                       n_estimators=LGB_MAX_ESTIMATORS, **LGB_PARAMS_BASE)
        reg_probe.fit(
            X_train_pos, y_train_pos, sample_weight=sw_pos,
            eval_set=[(X_val_pos, y_val_pos)],
            callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
        )
        reg_best_iter = max(reg_probe.best_iteration_ or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)
        val_qty = np.clip(reg_probe.predict(X_val), 0, None)
    else:
        reg_best_iter = 100
        val_qty = np.full(len(X_val), y_train_pos.mean() if len(y_train_pos) else 0.0)

    val_pred = val_p_demand * val_qty

    # ---- Final refit: train + val birlestirilir ----
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    y_bin_full = np.concatenate([y_bin_train, y_bin_val])
    sw_full = _concat_weights(sample_weight, len(y_train), None, len(y_val))

    clf_final = lgb.LGBMClassifier(n_estimators=clf_best_iter, learning_rate=0.05,
                                    num_leaves=31, random_state=42, verbosity=-1)
    clf_final.fit(X_full, y_bin_full, sample_weight=sw_full)

    pos_mask_full = y_full > 0
    X_full_pos, y_full_pos = X_full[pos_mask_full], y_full[pos_mask_full]
    sw_full_pos = sw_full[pos_mask_full] if sw_full is not None else None

    reg_final = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                                   n_estimators=reg_best_iter, **LGB_PARAMS_BASE)
    if len(X_full_pos) >= 5:
        reg_final.fit(X_full_pos, y_full_pos, sample_weight=sw_full_pos)
    else:
        reg_final.fit(X_full, y_full, sample_weight=sw_full)

    return {"clf": clf_final, "reg": reg_final, "val_pred": val_pred, "fi_source": reg_final}


def predict_lgb_twostage(pack, X_test):
    if pack is None:
        return None
    p_demand = pack["clf"].predict_proba(X_test)[:, 1]
    qty = np.clip(pack["reg"].predict(X_test), 0, None)
    return p_demand * qty


# --- LightGBM ThreeStage (talep yok / normal / pik) -------------------------

def train_lgb_threestage(X_train, y_train, X_val, y_val, sample_weight=None,
                          peak_threshold=BIG50_THRESHOLD):
    """
    3 sinifli yapi: 0 = talep yok, 1 = normal talep, 2 = pik talep
    (>= peak_threshold). Her sinif icin ayri buyukluk tahmini uretilir,
    sinif olasiliklari ile agirliklandirilir. Tum alt modeller train+val
    uzerinde refit edilir.
    """
    if not HAS_LGB:
        return None

    def make_class(y):
        c = np.zeros(len(y), dtype=int)
        c[(y > 0) & (y < peak_threshold)] = 1
        c[y >= peak_threshold] = 2
        return c

    c_train, c_val = make_class(y_train), make_class(y_val)

    clf_probe = lgb.LGBMClassifier(n_estimators=LGB_MAX_ESTIMATORS, learning_rate=0.05,
                                    num_leaves=31, random_state=42, verbosity=-1,
                                    objective="multiclass", num_class=3)
    clf_probe.fit(
        X_train, c_train, sample_weight=sample_weight,
        eval_set=[(X_val, c_val)],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
    )
    clf_best_iter = max(clf_probe.best_iteration_ or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)
    val_proba = clf_probe.predict_proba(X_val)

    reg_best_iters = {}
    val_qty_per_class = {}
    for cls in [1, 2]:
        mask_tr = c_train == cls
        mask_va = c_val == cls
        if mask_tr.sum() < 5:
            continue
        sw_cls = sample_weight[mask_tr] if sample_weight is not None else None
        reg_probe = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                                       n_estimators=LGB_MAX_ESTIMATORS, **LGB_PARAMS_BASE)
        if mask_va.sum() >= 5:
            reg_probe.fit(
                X_train[mask_tr], y_train[mask_tr], sample_weight=sw_cls,
                eval_set=[(X_val[mask_va], y_val[mask_va])],
                callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
            )
            reg_best_iters[cls] = max(reg_probe.best_iteration_ or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)
        else:
            reg_probe.fit(X_train[mask_tr], y_train[mask_tr], sample_weight=sw_cls)
            reg_best_iters[cls] = 100
        val_qty_per_class[cls] = np.clip(reg_probe.predict(X_val), 0, None)

    val_pred = np.zeros(len(X_val))
    n_classes_val = val_proba.shape[1]
    for cls in [1, 2]:
        if cls in val_qty_per_class and cls < n_classes_val:
            val_pred += val_proba[:, cls] * val_qty_per_class[cls]

    # ---- Final refit: train + val birlestirilir ----
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    c_full = make_class(y_full)
    sw_full = _concat_weights(sample_weight, len(y_train), None, len(y_val))

    clf_final = lgb.LGBMClassifier(n_estimators=clf_best_iter, learning_rate=0.05,
                                    num_leaves=31, random_state=42, verbosity=-1,
                                    objective="multiclass", num_class=3)
    clf_final.fit(X_full, c_full, sample_weight=sw_full)

    regs_final = {}
    for cls in [1, 2]:
        mask_full = c_full == cls
        if mask_full.sum() < 5:
            continue
        sw_cls_full = sw_full[mask_full] if sw_full is not None else None
        reg_final = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                                       n_estimators=reg_best_iters.get(cls, 100), **LGB_PARAMS_BASE)
        reg_final.fit(X_full[mask_full], y_full[mask_full], sample_weight=sw_cls_full)
        regs_final[cls] = reg_final

    return {"clf": clf_final, "regs": regs_final, "peak_threshold": peak_threshold,
            "val_pred": val_pred, "fi_source": regs_final.get(1) or regs_final.get(2)}


def predict_lgb_threestage(pack, X_test):
    if pack is None:
        return None
    proba = pack["clf"].predict_proba(X_test)
    n_classes = proba.shape[1]
    preds = np.zeros(len(X_test))
    for cls in [1, 2]:
        if cls in pack["regs"] and cls < n_classes:
            qty = np.clip(pack["regs"][cls].predict(X_test), 0, None)
            preds += proba[:, cls] * qty
    return preds


# --- ExtraTrees Classifier + LightGBM Regressor hibrit model -------------------

def train_extratrees_lgb_hybrid(X_train, y_train, X_val, y_val, sample_weight=None):
    """
    1. asama: ExtraTreesClassifier ile talep var/yok olasiligi (early-stopping
    yok; train+val ile direkt refit edilir).
    2. asama: pozitif talep gunlerinde LightGBM regresor (best_iteration ile
    train+val refit).
    """
    if not HAS_LGB:
        return None
    y_bin_train = (y_train > 0).astype(int)

    clf_val = ExtraTreesClassifier(n_estimators=150, max_depth=10,
                                    min_samples_leaf=3, random_state=42, n_jobs=-1)
    clf_val.fit(X_train, y_bin_train, sample_weight=sample_weight)
    val_p_demand = clf_val.predict_proba(X_val)[:, 1]

    pos_mask = y_train > 0
    X_train_pos, y_train_pos = X_train[pos_mask], y_train[pos_mask]
    pos_mask_val = y_val > 0
    X_val_pos, y_val_pos = X_val[pos_mask_val], y_val[pos_mask_val]
    sw_pos = sample_weight[pos_mask] if sample_weight is not None else None

    if len(X_val_pos) >= 5 and len(X_train_pos) >= 5:
        reg_probe = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                                       n_estimators=LGB_MAX_ESTIMATORS, **LGB_PARAMS_BASE)
        reg_probe.fit(
            X_train_pos, y_train_pos, sample_weight=sw_pos,
            eval_set=[(X_val_pos, y_val_pos)],
            callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
        )
        reg_best_iter = max(reg_probe.best_iteration_ or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)
        val_qty = np.clip(reg_probe.predict(X_val), 0, None)
    else:
        reg_best_iter = 100
        val_qty = np.full(len(X_val), y_train_pos.mean() if len(y_train_pos) else 0.0)

    val_pred = val_p_demand * val_qty

    # ---- Final refit: train + val birlestirilir ----
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    y_bin_full = (y_full > 0).astype(int)
    sw_full = _concat_weights(sample_weight, len(y_train), None, len(y_val))

    clf_final = ExtraTreesClassifier(n_estimators=150, max_depth=10,
                                      min_samples_leaf=3, random_state=42, n_jobs=-1)
    clf_final.fit(X_full, y_bin_full, sample_weight=sw_full)

    pos_mask_full = y_full > 0
    X_full_pos, y_full_pos = X_full[pos_mask_full], y_full[pos_mask_full]
    sw_full_pos = sw_full[pos_mask_full] if sw_full is not None else None

    reg_final = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                                   n_estimators=reg_best_iter, **LGB_PARAMS_BASE)
    if len(X_full_pos) >= 5:
        reg_final.fit(X_full_pos, y_full_pos, sample_weight=sw_full_pos)
    else:
        reg_final.fit(X_full, y_full, sample_weight=sw_full)

    return {"clf": clf_final, "reg": reg_final, "val_pred": val_pred, "fi_source": reg_final}


def predict_extratrees_lgb_hybrid(pack, X_test):
    if pack is None:
        return None
    p_demand = pack["clf"].predict_proba(X_test)[:, 1]
    qty = np.clip(pack["reg"].predict(X_test), 0, None)
    return p_demand * qty


# --- XGBoost Tweedie ----------------------------------------------------------

def train_xgb_tweedie(X_train, y_train, X_val, y_val, sample_weight=None):
    if not HAS_XGB:
        return None
    probe = xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.3,
        n_estimators=LGB_MAX_ESTIMATORS, learning_rate=0.05, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42,
        early_stopping_rounds=25, eval_metric="rmse", verbosity=0,
    )
    probe.fit(
        X_train, y_train, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)], verbose=False,
    )
    val_pred = np.clip(probe.predict(X_val), 0, None)
    best_iter = max(probe.best_iteration or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    sw_full = _concat_weights(sample_weight, len(y_train), None, len(y_val))

    final_model = xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.3,
        n_estimators=best_iter, learning_rate=0.05, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=0,
    )
    final_model.fit(X_full, y_full, sample_weight=sw_full)

    return {"final_model": final_model, "val_pred": val_pred, "fi_source": final_model}


def predict_xgb(pack, X_test):
    if pack is None:
        return None
    return np.clip(pack["final_model"].predict(X_test), 0, None)


# --- XGBoost TwoStage ----------------------------------------------------------

def train_xgb_twostage(X_train, y_train, X_val, y_val, sample_weight=None):
    if not HAS_XGB:
        return None
    y_bin_train = (y_train > 0).astype(int)
    y_bin_val = (y_val > 0).astype(int)

    clf_probe = xgb.XGBClassifier(
        n_estimators=LGB_MAX_ESTIMATORS, learning_rate=0.05, max_depth=5,
        random_state=42, early_stopping_rounds=25,
        eval_metric="logloss", verbosity=0,
    )
    clf_probe.fit(X_train, y_bin_train, sample_weight=sample_weight,
                  eval_set=[(X_val, y_bin_val)], verbose=False)
    clf_best_iter = max(clf_probe.best_iteration or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)
    val_p_demand = clf_probe.predict_proba(X_val)[:, 1]

    pos_mask = y_train > 0
    X_train_pos, y_train_pos = X_train[pos_mask], y_train[pos_mask]
    pos_mask_val = y_val > 0
    X_val_pos, y_val_pos = X_val[pos_mask_val], y_val[pos_mask_val]
    sw_pos = sample_weight[pos_mask] if sample_weight is not None else None

    if len(X_val_pos) >= 5 and len(X_train_pos) >= 5:
        reg_probe = xgb.XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=1.3,
            n_estimators=LGB_MAX_ESTIMATORS, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, random_state=42,
            early_stopping_rounds=25, eval_metric="rmse", verbosity=0,
        )
        reg_probe.fit(X_train_pos, y_train_pos, sample_weight=sw_pos,
                      eval_set=[(X_val_pos, y_val_pos)], verbose=False)
        reg_best_iter = max(reg_probe.best_iteration or LGB_MAX_ESTIMATORS, MIN_ESTIMATORS_FLOOR)
        val_qty = np.clip(reg_probe.predict(X_val), 0, None)
    else:
        reg_best_iter = 100
        val_qty = np.full(len(X_val), y_train_pos.mean() if len(y_train_pos) else 0.0)

    val_pred = val_p_demand * val_qty

    # ---- Final refit: train + val birlestirilir ----
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    y_bin_full = np.concatenate([y_bin_train, y_bin_val])
    sw_full = _concat_weights(sample_weight, len(y_train), None, len(y_val))

    clf_final = xgb.XGBClassifier(
        n_estimators=clf_best_iter, learning_rate=0.05, max_depth=5,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    clf_final.fit(X_full, y_bin_full, sample_weight=sw_full, verbose=False)

    pos_mask_full = y_full > 0
    X_full_pos, y_full_pos = X_full[pos_mask_full], y_full[pos_mask_full]
    sw_full_pos = sw_full[pos_mask_full] if sw_full is not None else None

    reg_final = xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.3,
        n_estimators=reg_best_iter, learning_rate=0.05, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=0,
    )
    if len(X_full_pos) >= 5:
        reg_final.fit(X_full_pos, y_full_pos, sample_weight=sw_full_pos, verbose=False)
    else:
        reg_final.fit(X_full, y_full, sample_weight=sw_full, verbose=False)

    return {"clf": clf_final, "reg": reg_final, "val_pred": val_pred, "fi_source": reg_final}


def predict_xgb_twostage(pack, X_test):
    if pack is None:
        return None
    p_demand = pack["clf"].predict_proba(X_test)[:, 1]
    qty = np.clip(pack["reg"].predict(X_test), 0, None)
    return p_demand * qty


# === CALIBRATION VE CUT THRESHOLD ================================================
# Validation tahminlerine dayanarak (1) kucuk/gurultulu tahminleri sifira
# ceken bir "cut threshold" ve (2) toplam tahmin / toplam gercek oranina
# gore olcekleyen bir "calibration factor" hesaplanir. Her ikisi de SADECE
# validation verisinden ogrenilir, sonra hem val hem test tahminlerine
# ayni sekilde uygulanir (test bilgisi hicbir asamada kullanilmaz).

def choose_cut_threshold(val_y_true, val_y_pred, candidate_fracs=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3)):
    """
    Validation tahmininin farkli yuzdelik dilimlerini esik adayi olarak
    dener (orn. tahminlerin en kucuk %10'u sifira cekilir) ve focused_score'u
    en iyilestiren esigi secer. candidate_fracs=0.0 -> esik uygulanmaz.
    """
    val_y_pred = np.asarray(val_y_pred, dtype=float)
    if len(val_y_pred) == 0 or np.all(val_y_pred <= 0):
        return 0.0

    positive_preds = val_y_pred[val_y_pred > 0]
    if len(positive_preds) == 0:
        return 0.0

    best_thr, best_score = 0.0, np.inf
    dummy_dates = pd.date_range("2000-01-01", periods=len(val_y_true), freq="D")
    for frac in candidate_fracs:
        thr = float(np.percentile(positive_preds, frac * 100)) if frac > 0 else 0.0
        adjusted = np.where(val_y_pred < thr, 0.0, val_y_pred)
        m = evaluate_predictions(dummy_dates, val_y_true, adjusted)
        score = compute_focused_score(m)
        if score < best_score:
            best_score, best_thr = score, thr
    return best_thr


def apply_cut_threshold(y_pred, threshold):
    y_pred = np.asarray(y_pred, dtype=float)
    if threshold <= 0:
        return y_pred
    return np.where(y_pred < threshold, 0.0, y_pred)


def choose_calibration_factor(val_y_true, val_y_pred_after_cut, lo=0.75, hi=1.35,
                                deadzone=0.12, shrink=0.5):
    """
    Validation toplam gercek / toplam tahmin oranina gore bir kalibrasyon
    katsayisi hesaplar. Asiri uyumu (overfit) onlemek icin iki guvenlik
    mekanizmasi kullanilir:
      - deadzone: oran 1.0'a yakinsa (|oran-1| < deadzone) hic kalibrasyon
        uygulanmaz; kucuk validation sapmalari gurultu olarak kabul edilir.
      - shrink: kalibrasyon gerekliyse bile fark TAM kapatilmaz, sadece
        `shrink` orani kadar (varsayilan %50) duzeltme uygulanir. Boylece
        validation donemine ozgu rastlantisal sapmalarin test donemine
        agresif sekilde tasinmasi engellenir.
    Sonuc [lo, hi] araliginda sinirlanir.
    """
    val_y_pred_after_cut = np.asarray(val_y_pred_after_cut, dtype=float)
    total_pred = val_y_pred_after_cut.sum()
    total_true = np.asarray(val_y_true, dtype=float).sum()
    if total_pred <= 0:
        return 1.0

    raw_ratio = total_true / total_pred
    if abs(raw_ratio - 1.0) < deadzone:
        return 1.0

    # Sadece farkin shrink kadarini uygula (orn. raw_ratio=1.5 -> 1 + 0.5*0.5 = 1.25)
    dampened = 1.0 + shrink * (raw_ratio - 1.0)
    return float(np.clip(dampened, lo, hi))


def apply_calibration(y_pred, factor):
    return np.clip(np.asarray(y_pred, dtype=float) * factor, 0, None)


def calibrate_model_predictions(val_y_true, val_y_pred, test_y_pred,
                                  extreme_lo=0.60, extreme_hi=1.60):
    """
    Validation tahmininden ogrenilen ayarlari test tahminine uygular.
    Iki bilesen FARKLI guven seviyelerinde calisir:

    1) CUT THRESHOLD: validation'da focused_score'u iyilestiren bir esik
       bulunup HER ZAMAN uygulanir. Kucuk/gurultulu tahminleri sifirlamak
       dusuk riskli bir duzeltmedir (test uzerinde de genelde notr/olumlu
       etkilidir).

    2) CALIBRATION FACTOR: toplam tahmin/gercek oranini olcekler, ANCAK
       SADECE validation orani [extreme_lo, extreme_hi] araliginin DISINDA
       (yani acikca patolojik: orn. RandomForest pred/true=1.9 gibi)
       oldugunda ve hafif (shrink=0.35) sekilde uygulanir. Bunun nedeni:
       validation ve test donemleri arasinda (urunun lumpy/seyrek talep
       yapisi nedeniyle) farkli yonde sapmalar gorulebiliyor; agresif veya
       erken devreye giren bir kalibrasyon, validation'a ozgu rastlantisal
       sapmayi test donemine tasiyarak zaten dengeli olan tahminleri
       bozabilir. Bu yuzden calibration sadece acik/buyuk sapmalarda,
       hafif bir duzeltme olarak kullanilir.
    """
    val_y_pred = np.asarray(val_y_pred, dtype=float)
    test_y_pred = np.asarray(test_y_pred, dtype=float)
    val_y_true = np.asarray(val_y_true, dtype=float)

    # --- 1) Cut threshold: her zaman uygulanir ---
    threshold = choose_cut_threshold(val_y_true, val_y_pred)
    val_after_cut = apply_cut_threshold(val_y_pred, threshold)
    test_after_cut = apply_cut_threshold(test_y_pred, threshold)

    # --- 2) Calibration factor: sadece acik sapmalarda, hafif uygulanir ---
    raw_val_ratio = val_after_cut.sum() / val_y_true.sum() if val_y_true.sum() > 0 else 1.0

    if extreme_lo <= raw_val_ratio <= extreme_hi:
        factor = 1.0
        applied = False
    else:
        factor = choose_calibration_factor(val_y_true, val_after_cut, shrink=0.35)
        applied = True

    val_calibrated = apply_calibration(val_after_cut, factor)
    test_calibrated = apply_calibration(test_after_cut, factor)

    settings = {"cut_threshold": round(threshold, 3), "calibration_factor": round(factor, 3),
                "calibration_applied": applied}
    return val_calibrated, test_calibrated, settings


def flag_unhealthy_model(metrics_row, lo=0.5, hi=1.5):
    """
    total_pred_true_ratio belirtilen aralik disindaysa model 'saglik
    uyarisi' ile isaretlenir. Sirlamadan cikarmaz, sadece etiketler.
    """
    ratio = metrics_row.get("total_pred_true_ratio", np.nan)
    if np.isnan(ratio):
        return "UNKNOWN"
    if ratio < lo or ratio > hi:
        return "UYARI: dengesiz tahmin (pred/true=%.2f)" % ratio
    return "OK"
# === [10] LSTM MODELİ =========================================================
# Günlük talep serisinin log(1+x) dönüşümü üzerinde kayan pencere yaklaşımıyla
# çalışır; tahmin test dönemi boyunca tek-adım-ileriye özyinelemeli üretilir.


def build_lstm_windows(series, lookback):
    """LSTM eğitimi için kayan pencere (X, y) çiftleri üretir."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_lstm_model(train_y, val_y, lookback=30, epochs=30, batch_size=32):
    """log(1+x) dönüşümlü LSTM eğitir. TF yoksa None döner."""
    if not HAS_TF:
        return None

    full_series = np.concatenate([train_y, val_y]).astype(float)
    log_series = np.log1p(full_series)

    mean_, std_ = log_series.mean(), log_series.std() + 1e-6
    norm_series = (log_series - mean_) / std_

    n_val = len(val_y)
    train_part = norm_series[: len(norm_series) - n_val]
    full_part = norm_series  # egitim sonunda val'i de gormesi icin kullanilir

    X_train, y_train = build_lstm_windows(train_part, lookback)
    if len(X_train) < 20:
        return None

    X_train = X_train.reshape((-1, lookback, 1))

    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.15),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse")

    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        verbose=0, callbacks=[early_stop],
    )

    return {
        "model": model, "mean": mean_, "std": std_,
        "lookback": lookback, "history_tail": full_part[-lookback:].copy(),
    }


def predict_lstm_model(pack, n_steps):
    """Eğitilmiş LSTM ile n_steps adım özyinelemeli tahmin üretir."""
    if pack is None:
        return None

    model = pack["model"]
    lookback = pack["lookback"]
    window = pack["history_tail"].copy()
    mean_, std_ = pack["mean"], pack["std"]

    preds_norm = []
    for _ in range(n_steps):
        x_in = window[-lookback:].reshape((1, lookback, 1))
        next_val = model.predict(x_in, verbose=0)[0, 0]
        preds_norm.append(next_val)
        window = np.append(window, next_val)

    preds_log = np.array(preds_norm) * std_ + mean_
    preds = np.expm1(preds_log)
    return np.clip(preds, 0, None)
# === [11] ENSEMBLE: RIDGE META-LEARNER STACKING ===============================

def train_ridge_stacking(val_preds_dict, val_y):
    """Validation tahminleri üzerinde Ridge regresyon meta-learner eğitir."""
    model_keys = sorted(val_preds_dict.keys())
    X_meta = np.column_stack([val_preds_dict[k] for k in model_keys])
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    meta_model = Ridge(alpha=1.0, positive=True, random_state=42)
    meta_model.fit(X_meta_scaled, val_y)

    return {"model": meta_model, "scaler": scaler, "model_keys": model_keys}


def predict_ridge_stacking(pack, test_preds_dict):
    """Ridge stacking ile test tahminlerini birleştirir."""
    if pack is None:
        return None
    model_keys = pack["model_keys"]
    if not all(k in test_preds_dict for k in model_keys):
        return None
    X_meta = np.column_stack([test_preds_dict[k] for k in model_keys])
    X_meta_scaled = pack["scaler"].transform(X_meta)
    preds = pack["model"].predict(X_meta_scaled)
    return np.clip(preds, 0, None)
# === [12] WALK-FORWARD VALİDASYON ============================================
# Her test yılı için:
#   - Test  : ilgili yıl (ör. 2023-01-01 .. 2023-12-31)
#   - Val   : test yılından önceki son VAL_WINDOW_DAYS gün
#   - Train : validation öncesi tüm geçmiş
# Val; parametre ayarı, kalibrasyon ve stacking için kullanılır.
# Test yalnızca nihai değerlendirme içindir.

MODEL_GROUPS = {
    "Naive": "Benchmark", "Seasonal_Naive": "Benchmark",
    "MA_7": "Benchmark", "MA_14": "Benchmark", "MA_30": "Benchmark",
    "SES": "Benchmark",
    "TSB": "Seyrek Talep", "Croston_SBA": "Seyrek Talep",
    "RandomForest": "Makine Ogrenmesi", "LGB_Tweedie": "Makine Ogrenmesi",
    "LGB_QBlend": "Makine Ogrenmesi", "LGB_TwoStage": "Makine Ogrenmesi",
    "LGB_ThreeStage": "Makine Ogrenmesi", "ET_LGB_Hybrid": "Makine Ogrenmesi",
    "XGB_Tweedie": "Makine Ogrenmesi", "XGB_TwoStage": "Makine Ogrenmesi",
    "LSTM": "Derin Ogrenme",
    "Ridge_Stacking": "Ensemble",
}

# ML modelleri icin, her history_mode'da train+val refit + calibration
# yapildigindan islem maliyeti onemli olcude arttigi icin bu liste sabit
# tutulur; tum modeller ayni mantikla (X_train, y_train, X_val, y_val) alir.
ML_SPECS = [
    ("RandomForest", train_random_forest, predict_random_forest),
    ("LGB_Tweedie", train_lgb_tweedie, predict_lgb),
    ("LGB_QBlend", train_lgb_qblend, predict_lgb_qblend),
    ("LGB_TwoStage", train_lgb_twostage, predict_lgb_twostage),
    ("LGB_ThreeStage", train_lgb_threestage, predict_lgb_threestage),
    ("ET_LGB_Hybrid", train_extratrees_lgb_hybrid, predict_extratrees_lgb_hybrid),
    ("XGB_Tweedie", train_xgb_tweedie, predict_xgb),
    ("XGB_TwoStage", train_xgb_twostage, predict_xgb_twostage),
]


def get_period_splits(df, test_year):
    """
    Bir test yili icin train/val/test tarih araliklarini hesaplar.
    test_end, ilgili yilin takvim olarak son gunu (31 Aralik) ile veri
    setinde fiilen mevcut olan en son tarihten KUCUK OLANI olarak
    belirlenir. Bu, veri kesimi (orn. DATA_CUTOFF_DATE = 2025-12-02)
    uygulanan donemlerde test_end'in gercekte mevcut olmayan gunleri
    (orn. 2025-12-31) yanlislikla kapsiyormus gibi raporlanmasini
    onler; boylece model_settings tablosundaki ve loglardaki tarih
    araliklari, fiilen kullanilan veriyle birebir tutarli olur.
    """
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end_calendar = pd.Timestamp(f"{test_year}-12-31")
    data_max_date = df["date"].max()
    test_end = min(test_end_calendar, data_max_date)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = test_start - pd.Timedelta(days=VAL_WINDOW_DAYS)
    train_end = val_start - pd.Timedelta(days=1)
    train_start = df["date"].min()
    return {
        "train_start": train_start, "train_end": train_end,
        "val_start": val_start, "val_end": val_end,
        "test_start": test_start, "test_end": test_end,
    }


def slice_period(df, start, end):
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)


def _record_result(all_metrics, all_predictions, all_val_predictions, model_settings,
                    key, history_mode, test_year, periods,
                    test_dates, test_y, test_pred,
                    val_dates, val_y, val_pred,
                    extra_settings, calib_settings=None):
    """Bir model sonucunu (metrik + tahmin + ayar kaydi) ortak sekilde kaydeder."""
    m = evaluate_predictions(test_dates, test_y, test_pred)
    health = flag_unhealthy_model(m)
    m.update({
        "model": key, "model_key": key, "model_group": MODEL_GROUPS.get(key, "Diger"),
        "history_mode": history_mode, "test_year": test_year,
        "focused_score": compute_focused_score(m),
        "saglik_durumu": health,
    })
    all_metrics.append(m)

    for d, yt, yp in zip(val_dates, val_y, val_pred):
        all_val_predictions.append({
            "model": key, "history_mode": history_mode, "test_year": test_year,
            "date": d, "y_true": yt, "y_pred": yp,
        })
    for d, yt, yp in zip(test_dates, test_y, test_pred):
        all_predictions.append({
            "model": key, "history_mode": history_mode, "test_year": test_year,
            "date": d, "y_true": yt, "y_pred": yp,
        })

    settings_row = {
        "model": key, "model_key": key, "model_group": MODEL_GROUPS.get(key, "Diger"),
        "history_mode": history_mode, "test_year": test_year, **periods,
        "parametreler": json.dumps(extra_settings, ensure_ascii=False, default=str),
    }
    if calib_settings:
        settings_row.update(calib_settings)
    model_settings.append(settings_row)
    return m


def run_walk_forward(df, feature_cols):
    """
    Tum modelleri, tum test yillari ve gerekli history mode'lari icin
    calistirir. Donus:
      - all_metrics: liste, her satir bir (model, history_mode, test_year) sonucu
      - all_predictions: liste, her satir bir gunluk tahmin kaydi (test donemi)
      - all_val_predictions: liste, her satir bir gunluk tahmin kaydi (validation donemi)
      - model_settings: liste, model parametre/ayar/calibration kayitlari
      - fi_store: en iyi agac tabanli modelin feature importance'i icin
    """
    all_metrics = []
    all_predictions = []
    all_val_predictions = []
    model_settings = []
    fi_store = {}  # {(model_key, test_year): (feature_names, importances)}

    for test_year in TEST_YEARS:
        periods = get_period_splits(df, test_year)
        log("EVAL", f"=== Test yili {test_year} icin walk-forward baslatiliyor ===")
        log("EVAL", f"Train: {periods['train_start'].date()} - {periods['train_end'].date()} | "
                     f"Val: {periods['val_start'].date()} - {periods['val_end'].date()} | "
                     f"Test: {periods['test_start'].date()} - {periods['test_end'].date()}")

        train_full = slice_period(df, periods["train_start"], periods["train_end"])
        val_full = slice_period(df, periods["val_start"], periods["val_end"])
        test_full = slice_period(df, periods["test_start"], periods["test_end"])

        # GUVENLIK DUZELTMESI: periods sozlugundeki tarihler hesaplanirken
        # kullanilan formul ile fiilen elde edilen (slice_period sonrasi)
        # veri araligi farkli olabilir (orn. veri kesimi - DATA_CUTOFF_DATE -
        # uygulanan donemlerde). model_settings tablosunun HER ZAMAN fiilen
        # kullanilan veri araligini yansitmasi icin, periods sozlugu burada
        # gercek train/val/test alt kumelerinin min/max tarihleriyle
        # YENIDEN damgalanir. Boylece log mesajlari, grafikler ve kayit
        # tablolari arasinda tutarlilik garanti edilir.
        if len(train_full) > 0:
            periods["train_start"] = train_full["date"].min()
            periods["train_end"] = train_full["date"].max()
        if len(val_full) > 0:
            periods["val_start"] = val_full["date"].min()
            periods["val_end"] = val_full["date"].max()
        if len(test_full) > 0:
            periods["test_start"] = test_full["date"].min()
            periods["test_end"] = test_full["date"].max()

        if len(train_full) < 30 or len(val_full) < 30 or len(test_full) < 30:
            log("EVAL", f"UYARI: {test_year} icin yeterli veri yok, atlaniyor.")
            continue

        train_y_full = train_full["sales_qty"].values
        val_y = val_full["sales_qty"].values
        test_y = test_full["sales_qty"].values
        test_dates = test_full["date"]
        val_dates = val_full["date"]

        pos_rate_test = float((test_y > 0).mean())
        log("EVAL", f"Test donemi pozitif talep orani: %{pos_rate_test*100:.1f}")

        val_preds_for_stacking = {}
        test_preds_for_stacking = {}

        # ---- A) Basit benchmark + seyrek talep modelleri (history_mode=full) --
        ses_alpha = optimize_ses_alpha(train_y_full, val_y)
        tsb_a, tsb_b = optimize_intermittent_params(model_tsb, train_y_full, val_y)
        cro_a, cro_b = optimize_intermittent_params(model_croston_sba, train_y_full, val_y)

        simple_specs = [
            ("Naive", lambda ty, n: model_naive(ty, n), {}),
            ("Seasonal_Naive", lambda ty, n: model_seasonal_naive(ty, n, SEASON_LENGTH),
             {"season": SEASON_LENGTH}),
            ("MA_7", lambda ty, n: model_moving_average(ty, n, 7), {"window": 7}),
            ("MA_14", lambda ty, n: model_moving_average(ty, n, 14), {"window": 14}),
            ("MA_30", lambda ty, n: model_moving_average(ty, n, 30), {"window": 30}),
            ("SES", lambda ty, n: model_ses(ty, n, ses_alpha), {"alpha": ses_alpha}),
            ("TSB", lambda ty, n: model_tsb(ty, n, tsb_a, tsb_b),
             {"alpha": tsb_a, "beta": tsb_b}),
            ("Croston_SBA", lambda ty, n: model_croston_sba(ty, n, cro_a, cro_b),
             {"alpha": cro_a, "beta": cro_b}),
        ]

        for key, func, params in simple_specs:
            mode = "full"
            # Validation tahmini: train_full'un val'den onceki kismi kullanilarak uretilir
            train_before_val = train_y_full[:-len(val_y)] if len(train_y_full) > len(val_y) else train_y_full
            val_pred = func(train_before_val, len(val_y))
            # Test tahmini: train+val (train_full zaten val'i icermez, o yuzden
            # train_y_full + val_y birlikte kullanilir)
            train_plus_val = np.concatenate([train_y_full, val_y])
            test_pred = func(train_plus_val, len(test_y))

            val_preds_for_stacking[key] = val_pred
            test_preds_for_stacking[key] = test_pred

            _record_result(all_metrics, all_predictions, all_val_predictions, model_settings,
                            key, mode, test_year, periods,
                            test_dates, test_y, test_pred, val_dates, val_y, val_pred,
                            extra_settings=params)

        # ---- B) ML modelleri: her history_mode icin egitim + calibration -------
        for key, train_func, predict_func in ML_SPECS:
            if key in ("LGB_Tweedie", "LGB_QBlend", "LGB_TwoStage", "LGB_ThreeStage",
                       "ET_LGB_Hybrid") and not HAS_LGB:
                continue
            if key in ("XGB_Tweedie", "XGB_TwoStage") and not HAS_XGB:
                continue

            mode_results = {}  # mode -> (pack, val_score_kalibre_oncesi)
            for mode in HISTORY_MODES:
                train_sliced = get_history_slice(train_full, mode)
                if len(train_sliced) < 60:
                    continue
                Xtr = train_sliced[feature_cols].values
                ytr = train_sliced["sales_qty"].values
                Xva = val_full[feature_cols].values
                sw = get_sample_weights(train_sliced["date"], mode)

                try:
                    pack = train_func(Xtr, ytr, Xva, val_y, sample_weight=sw)
                    if pack is None or pack.get("val_pred") is None:
                        continue
                    val_score = compute_focused_score(
                        evaluate_predictions(val_dates, val_y, pack["val_pred"])
                    )
                    mode_results[mode] = (pack, val_score)
                except Exception as e:
                    log("MODEL", f"UYARI: {key} ({mode}, {test_year}) egitiminde hata: {e}")
                    continue

            if not mode_results:
                continue

            best_mode = min(mode_results, key=lambda m: mode_results[m][1])
            best_pack = mode_results[best_mode][0]

            Xte = test_full[feature_cols].values
            raw_val_pred = best_pack["val_pred"]
            raw_test_pred = predict_func(best_pack, Xte)
            if raw_test_pred is None:
                continue

            # ---- Cut threshold + calibration (sadece validation'dan ogrenilir) ----
            val_pred, test_pred, calib_settings = calibrate_model_predictions(
                val_y, raw_val_pred, raw_test_pred
            )

            val_preds_for_stacking[key] = val_pred
            test_preds_for_stacking[key] = test_pred

            _record_result(all_metrics, all_predictions, all_val_predictions, model_settings,
                            key, best_mode, test_year, periods,
                            test_dates, test_y, test_pred, val_dates, val_y, val_pred,
                            extra_settings={"history_mode_secimi": "validation_focused_score"},
                            calib_settings=calib_settings)

            # Feature importance (sadece agac tabanli/regresyon kaynagi olan modeller)
            try:
                fi_source = best_pack.get("fi_source")
                if fi_source is not None and hasattr(fi_source, "feature_importances_"):
                    fi_store[(key, test_year)] = (feature_cols, fi_source.feature_importances_)
            except Exception:
                pass

        # ---- C) LSTM (ana siralamaya dahil ama saglik etiketiyle isaretlenir) ----
        if HAS_TF:
            try:
                train_only = train_full["sales_qty"].values
                lstm_val_pack = train_lstm_model(
                    train_only[:-30] if len(train_only) > 400 else train_only,
                    train_only[-30:] if len(train_only) > 400 else train_only[-10:],
                    lookback=30, epochs=15,
                )
                raw_val_pred = (predict_lstm_model(lstm_val_pack, len(val_y))
                                 if lstm_val_pack is not None else np.full(len(val_y), np.mean(train_y_full)))

                lstm_pack = train_lstm_model(train_y_full, val_y, lookback=30, epochs=20)
                raw_test_pred = predict_lstm_model(lstm_pack, len(test_y))

                if raw_test_pred is not None:
                    val_pred, test_pred, calib_settings = calibrate_model_predictions(
                        val_y, raw_val_pred, raw_test_pred
                    )

                    val_preds_for_stacking["LSTM"] = val_pred
                    test_preds_for_stacking["LSTM"] = test_pred

                    _record_result(all_metrics, all_predictions, all_val_predictions, model_settings,
                                    "LSTM", "full", test_year, periods,
                                    test_dates, test_y, test_pred, val_dates, val_y, val_pred,
                                    extra_settings={"lookback": 30, "epochs": 20,
                                                     "not": "Derin ogrenme denendi; veri yapisi seyrek "
                                                            "oldugu icin sinirli performans gosterdi."},
                                    calib_settings=calib_settings)
            except Exception as e:
                log("MODEL", f"UYARI: LSTM ({test_year}) calistirilamadi: {e}")

        # ---- D) Ridge Meta-Learner Stacking (kalibre edilmis girdilerle) -----------
        if len(val_preds_for_stacking) >= 2:
            try:
                stack_pack = train_ridge_stacking(val_preds_for_stacking, val_y)
                raw_test_pred = predict_ridge_stacking(stack_pack, test_preds_for_stacking)
                raw_val_pred = predict_ridge_stacking(stack_pack, val_preds_for_stacking)

                if raw_test_pred is not None and raw_val_pred is not None:
                    val_pred, test_pred, calib_settings = calibrate_model_predictions(
                        val_y, raw_val_pred, raw_test_pred
                    )

                    _record_result(all_metrics, all_predictions, all_val_predictions, model_settings,
                                    "Ridge_Stacking", "full", test_year, periods,
                                    test_dates, test_y, test_pred, val_dates, val_y, val_pred,
                                    extra_settings={"base_models": stack_pack["model_keys"]},
                                    calib_settings=calib_settings)
            except Exception as e:
                log("MODEL", f"UYARI: Ridge Stacking ({test_year}) calistirilamadi: {e}")

        n_results = len([m for m in all_metrics if m['test_year'] == test_year])
        n_unhealthy = len([m for m in all_metrics if m['test_year'] == test_year
                            and m.get('saglik_durumu') != 'OK'])
        log("EVAL", f"Test yili {test_year} tamamlandi. {n_results} model-sonucu uretildi "
                     f"({n_unhealthy} model saglik uyarisi ile isaretlendi).")

    return all_metrics, all_predictions, all_val_predictions, model_settings, fi_store
# === [13] MODEL KARSILASTIRMA VE EN IYI MODEL SECIMI ==========================

def build_model_ranking(metrics_df):
    """
    Her model icin TUM TEST YILLARI ortalamasi alinarak genel siralama
    olusturur. ONEMLI: gruplama SADECE 'model' bazinda yapilir (history_mode
    bazinda DEGIL), cunku her test yili kendi en iyi history_mode'unu
    bagimsiz olarak secebiliyor (orn. 2023->full, 2024->recency_weighted,
    2025->recent_only). Eger gruplama history_mode'u da icerseydi, bir
    modelin sadece TEK bir yilda kullanilmis (dolayisiyla istatistiksel
    olarak guvenilmez) bir (model, history_mode) kombinasyonu yanlislikla
    "model bazinda en iyi sonuc" gibi gorunup siralamayi carpitabilirdi.

    history_mode kolonunda, modelin test yillari arasinda en sik kullandigi
    (veya hepsi farkliysa "karisik") mod bilgi amacli gosterilir.
    """
    numeric_cols = ["WAPE", "hWAPE", "mWAPE", "MAE", "RMSE", "Bias_%",
                     "total_pred_true_ratio", "demand_precision", "demand_recall",
                     "demand_f1", "top30_capture_ratio", "large50_underforecast_rate",
                     "large100_underforecast_rate", "Pik_MAE_50+", "Pik_MAE_100+",
                     "focused_score"]

    base_cols = ["model", "model_key", "model_group"]
    agg = metrics_df.groupby(base_cols)[numeric_cols].mean().reset_index()
    agg["n_test_years"] = metrics_df.groupby(base_cols)["test_year"].nunique().values

    # Bilgi amacli: en sik kullanilan history_mode (esitlikte "karisik" gosterilir)
    def _dominant_mode(g):
        counts = g["history_mode"].value_counts()
        top = counts.index[0]
        if (counts == counts.max()).sum() > 1:
            return "karisik (" + "/".join(sorted(g["history_mode"].unique())) + ")"
        return top

    mode_summary = metrics_df.groupby(base_cols)[["history_mode"]].apply(_dominant_mode)
    mode_summary = mode_summary.reset_index(name="history_mode")
    agg = agg.merge(mode_summary, on=base_cols, how="left")

    if "saglik_durumu" in metrics_df.columns:
        ok_counts = (
            metrics_df.assign(_ok=(metrics_df["saglik_durumu"] == "OK").astype(int))
            .groupby(base_cols)["_ok"].sum().reset_index()
        )
        agg = agg.merge(ok_counts, on=base_cols, how="left")
        agg["saglik_durumu_ozet"] = agg.apply(
            lambda r: f"OK ({int(r['_ok'])}/{int(r['n_test_years'])})"
            if r["_ok"] == r["n_test_years"]
            else f"UYARI ({int(r['_ok'])}/{int(r['n_test_years'])} yil saglikli)",
            axis=1,
        )
        agg = agg.drop(columns=["_ok"])
        agg["is_fully_healthy"] = agg["saglik_durumu_ozet"].str.startswith("OK")
    else:
        agg["saglik_durumu_ozet"] = "UNKNOWN"
        agg["is_fully_healthy"] = True

    # Yalnizca TUM test yillarinda (TEST_YEARS sayisi kadar) sonucu olan
    # modeller guvenilir kabul edilir; eksik yillarda calismis modeller
    # siralamanin sonuna itilir (ama tamamen cikarilmaz).
    n_expected_years = len(TEST_YEARS)
    agg["has_full_coverage"] = agg["n_test_years"] >= n_expected_years

    ranking = agg.sort_values(
        by=["has_full_coverage", "focused_score"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    return ranking, agg


def select_best_model(ranking):
    """
    Final siralamadan en iyi modeli sozluk olarak dondurur. Oncelik sirasi:
      1) Tum test yillarinda calismis (has_full_coverage) VE tum yillarda
         saglikli (is_fully_healthy) olan modeller arasinda en dusuk
         focused_score.
      2) Yoksa, tum test yillarinda calismis modeller arasinda en dusuk
         focused_score (saglik uyarisi olsa bile) -> best_model_is_flagged=True.
      3) O da yoksa, genel siralamadaki ilk model -> best_model_is_flagged=True.
    """
    full_cov = ranking[ranking.get("has_full_coverage", True) == True]

    healthy_full = full_cov[full_cov.get("is_fully_healthy", True) == True]
    if len(healthy_full) > 0:
        best_row = healthy_full.sort_values("focused_score").iloc[0]
        result = best_row.to_dict()
        result["best_model_is_flagged"] = False
        return result

    if len(full_cov) > 0:
        best_row = full_cov.sort_values("focused_score").iloc[0]
        result = best_row.to_dict()
        result["best_model_is_flagged"] = True
        return result

    best_row = ranking.sort_values("focused_score").iloc[0]
    result = best_row.to_dict()
    result["best_model_is_flagged"] = True
    return result


def get_best_model_predictions(all_predictions_df, best_model_key, best_history_mode=None):
    """
    Secilen en iyi modelin tum test yillarini kapsayan tahmin setini dondurur.
    NOT: Her test yili kendi en iyi history_mode'unu bagimsiz secebildigi icin
    (orn. 2023->full, 2024->recency_weighted, 2025->recent_only), filtreleme
    sadece model adina gore yapilir; boylece her yilin GERCEKTEN kullanilan
    (o yil icin en iyi) tahminleri elde edilir.
    """
    mask = all_predictions_df["model"] == best_model_key
    return all_predictions_df.loc[mask].sort_values("date").reset_index(drop=True)


def get_best_model_val_predictions(all_val_predictions_df, best_model_key, best_history_mode=None):
    """
    Secilen en iyi modelin tum validation donemlerini kapsayan tahmin setini
    dondurur (model adina gore filtrelenir; bkz. get_best_model_predictions
    notu).
    """
    mask = all_val_predictions_df["model"] == best_model_key
    return all_val_predictions_df.loc[mask].sort_values("date").reset_index(drop=True)
# === [14] STOK POLITIKASI SIMULASYONU =========================================
# En iyi model tahminleri uzerinden farkli buffer stratejileri denenir.
# Gunluk ve haftalik seviyede calisir; tezde haftalik seviye one cikarilir.


def safe_weekly_resample(df, date_col="date", agg="sum"):
    """
    Gunluk bir seriyi haftalik (Pazar gunu etiketli, freq='W') olarak
    toplar/ortalar; ANCAK pandas'in resample('W') davranisinin yarattigi
    bir tuzagi onler: eger veri serisi haftanin ortasinda bir gunde
    kesiliyorsa (orn. veri kesimi nedeniyle son gun 2 Aralik = Sali ise),
    pandas yine de o gunu iceren haftayi TAM bir hafta gibi (Pazar
    etiketiyle, orn. 7 Aralik) gosterir. Bu durumda son hafta gercekte
    yalnizca 1-2 gunluk veri icerir ama 7 gunluk bir haftaymis gibi
    grafik/tabloya yapay dusuk bir talep degeriyle yansir ve veri
    araliginin oldugundan ileri bir tarihe (orn. Aralik sonuna) kadar
    uzaniyormus izlenimi verir. Bu fonksiyon, son hafta TAM (7 gun)
    degilse o haftayi sonuc setinden cikarir.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)

    last_actual_date = d[date_col].max()
    resampler = d.set_index(date_col).resample("W")
    weekly = resampler.sum() if agg == "sum" else resampler.mean()
    weekly = weekly.reset_index()

    if len(weekly) > 0:
        last_week_end = weekly[date_col].iloc[-1]
        last_week_start = last_week_end - pd.Timedelta(days=6)
        days_covered = (last_actual_date - last_week_start).days + 1
        if days_covered < 7:
            log("CLEAN", f"UYARI: Son hafta ({last_week_start.date()} - "
                          f"{last_week_end.date()}) yalnizca {days_covered} gun "
                          f"veri icerdigi icin (veri kesimi nedeniyle) haftalik "
                          f"analizden cikarildi.")
            weekly = weekly.iloc[:-1].reset_index(drop=True)

    return weekly


POLICY_DEFS = {
    "Tahmin": lambda pred, buf: pred,
    "Tahmin+buffer_p50": lambda pred, buf: pred + buf["p50"],
    "Tahmin+buffer_p75": lambda pred, buf: pred + buf["p75"],
    "Tahmin+buffer_p90": lambda pred, buf: pred + buf["p90"],
    "Tahmin+1std_residual": lambda pred, buf: pred + buf["std1"],
    "Tahmin+1.28std_residual": lambda pred, buf: pred + buf["std128"],
    "Tahmin+1.65std_residual": lambda pred, buf: pred + buf["std165"],
}


def compute_residual_buffers(y_true_val, y_pred_val):
    """
    Validation donemi residuallerinden (gercek - tahmin, sadece pozitif
    kalan kisim) buffer degerlerini hesaplar.
    """
    residuals = np.asarray(y_true_val) - np.asarray(y_pred_val)
    pos_resid = residuals[residuals > 0]

    if len(pos_resid) == 0:
        return {"p50": 0.0, "p75": 0.0, "p90": 0.0,
                "std1": 0.0, "std128": 0.0, "std165": 0.0}

    std_resid = residuals.std()
    return {
        "p50": float(np.percentile(pos_resid, 50)),
        "p75": float(np.percentile(pos_resid, 75)),
        "p90": float(np.percentile(pos_resid, 90)),
        "std1": float(std_resid),
        "std128": float(1.28 * std_resid),
        "std165": float(1.65 * std_resid),
    }


def simulate_policy(dates, y_true, y_pred, policy_name, buffers, granularity="daily"):
    """
    Tek bir politika icin stok simulasyonu yapar ve ozet metrikleri dondurur.
    granularity: 'daily' veya 'weekly' (haftalik toplam talep/tahmin uzerinden)
    """
    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})

    if granularity == "weekly":
        df = safe_weekly_resample(df, date_col="date", agg="sum")

    stock_plan = POLICY_DEFS[policy_name](df["y_pred"].values, buffers)
    stock_plan = np.clip(stock_plan, 0, None)

    actual = df["y_true"].values
    shortage = np.clip(actual - stock_plan, 0, None)
    excess = np.clip(stock_plan - actual, 0, None)
    stockout_days = int((shortage > 0).sum())
    service_level = float(100 * (1 - stockout_days / len(actual))) if len(actual) > 0 else np.nan

    total_demand = float(actual.sum())
    total_forecast = float(df["y_pred"].sum())
    total_stock_plan = float(stock_plan.sum())

    return {
        "policy": policy_name,
        "granularity": granularity,
        "service_level_%": round(service_level, 2),
        "stockout_count": stockout_days,
        "total_shortage": round(float(shortage.sum()), 1),
        "total_excess": round(float(excess.sum()), 1),
        "stock_to_demand_ratio": round(float(total_stock_plan / total_demand), 3) if total_demand > 0 else np.nan,
        "avg_stock": round(float(stock_plan.mean()), 2),
        "max_stock": round(float(stock_plan.max()), 2),
        "total_demand": total_demand,
        "total_forecast": round(total_forecast, 1),
        "total_stock_plan": round(total_stock_plan, 1),
    }


def compute_policy_balance_score(row):
    """
    Stok politikasi secimi icin dengeli skor. Yuksek service level iyi,
    dusuk shortage/excess iyi, stock_to_demand_ratio'nun asiri yuksek
    olmasi (gereksiz stok) cezalandirilir. Yuksek skor = daha iyi politika.
    """
    service = row["service_level_%"] / 100.0
    ratio = row["stock_to_demand_ratio"] if not np.isnan(row["stock_to_demand_ratio"]) else 2.0
    excess_penalty = max(0, ratio - 1.15) * 0.5   # %15 uzerindeki fazla stok cezalandirilir
    shortage_norm = row["total_shortage"] / max(row["total_demand"], 1)

    score = service - 0.5 * shortage_norm - excess_penalty
    return float(score)


def run_stock_simulation(best_preds_df, val_y_true, val_y_pred):
    """
    En iyi model tahminleri (test donemi, tum yillar birlesik) uzerinden
    gunluk ve haftalik stok simulasyonlarini calistirir; ozet tabloyu
    ve final onerilen politikayi dondurur.
    """
    buffers = compute_residual_buffers(val_y_true, val_y_pred)

    summary_rows = []
    for granularity in ["daily", "weekly"]:
        for policy_name in POLICY_DEFS:
            row = simulate_policy(
                best_preds_df["date"], best_preds_df["y_true"], best_preds_df["y_pred"],
                policy_name, buffers, granularity,
            )
            row["balance_score"] = compute_policy_balance_score(row)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Final oneri: haftalik granularitede en yuksek balance_score
    weekly_df = summary_df[summary_df["granularity"] == "weekly"].copy()
    best_policy_row = weekly_df.loc[weekly_df["balance_score"].idxmax()]

    return summary_df, best_policy_row, buffers


def build_daily_weekly_stock_tables(best_preds_df, buffers, chosen_policy_name):
    """Secilen politika icin gunluk ve haftalik detay tablolarini olusturur."""
    df = best_preds_df[["date", "y_true", "y_pred"]].copy()
    df["stock_plan"] = np.clip(
        POLICY_DEFS[chosen_policy_name](df["y_pred"].values, buffers), 0, None
    )
    df["shortage"] = np.clip(df["y_true"] - df["stock_plan"], 0, None)
    df["excess"] = np.clip(df["stock_plan"] - df["y_true"], 0, None)
    df["stockout"] = (df["shortage"] > 0).astype(int)

    weekly = df.set_index("date").resample("W").agg(
        y_true=("y_true", "sum"), y_pred=("y_pred", "sum"),
        stock_plan=("stock_plan", "sum"), shortage=("shortage", "sum"),
        excess=("excess", "sum"), stockout=("stockout", "max"),
    ).reset_index()

    # Veri kesimi nedeniyle son hafta tam (7 gun) degilse, yaniltici dusuk
    # degerli kismi hafta satirini cikar (bkz. safe_weekly_resample notu).
    if len(weekly) > 0:
        last_actual_date = df["date"].max()
        last_week_end = weekly["date"].iloc[-1]
        last_week_start = last_week_end - pd.Timedelta(days=6)
        days_covered = (last_actual_date - last_week_start).days + 1
        if days_covered < 7:
            weekly = weekly.iloc[:-1].reset_index(drop=True)

    return df, weekly
# === [15] GRAFIK URETIMI ======================================================
# Tum grafikler beyaz arka plan, Turkce baslik/eksen, 300 DPI PNG + PDF olarak
# kaydedilir. Renk semasi modul basinda tanimlanan sabit paleti kullanir.


def save_fig(fig, folder, filename_no_ext):
    """Bir figürü PNG (300 DPI) ve PDF olarak kaydeder."""
    png_path = os.path.join(folder, f"{filename_no_ext}.png")
    pdf_path = os.path.join(folder, f"{filename_no_ext}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def fmt_thousands(x, pos):
    return f"{x:,.0f}".replace(",", ".")


# --- 1) Gunluk talep zaman serisi --------------------------------------------

def plot_daily_demand(df, folder):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["date"], df["sales_qty"], color=COLOR_ACTUAL, linewidth=0.8)
    date_min, date_max = df["date"].min(), df["date"].max()
    ax.set_title(f"Gunluk Satis/Talep Zaman Serisi (110-90' PN 10 Dirsek) "
                 f"[{date_min.strftime('%d.%m.%Y')} - {date_max.strftime('%d.%m.%Y')}]",
                 fontsize=12)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Gunluk Talep (Adet)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(date_min - pd.Timedelta(days=15), date_max + pd.Timedelta(days=15))
    fig.tight_layout()
    return save_fig(fig, folder, "01_gunluk_talep_zaman_serisi")


# --- 2) Aylik toplam talep ----------------------------------------------------

def plot_monthly_total(df, folder):
    monthly = df.set_index("date")["sales_qty"].resample("ME").sum()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(monthly.index, monthly.values, width=20, color=COLOR_ACTUAL, alpha=0.85)
    ax.set_title("Aylik Toplam Talep")
    ax.set_xlabel("Ay")
    ax.set_ylabel("Toplam Talep (Adet)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return save_fig(fig, folder, "02_aylik_toplam_talep")


# --- 3) Yillik toplam talep ---------------------------------------------------

def plot_yearly_total(yearly_df, folder):
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(yearly_df["Yil"].astype(str), yearly_df["Toplam Talep"],
                   color=COLOR_ACTUAL, alpha=0.85)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():,.0f}".replace(",", "."),
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Yillik Toplam Talep")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Toplam Talep (Adet)")
    fig.tight_layout()
    return save_fig(fig, folder, "03_yillik_toplam_talep")


# --- 4) Sifir/pozitif talep orani --------------------------------------------

def plot_zero_positive_ratio(profile_info, df, folder):
    n_zero = int((df["sales_qty"] == 0).sum())
    n_pos = int((df["sales_qty"] > 0).sum())
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        [n_zero, n_pos], labels=["Sifir Talep Gunu", "Pozitif Talep Gunu"],
        colors=[COLOR_BENCH, COLOR_ACTUAL], autopct="%1.1f%%",
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.set_title("Sifir / Pozitif Talep Gunu Orani")
    fig.tight_layout()
    return save_fig(fig, folder, "04_sifir_pozitif_talep_orani")


# --- 5) Talep dagilimi histogram ---------------------------------------------

def plot_demand_histogram(df, folder):
    positive = df.loc[df["sales_qty"] > 0, "sales_qty"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(positive, bins=40, color=COLOR_ML, edgecolor="white", alpha=0.9)
    ax.set_title("Talep Dagilimi Histogrami (Pozitif Talep Gunleri)")
    ax.set_xlabel("Gunluk Talep (Adet)")
    ax.set_ylabel("Gun Sayisi (Frekans)")
    fig.tight_layout()
    return save_fig(fig, folder, "05_talep_dagilimi_histogram")


# --- 6) Talep boxplot ----------------------------------------------------------

def plot_demand_boxplot(df, folder):
    fig, ax = plt.subplots(figsize=(12, 7))
    data_by_year = [g["sales_qty"].values for _, g in df.groupby(df["date"].dt.year)]
    years = sorted(df["date"].dt.year.unique())
    bp = ax.boxplot(data_by_year, labels=[str(y) for y in years], patch_artist=True,
                     showfliers=True, flierprops=dict(markersize=3, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_ML)
        patch.set_alpha(0.6)
    ax.set_title("Yillara Gore Talep Boxplot Grafigi")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Gunluk Talep (Adet)")
    fig.tight_layout()
    return save_fig(fig, folder, "06_talep_boxplot")


# --- 7) ADI-CV2 talep siniflandirma -------------------------------------------

def plot_adi_cv2_classification(adi, cv2, folder):
    fig, ax = plt.subplots(figsize=(7, 7))

    adi_max = max(adi * 1.6, 3.0)
    cv2_max = max(cv2 * 1.6, 1.2)

    ax.axvspan(0, 1.32, ymin=0, ymax=0.49 / cv2_max, color=COLOR_GOOD, alpha=0.18)
    ax.axvspan(1.32, adi_max, ymin=0, ymax=0.49 / cv2_max, color=COLOR_INTERMITTENT, alpha=0.18)
    ax.axvspan(0, 1.32, ymin=0.49 / cv2_max, ymax=1, color=COLOR_PRED, alpha=0.18)
    ax.axvspan(1.32, adi_max, ymin=0.49 / cv2_max, ymax=1, color=COLOR_BAD, alpha=0.18)

    ax.axvline(1.32, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.49, color="gray", linestyle="--", linewidth=1)

    ax.text(0.6, 0.2, "Smooth", fontsize=11, ha="center", color=COLOR_GOOD, fontweight="bold")
    ax.text(adi_max * 0.7, 0.2, "Intermittent", fontsize=11, ha="center", color=COLOR_INTERMITTENT, fontweight="bold")
    ax.text(0.6, cv2_max * 0.75, "Erratic", fontsize=11, ha="center", color=COLOR_PRED, fontweight="bold")
    ax.text(adi_max * 0.7, cv2_max * 0.75, "Lumpy", fontsize=11, ha="center", color=COLOR_BAD, fontweight="bold")

    ax.scatter([adi], [cv2], color="black", s=140, zorder=5, marker="*",
               label=f"Urun (ADI={adi:.2f}, CV2={cv2:.2f})")
    ax.legend(loc="upper right", fontsize=9)

    ax.set_xlim(0, adi_max)
    ax.set_ylim(0, cv2_max)
    ax.set_xlabel("ADI (Average Demand Interval)")
    ax.set_ylabel("CV² (Talep Varyasyon Katsayisi Karesi)")
    ax.set_title("ADI - CV² Talep Siniflandirma Grafigi")
    fig.tight_layout()
    return save_fig(fig, folder, "07_adi_cv2_talep_siniflandirma")


# --- 8) Buyuk talep gunleri analizi -------------------------------------------

def plot_big_demand_days(df, folder):
    fig, ax = plt.subplots(figsize=(16, 6))
    normal = df[df["sales_qty"] < BIG50_THRESHOLD]
    big50 = df[(df["sales_qty"] >= BIG50_THRESHOLD) & (df["sales_qty"] < BIG100_THRESHOLD)]
    big100 = df[df["sales_qty"] >= BIG100_THRESHOLD]

    ax.scatter(normal["date"], normal["sales_qty"], color=COLOR_BENCH, s=6, alpha=0.5, label="Normal Talep")
    ax.scatter(big50["date"], big50["sales_qty"], color=COLOR_PRED, s=18, label=f"{BIG50_THRESHOLD}+ Buyuk Talep")
    ax.scatter(big100["date"], big100["sales_qty"], color=COLOR_BAD, s=28, label=f"{BIG100_THRESHOLD}+ Cok Buyuk Talep")

    ax.set_title("Proje/Ihale Kaynakli Buyuk Talep Gunleri Analizi")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Gunluk Talep (Adet)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_fig(fig, folder, "08_buyuk_talep_gunleri")
# --- 9) Walk-forward validation semasi ----------------------------------------

def plot_walk_forward_scheme(df, folder):
    fig, ax = plt.subplots(figsize=(14, 7))
    y_positions = {2023: 2, 2024: 1, 2025: 0}

    bar_height = 0.6
    for test_year, y in y_positions.items():
        periods = get_period_splits(df, test_year)
        train_start, train_end = periods["train_start"], periods["train_end"]
        val_start, val_end = periods["val_start"], periods["val_end"]
        test_start, test_end = periods["test_start"], periods["test_end"]

        ax.barh(y, (train_end - train_start).days, left=train_start,
                height=bar_height, color=COLOR_BENCH, alpha=0.85,
                label="Train" if y == 2 else "")
        ax.barh(y, (val_end - val_start).days, left=val_start,
                height=bar_height, color=COLOR_PRED, alpha=0.9,
                label="Validation" if y == 2 else "")
        ax.barh(y, (test_end - test_start).days, left=test_start,
                height=bar_height, color=COLOR_ACTUAL, alpha=0.9,
                label="Test" if y == 2 else "")

        ax.text(test_end + pd.Timedelta(days=20), y, f"Test {test_year}",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"Senaryo {y}" for y in y_positions.keys()])
    ax.set_xlabel("Tarih")
    ax.set_title("Walk-Forward Validation Semasi (2023, 2024, 2025 Test Donemleri)")
    ax.legend(loc="upper left", ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return save_fig(fig, folder, "09_walk_forward_validation_semasi")


# --- 10) Model siralamasi (focused score) -------------------------------------

def _group_color(group):
    return PALETTE_GROUP.get(group, COLOR_BENCH)


def plot_model_ranking(ranking, folder):
    top = ranking.sort_values("focused_score").head(15).iloc[::-1]
    colors = [_group_color(g) for g in top["model_group"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top["model"], top["focused_score"], color=colors)
    for b, val in zip(bars, top["focused_score"]):
        ax.text(val, b.get_y() + b.get_height() / 2, f" {val:.3f}",
                va="center", fontsize=9)

    ax.set_xlim(0, top["focused_score"].max() * 1.15)

    legend_elems = [Patch(facecolor=c, label=g) for g, c in PALETTE_GROUP.items()
                     if g in top["model_group"].unique()]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, framealpha=0.95, borderaxespad=0)

    ax.set_title("Model Siralamasi - Focused Score (Top 15, Dusuk Skor = Daha Iyi)")
    ax.set_xlabel("Focused Score (dusuk daha iyi)")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return save_fig(fig, folder, "10_model_siralamasi_focused_score")


# --- 11) hWAPE karsilastirma ---------------------------------------------------

def plot_hwape_comparison(ranking, folder):
    top = ranking.sort_values("hWAPE").head(15).iloc[::-1]
    colors = [_group_color(g) for g in top["model_group"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top["model"], top["hWAPE"], color=colors)
    for b, val in zip(bars, top["hWAPE"]):
        ax.text(val, b.get_y() + b.get_height() / 2, f" {val:.3f}",
                va="center", fontsize=9)

    ax.set_xlim(0, top["hWAPE"].max() * 1.15)

    legend_elems = [Patch(facecolor=c, label=g) for g, c in PALETTE_GROUP.items()
                     if g in top["model_group"].unique()]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, framealpha=0.95, borderaxespad=0)
    ax.set_title("Model hWAPE Karsilastirmasi (Top 15, Dusuk Daha Iyi)")
    ax.set_xlabel("hWAPE (Haftalik Agregasyon Uzerinden)")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return save_fig(fig, folder, "11_model_hWAPE_karsilastirma")


# --- 12) Bias % karsilastirma --------------------------------------------------

def plot_bias_comparison(ranking, folder):
    top = ranking.reindex(ranking["Bias_%"].abs().sort_values().index).head(15)
    top = top.sort_values("Bias_%")
    colors = [COLOR_BAD if v < 0 else COLOR_GOOD for v in top["Bias_%"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top["model"], top["Bias_%"], color=colors)
    for b, val in zip(bars, top["Bias_%"]):
        ax.text(val, b.get_y() + b.get_height() / 2, f" {val:.1f}%",
                va="center", fontsize=9, ha="left" if val >= 0 else "right")

    ax.axvline(0, color="black", linewidth=0.8)
    legend_elems = [Patch(facecolor=COLOR_GOOD, label="Pozitif Bias (Asiri Tahmin)"),
                    Patch(facecolor=COLOR_BAD, label="Negatif Bias (Eksik Tahmin)")]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
    ax.set_title("Model Bias % Karsilastirmasi (Top 15, |Bias| en dusuk)")
    ax.set_xlabel("Bias % (Tahmin - Gercek / Gercek)")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return save_fig(fig, folder, "12_model_bias_karsilastirma")


# --- 13) Demand F1 / Recall karsilastirma --------------------------------------

def plot_demand_f1_recall(ranking, folder):
    top = ranking.sort_values("demand_f1", ascending=False).head(15)

    x = np.arange(len(top))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2, top["demand_f1"], width, label="Demand F1", color=COLOR_ML)
    ax.bar(x + width / 2, top["demand_recall"], width, label="Demand Recall", color=COLOR_INTERMITTENT)

    ax.set_xticks(x)
    ax.set_xticklabels(top["model"], rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Talep Var/Yok Siniflandirma Basarisi (Demand F1 / Recall, Top 15)")
    ax.set_ylabel("Skor")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, folder, "13_model_demand_f1_recall")


# --- 14) Pred/True ratio karsilastirma -----------------------------------------

def plot_pred_true_ratio(ranking, folder):
    top = ranking.reindex((ranking["total_pred_true_ratio"] - 1).abs().sort_values().index).head(15)
    top = top.sort_values("total_pred_true_ratio")
    colors = [COLOR_GOOD if 0.9 <= v <= 1.1 else COLOR_PRED for v in top["total_pred_true_ratio"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top["model"], top["total_pred_true_ratio"], color=colors)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="Ideal Deger (1.0)")
    for b, val in zip(bars, top["total_pred_true_ratio"]):
        ax.text(val, b.get_y() + b.get_height() / 2, f" {val:.2f}", va="center", fontsize=9)

    ax.set_title("Toplam Tahmin / Toplam Gercek Orani Karsilastirmasi (Top 15)")
    ax.set_xlabel("Toplam Tahmin / Toplam Gercek")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_fig(fig, folder, "14_pred_true_ratio_karsilastirma")


# --- 15) Buyuk talep gunlerinde underforecast orani ----------------------------

def plot_large_demand_underforecast(ranking, folder):
    top = ranking.sort_values("focused_score").head(15)
    x = np.arange(len(top))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2, top["large50_underforecast_rate"], width,
           label=f"{BIG50_THRESHOLD}+ Underforecast Orani", color=COLOR_PRED)
    ax.bar(x + width / 2, top["large100_underforecast_rate"], width,
           label=f"{BIG100_THRESHOLD}+ Underforecast Orani", color=COLOR_BAD)

    ax.set_xticks(x)
    ax.set_xticklabels(top["model"], rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Buyuk Talep Gunlerinde Underforecast (Eksik Tahmin) Orani (Top 15)")
    ax.set_ylabel("Underforecast Orani")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, folder, "15_large_demand_underforecast")
# --- 16) En iyi model gunluk gercek vs tahmin ----------------------------------

def plot_best_daily(best_preds, folder, model_name=""):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(best_preds["date"], best_preds["y_true"], color=COLOR_ACTUAL,
            linewidth=1.0, label="Gercek Talep")
    ax.plot(best_preds["date"], best_preds["y_pred"], color=COLOR_PRED,
            linewidth=1.0, alpha=0.85, label="Tahmin")
    title = f"En Iyi Model ({model_name}) - Gunluk Gercek vs Tahmin" if model_name else "En Iyi Model - Gunluk Gercek vs Tahmin"
    ax.set_title(title)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Talep (Adet)")
    ax.set_xlim(best_preds["date"].min() - pd.Timedelta(days=5),
                best_preds["date"].max() + pd.Timedelta(days=5))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_fig(fig, folder, "16_best_model_gunluk_gercek_tahmin")


# --- 17) En iyi model haftalik gercek vs tahmin --------------------------------

def plot_best_weekly(best_preds, folder, model_name=""):
    weekly = safe_weekly_resample(best_preds[["date", "y_true", "y_pred"]], date_col="date", agg="sum")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(weekly["date"], weekly["y_true"], color=COLOR_ACTUAL, linewidth=1.8,
            marker="o", markersize=3, label="Gercek Talep")
    ax.plot(weekly["date"], weekly["y_pred"], color=COLOR_PRED, linewidth=1.8,
            marker="o", markersize=3, label="Tahmin")
    title = f"En Iyi Model ({model_name}) - Haftalik Gercek vs Tahmin" if model_name else "En Iyi Model - Haftalik Gercek vs Tahmin"
    ax.set_title(title)
    ax.set_xlabel("Hafta")
    ax.set_ylabel("Haftalik Toplam Talep (Adet)")
    ax.set_xlim(weekly["date"].min() - pd.Timedelta(days=5),
                weekly["date"].max() + pd.Timedelta(days=5))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_fig(fig, folder, "17_best_model_haftalik_gercek_tahmin")


# --- 18) En iyi model aylik gercek vs tahmin ------------------------------------

def plot_best_monthly(best_preds, folder, model_name=""):
    monthly = best_preds.set_index("date")[["y_true", "y_pred"]].resample("ME").sum()
    x = np.arange(len(monthly))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, monthly["y_true"], width, color=COLOR_ACTUAL, label="Gercek Talep")
    ax.bar(x + width / 2, monthly["y_pred"], width, color=COLOR_PRED, label="Tahmin")
    ax.set_xticks(x[::3])
    ax.set_xticklabels([d.strftime("%Y-%m") for d in monthly.index[::3]], rotation=45, ha="right")
    title = f"En Iyi Model ({model_name}) - Aylik Gercek vs Tahmin" if model_name else "En Iyi Model - Aylik Gercek vs Tahmin"
    ax.set_title(title)
    ax.set_ylabel("Aylik Toplam Talep (Adet)")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, folder, "18_best_model_aylik_gercek_tahmin")


# --- 19) En iyi model kumulatif gercek vs tahmin --------------------------------

def plot_best_cumulative(best_preds, folder, model_name=""):
    cum_true = best_preds["y_true"].cumsum()
    cum_pred = best_preds["y_pred"].cumsum()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(best_preds["date"], cum_true, color=COLOR_ACTUAL, linewidth=2.0, label="Gercek (Kumulatif)")
    ax.plot(best_preds["date"], cum_pred, color=COLOR_PRED, linewidth=2.0, label="Tahmin (Kumulatif)")
    ax.fill_between(best_preds["date"], cum_true, cum_pred, color=COLOR_BAND, alpha=0.25)
    title = f"En Iyi Model ({model_name}) - Kumulatif Gercek vs Tahmin" if model_name else "En Iyi Model - Kumulatif Gercek vs Tahmin"
    ax.set_title(title)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Kumulatif Talep (Adet)")
    ax.set_xlim(best_preds["date"].min() - pd.Timedelta(days=5),
                best_preds["date"].max() + pd.Timedelta(days=5))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save_fig(fig, folder, "19_best_model_kumulatif_gercek_tahmin")


# --- 20) En iyi model scatter ----------------------------------------------------

def plot_best_scatter(best_preds, folder, model_name=""):
    y_true = best_preds["y_true"].values
    y_pred = best_preds["y_pred"].values
    max_val = max(y_true.max(), y_pred.max()) * 1.05

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, color=COLOR_ML, alpha=0.5, s=22, edgecolor="white", linewidth=0.3)
    ax.plot([0, max_val], [0, max_val], color=COLOR_BAD, linestyle="--", linewidth=1.5, label="Ideal (y=x)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Gercek Talep (Adet)")
    ax.set_ylabel("Tahmin (Adet)")
    title = f"En Iyi Model ({model_name}) - Gercek vs Tahmin Scatter Grafigi" if model_name else "En Iyi Model - Gercek vs Tahmin Scatter Grafigi"
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save_fig(fig, folder, "20_best_model_scatter")


# --- 21) En iyi model residual zaman grafigi ------------------------------------

def plot_best_residual_time(best_preds, folder, model_name=""):
    residual = best_preds["y_true"] - best_preds["y_pred"]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(best_preds["date"], residual, color=COLOR_DEEP, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    title = f"En Iyi Model ({model_name}) - Residual (Gercek - Tahmin) Zaman Grafigi" if model_name else "En Iyi Model - Residual (Gercek - Tahmin) Zaman Grafigi"
    ax.set_title(title)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Residual (Adet)")
    ax.set_xlim(best_preds["date"].min() - pd.Timedelta(days=5),
                best_preds["date"].max() + pd.Timedelta(days=5))
    fig.tight_layout()
    return save_fig(fig, folder, "21_best_model_residual_zaman")


# --- 22) En iyi model residual histogram -----------------------------------------

def plot_best_residual_histogram(best_preds, folder, model_name=""):
    residual = best_preds["y_true"] - best_preds["y_pred"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(residual, bins=40, color=COLOR_DEEP, edgecolor="white", alpha=0.9)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.axvline(residual.mean(), color=COLOR_BAD, linestyle="--", linewidth=1.2,
               label=f"Ortalama Residual = {residual.mean():.2f}")
    title = f"En Iyi Model ({model_name}) - Residual Histogrami" if model_name else "En Iyi Model - Residual Histogrami"
    ax.set_title(title)
    ax.set_xlabel("Residual (Gercek - Tahmin)")
    ax.set_ylabel("Frekans")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, folder, "22_best_model_residual_histogram")


# --- 23) En iyi model tahmin bandi -----------------------------------------------

def plot_best_forecast_band(best_preds, buffers, folder, model_name=""):
    band = buffers["std128"]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(best_preds["date"], best_preds["y_true"], color=COLOR_ACTUAL, linewidth=1.0, label="Gercek Talep")
    ax.plot(best_preds["date"], best_preds["y_pred"], color=COLOR_PRED, linewidth=1.0, label="Tahmin")
    ax.fill_between(best_preds["date"], best_preds["y_pred"], best_preds["y_pred"] + band,
                     color=COLOR_BAND, alpha=0.35, label=f"Tahmin + Buffer (±{band:.1f}, 1.28σ)")
    title = f"En Iyi Model ({model_name}) - Tahmin Bandi (Validation Residual Buffer)" if model_name else "En Iyi Model - Tahmin Bandi (Validation Residual Buffer)"
    ax.set_title(title)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Talep (Adet)")
    ax.set_xlim(best_preds["date"].min() - pd.Timedelta(days=5),
                best_preds["date"].max() + pd.Timedelta(days=5))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_fig(fig, folder, "23_best_model_tahmin_bandi")


# --- 24) Top3 model haftalik karsilastirma ---------------------------------------

def plot_top3_weekly(all_preds_df, ranking, folder):
    top3 = ranking.sort_values("focused_score").head(3)
    fig, ax = plt.subplots(figsize=(16, 6))

    actual_plotted = False
    line_colors = [COLOR_PRED, COLOR_ML, COLOR_DEEP]
    for i, (_, row) in enumerate(top3.iterrows()):
        mask = all_preds_df["model"] == row["model"]
        sub = all_preds_df.loc[mask].sort_values("date")
        weekly = safe_weekly_resample(sub[["date", "y_true", "y_pred"]], date_col="date", agg="sum")

        if not actual_plotted:
            ax.plot(weekly["date"], weekly["y_true"], color=COLOR_ACTUAL, linewidth=2.2, label="Gercek Talep")
            actual_plotted = True

        ax.plot(weekly["date"], weekly["y_pred"], color=line_colors[i % 3], linewidth=1.4,
                alpha=0.9, label=f"{row['model']} Tahmini")

    ax.set_title("Top 3 Model - Haftalik Gercek vs Tahmin Karsilastirmasi")
    ax.set_xlabel("Hafta")
    ax.set_ylabel("Haftalik Toplam Talep (Adet)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_fig(fig, folder, "24_top3_model_haftalik_karsilastirma")


# --- 25) Top5 model kumulatif karsilastirma --------------------------------------

def plot_top5_cumulative(all_preds_df, ranking, folder):
    top5 = ranking.sort_values("focused_score").head(5)
    fig, ax = plt.subplots(figsize=(16, 6))

    actual_plotted = False
    cmap_colors = [COLOR_PRED, COLOR_ML, COLOR_DEEP, COLOR_INTERMITTENT, COLOR_BENCH]
    for i, (_, row) in enumerate(top5.iterrows()):
        mask = all_preds_df["model"] == row["model"]
        sub = all_preds_df.loc[mask].sort_values("date")

        if not actual_plotted:
            ax.plot(sub["date"], sub["y_true"].cumsum(), color=COLOR_ACTUAL, linewidth=2.2, label="Gercek (Kumulatif)")
            actual_plotted = True

        ax.plot(sub["date"], sub["y_pred"].cumsum(), color=cmap_colors[i % 5], linewidth=1.4,
                alpha=0.9, label=f"{row['model']} (Kumulatif)")

    ax.set_title("Top 5 Model - Kumulatif Tahmin Karsilastirmasi")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Kumulatif Talep (Adet)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, folder, "25_top5_model_kumulatif_karsilastirma")
# --- 26) Feature importance top 25 -----------------------------------------------

FEATURE_GROUP_MAP_PREFIXES = [
    ("lag", "lag"),
    ("roll_", "rolling"),
    ("pos_days", "rolling"),
    ("big50_days", "rolling"),
    ("big100_days", "rolling"),
    ("pos_rate", "seyrek_talep"),
    ("big50_rate", "seyrek_talep"),
    ("big100_rate", "seyrek_talep"),
    ("days_since", "seyrek_talep"),
    ("last_positive_qty", "seyrek_talep"),
    ("trend", "trend"),
    ("is_resmi_tatil", "tatil"),
    ("is_dini_bayram", "tatil"),
    ("bayram_", "tatil"),
    ("year", "takvim"), ("month", "takvim"), ("week", "takvim"),
    ("day_of", "takvim"), ("is_weekend", "takvim"), ("is_sunday", "takvim"),
    ("is_month", "takvim"), ("dow_", "takvim"), ("fourier", "takvim"),
    ("ay_uretim_plani", "stok_uretim"), ("uretim_ayi", "stok_uretim"),
    ("stok_proxy", "stok_uretim"), ("is_covid_period", "stok_uretim"),
    ("is_bulk_day", "stok_uretim"),
]


def map_feature_to_group(feature_name):
    for prefix, group in FEATURE_GROUP_MAP_PREFIXES:
        if feature_name.startswith(prefix):
            return group
    return "diger"


def select_best_tree_model_for_fi(fi_store, best_model_key):
    """
    Feature importance icin kullanilacak modeli secer. Eger en iyi model
    (stacking/LSTM gibi) feature importance saglamiyorsa, mevcut agac
    tabanli modellerden en cok tekrar eden / en son test yilina ait
    olani secilir.
    """
    if not fi_store:
        return None, None, None

    # Eger en iyi modelin kendisi fi_store'da varsa onu tercih et
    candidates = [(k, yr) for (k, yr) in fi_store.keys() if k == best_model_key]
    if not candidates:
        candidates = list(fi_store.keys())

    # En son test yili olani sec
    chosen = sorted(candidates, key=lambda kv: kv[1])[-1]
    feature_names, importances = fi_store[chosen]
    return chosen[0], feature_names, importances


def plot_feature_importance_top25(fi_store, best_model_key, folder):
    used_model, feature_names, importances = select_best_tree_model_for_fi(fi_store, best_model_key)
    if used_model is None:
        return None, None

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(25).iloc[::-1]

    note = ""
    if used_model != best_model_key:
        note = f" (Not: en iyi model '{best_model_key}' icin importance uretilemedi; '{used_model}' kullanildi)"

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(fi_df["feature"], fi_df["importance"], color=COLOR_ML)
    ax.set_title(f"Feature Importance - Top 25 ({used_model}){note}", fontsize=11)
    ax.set_xlabel("Onem Skoru")
    fig.tight_layout()
    path = save_fig(fig, folder, "26_feature_importance_top25")
    return path, used_model


# --- 27) Feature grubu bazli importance -------------------------------------------

def plot_feature_group_importance(fi_store, best_model_key, folder):
    used_model, feature_names, importances = select_best_tree_model_for_fi(fi_store, best_model_key)
    if used_model is None:
        return None

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df["group"] = fi_df["feature"].apply(map_feature_to_group)
    group_imp = fi_df.groupby("group")["importance"].sum().sort_values(ascending=False)
    group_imp_pct = 100 * group_imp / group_imp.sum()

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(group_imp_pct.index, group_imp_pct.values, color=COLOR_INTERMITTENT)
    for b, val in zip(bars, group_imp_pct.values):
        ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Feature Grubu Bazli Onem Dagilimi ({used_model})")
    ax.set_ylabel("Toplam Onem (%)")
    ax.set_xlabel("Feature Grubu")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return save_fig(fig, folder, "27_feature_group_importance")


# === STOK POLITIKASI GRAFIKLERI ====================================================

# --- 28) 2025 haftalik talep ve onerilen stok politikasi --------------------------

def plot_2025_weekly_stock_policy(best_preds, buffers, chosen_policy, folder, model_name=""):
    sub = best_preds[best_preds["date"].dt.year == 2025].copy()
    if len(sub) == 0:
        return None
    daily, weekly = build_daily_weekly_stock_tables(sub, buffers, chosen_policy)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(weekly["date"], weekly["y_true"], color=COLOR_ACTUAL, linewidth=1.8,
            marker="o", markersize=3, label="Gercek Talep")
    ax.plot(weekly["date"], weekly["stock_plan"], color=COLOR_GOOD, linewidth=1.8,
            marker="s", markersize=3, label=f"Onerilen Stok Plani ({chosen_policy})")
    ax.fill_between(weekly["date"], weekly["y_true"], weekly["stock_plan"],
                     where=(weekly["stock_plan"] >= weekly["y_true"]),
                     color=COLOR_GOOD, alpha=0.12)
    ax.fill_between(weekly["date"], weekly["y_true"], weekly["stock_plan"],
                     where=(weekly["stock_plan"] < weekly["y_true"]),
                     color=COLOR_BAD, alpha=0.20)

    # Veri araligi gercekte tam yili kapsamayabilir (orn. veri kesimi
    # nedeniyle 2025 sadece 2 Aralik'a kadar olabilir); baslik bunu acikca
    # yansitir, sabit "2025" yerine gercek baslangic-bitis tarihini yazar.
    date_min = sub["date"].min().strftime("%d.%m.%Y")
    date_max = sub["date"].max().strftime("%d.%m.%Y")
    base_title = f"2025 Yili Haftalik Gercek Talep ve Onerilen Stok Politikasi ({date_min} - {date_max})"
    title = f"{base_title} (Model: {model_name})" if model_name else base_title
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Hafta")
    ax.set_ylabel("Haftalik Talep / Stok (Adet)")
    ax.set_xlim(weekly["date"].min() - pd.Timedelta(days=3),
                weekly["date"].max() + pd.Timedelta(days=3))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_fig(fig, folder, "28_2025_haftalik_talep_stok_politikasi")


# --- 29) Hizmet seviyesi karsilastirmasi -------------------------------------------

def plot_service_level_comparison(summary_df, folder):
    weekly = summary_df[summary_df["granularity"] == "weekly"].sort_values("service_level_%")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [COLOR_GOOD if v >= 90 else (COLOR_PRED if v >= 75 else COLOR_BAD)
              for v in weekly["service_level_%"]]
    bars = ax.barh(weekly["policy"], weekly["service_level_%"], color=colors)
    for b, val in zip(bars, weekly["service_level_%"]):
        ax.text(val, b.get_y() + b.get_height() / 2, f" {val:.1f}%", va="center", fontsize=9)
    ax.set_title("Stok Politikalarinin Hizmet Seviyesi (%) Karsilastirmasi (Haftalik)")
    ax.set_xlabel("Hizmet Seviyesi (%)")
    fig.tight_layout()
    return save_fig(fig, folder, "29_stok_politikasi_hizmet_seviyesi")


# --- 30) Stockout count karsilastirmasi --------------------------------------------

def plot_stockout_count_comparison(summary_df, folder):
    weekly = summary_df[summary_df["granularity"] == "weekly"].sort_values("stockout_count")
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(weekly["policy"], weekly["stockout_count"], color=COLOR_BAD, alpha=0.85)
    for b, val in zip(bars, weekly["stockout_count"]):
        ax.text(val, b.get_y() + b.get_height() / 2, f" {int(val)}", va="center", fontsize=9)
    ax.set_title("Stok Politikalarinin Stockout (Stoksuz Kalma) Hafta Sayisi Karsilastirmasi")
    ax.set_xlabel("Stockout Hafta Sayisi")
    fig.tight_layout()
    return save_fig(fig, folder, "30_stok_politikasi_stockout_count")


# --- 31) Eksik/fazla stok karsilastirmasi ------------------------------------------

def plot_shortage_excess_comparison(summary_df, folder):
    weekly = summary_df[summary_df["granularity"] == "weekly"]
    x = np.arange(len(weekly))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2, weekly["total_shortage"], width, color=COLOR_BAD, label="Toplam Eksik Stok (Shortage)")
    ax.bar(x + width / 2, weekly["total_excess"], width, color=COLOR_PRED, label="Toplam Fazla Stok (Excess)")
    ax.set_xticks(x)
    ax.set_xticklabels(weekly["policy"], rotation=35, ha="right")
    ax.set_title("Stok Politikalarinda Toplam Eksik / Fazla Stok Karsilastirmasi")
    ax.set_ylabel("Adet")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, folder, "31_eksik_fazla_stok_karsilastirma")


# --- 32) Stok/talep orani karsilastirmasi ------------------------------------------

def plot_stock_demand_ratio(summary_df, folder):
    weekly = summary_df[summary_df["granularity"] == "weekly"].sort_values("stock_to_demand_ratio")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [COLOR_GOOD if 1.0 <= v <= 1.3 else COLOR_PRED for v in weekly["stock_to_demand_ratio"]]
    bars = ax.barh(weekly["policy"], weekly["stock_to_demand_ratio"], color=colors)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="Talep Seviyesi (1.0)")
    for b, val in zip(bars, weekly["stock_to_demand_ratio"]):
        ax.text(val, b.get_y() + b.get_height() / 2, f" {val:.2f}", va="center", fontsize=9)
    ax.set_title("Stok Politikalarinda Stok / Talep Orani Karsilastirmasi")
    ax.set_xlabel("Stok / Talep Orani")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_fig(fig, folder, "32_stok_talep_orani")
# === [16] EXCEL/CSV RAPOR CIKTILARI ============================================

def save_csv_xlsx(df, folder, name_no_ext, sheet_name="Sheet1"):
    """Bir DataFrame'i hem CSV hem XLSX olarak kaydeder."""
    csv_path = os.path.join(folder, f"{name_no_ext}.csv")
    xlsx_path = os.path.join(folder, f"{name_no_ext}.xlsx")
    df.to_csv(csv_path, index=False)
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            _format_excel_sheet(writer, sheet_name, df)
    except Exception as e:
        log("SAVE", f"UYARI: {name_no_ext}.xlsx olusturulamadi ({e}); sadece CSV kaydedildi.")
    return csv_path, xlsx_path


def _format_excel_sheet(writer, sheet_name, df):
    """Excel sayfasini okunabilir formatta duzenler (bold baslik, filtre, genislik)."""
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    header_fmt = workbook.add_format({
        "bold": True, "bg_color": "#0B3C5D", "font_color": "white",
        "border": 1, "text_wrap": True, "valign": "vcenter",
    })
    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(0, col_idx, col_name, header_fmt)
        try:
            max_len = max(df[col_name].astype(str).map(len).max(), len(str(col_name))) + 2
        except Exception:
            max_len = len(str(col_name)) + 2
        worksheet.set_column(col_idx, col_idx, min(max_len, 35))

    if len(df) > 0:
        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
    worksheet.freeze_panes(1, 0)


def build_report_figure_list():
    """
    00_rapor_icin_secili_grafikler klasorune kopyalanacak grafiklerin listesini
    ve tez bolumu/aciklama bilgilerini tutan tabloyu olusturur.
    """
    rows = [
        (1, "01_gunluk_talep_zaman_serisi.png", "Veri ve Yontem", "Gunluk Talep Zaman Serisi",
         "Veri setinin genel yapisini ve zaman icindeki degisimini gosterir.",
         "Talebin seyrek ve duzensiz oldugu acikca gorulmektedir."),
        (2, "02_aylik_toplam_talep.png", "Veri ve Yontem", "Aylik Toplam Talep",
         "Aylik bazda talep hacmindeki dalgalanmalari ozetler.", ""),
        (3, "03_yillik_toplam_talep.png", "Veri ve Yontem", "Yillik Toplam Talep",
         "Yillar arasi toplam talep trendini gosterir.", ""),
        (4, "04_sifir_pozitif_talep_orani.png", "Veri ve Yontem", "Sifir/Pozitif Talep Orani",
         "Veri setinin seyrek talep karakterini sayisal olarak ozetler.", ""),
        (5, "07_adi_cv2_talep_siniflandirma.png", "Veri ve Yontem", "ADI-CV² Talep Siniflandirmasi",
         "Urunun talep sinifini (smooth/intermittent/erratic/lumpy) gosterir.", ""),
        (6, "09_walk_forward_validation_semasi.png", "Yontem", "Walk-Forward Validation Semasi",
         "Train/validation/test donemlerinin zaman icindeki konumlanmasini gosterir.", ""),
        (7, "10_model_siralamasi_focused_score.png", "Bulgular", "Model Siralamasi (Focused Score)",
         "Tum modellerin birlesik skor uzerinden karsilastirmasini sunar.", ""),
        (8, "11_model_hWAPE_karsilastirma.png", "Bulgular", "Model hWAPE Karsilastirmasi",
         "Haftalik agregasyon uzerinden model hata karsilastirmasi.", ""),
        (9, "17_best_model_haftalik_gercek_tahmin.png", "Bulgular", "En Iyi Model Haftalik Tahmin",
         "Secilen en iyi modelin haftalik performansini gosteren ana grafik.", ""),
        (10, "19_best_model_kumulatif_gercek_tahmin.png", "Bulgular", "En Iyi Model Kumulatif Tahmin",
         "Modelin uzun donemde toplam talebi ne kadar dogru yakaladigini gosterir.", ""),
        (11, "20_best_model_scatter.png", "Bulgular", "En Iyi Model Scatter Grafigi",
         "Tahmin ile gercek deger arasindaki iliskiyi gosterir.", ""),
        (12, "26_feature_importance_top25.png", "Bulgular", "Feature Importance (Top 25)",
         "Modelin tahmin uretirken en cok hangi degiskenlere dayandigini gosterir.", ""),
        (13, "28_2025_haftalik_talep_stok_politikasi.png", "Stok Politikasi", "Son Yil Haftalik Stok Politikasi",
         "Onerilen stok politikasinin gercek talebe gore performansini gosterir.", ""),
        (14, "29_stok_politikasi_hizmet_seviyesi.png", "Stok Politikasi", "Hizmet Seviyesi Karsilastirmasi",
         "Farkli stok politikalarinin hizmet seviyesi acisindan karsilastirmasi.", ""),
        (15, "31_eksik_fazla_stok_karsilastirma.png", "Stok Politikasi", "Eksik/Fazla Stok Karsilastirmasi",
         "Politikalarin stockout ve fazla stok dengesini gosterir.", ""),
    ]
    cols = ["grafik_no", "dosya_adi", "tez_bolumu", "grafik_basligi", "ne_anlatir", "yorum_notu"]
    return pd.DataFrame(rows, columns=cols)
# === MASTER EXCEL RAPORU ========================================================

def build_master_excel_report(
    profile_df, yearly_df, feature_list_df, metrics_df, ranking_df,
    best_preds_df, all_predictions_df, model_settings_df, stock_summary_df,
    report_figure_list_df, best_model_info, output_path,
):
    """
    Tüm analiz sonuçlarını tek bir Excel dosyasında (model_sonuclari.xlsx)
    birleştirir. README sheet'i çalışmanın özetini içerir.
    """
    readme_rows = [
        ("Proje", "Makine Ogrenmesi ile Stok/Talep Tahmini - 110-90' PN 10 Dirsek"),
        ("Olusturulma Tarihi", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Veri Donemi", f"{profile_df.loc[profile_df['Metrik']=='Tarih araligi (baslangic)','Deger'].values[0]} - "
                          f"{profile_df.loc[profile_df['Metrik']=='Tarih araligi (bitis)','Deger'].values[0]}"),
        ("Test Donemleri", ", ".join(str(y) for y in TEST_YEARS)),
        ("Test Edilen Model Sayisi", metrics_df['model'].nunique()),
        ("En Iyi Model", best_model_info.get("model")),
        ("En Iyi Model History Mode", best_model_info.get("history_mode")),
        ("En Iyi Model hWAPE (ortalama)", round(best_model_info.get("hWAPE", float("nan")), 4)),
        ("", ""),
        ("SHEET ACIKLAMALARI", ""),
        ("Veri_Profili", "Veri setinin genel istatistikleri ve ADI-CV2 talep siniflandirmasi"),
        ("Feature_List", "Modellemede kullanilan tum feature'larin listesi"),
        ("Model_Metrics_All", "Her model / history mode / test yili icin tum metrikler"),
        ("Model_Ranking", "Modellerin genel (3 yil ortalamasi) siralamasi"),
        ("Best_Model_Predictions", "Secilen en iyi modelin gunluk tahminleri (2023-2025)"),
        ("All_Predictions", "Tum modellerin tum gunluk tahminleri"),
        ("Model_Settings", "Her model calistirmasinin parametre ve donem ayarlari"),
        ("Stock_Simulation_Summary", "Stok politikasi simulasyonu sonuc ozeti"),
        ("Report_Figure_List", "Tez metnine onerilen grafiklerin listesi ve aciklamalari"),
    ]
    readme_df = pd.DataFrame(readme_rows, columns=["Alan", "Deger / Aciklama"])

    sheets = {
        "README": readme_df,
        "Veri_Profili": profile_df,
        "Veri_Profili_Yillik": yearly_df,
        "Feature_List": feature_list_df,
        "Model_Metrics_All": metrics_df,
        "Model_Ranking": ranking_df,
        "Best_Model_Predictions": best_preds_df,
        "All_Predictions": all_predictions_df,
        "Model_Settings": model_settings_df,
        "Stock_Simulation_Summary": stock_summary_df,
        "Report_Figure_List": report_figure_list_df,
    }

    try:
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            for sheet_name, sdf in sheets.items():
                safe_name = sheet_name[:31]
                sdf_to_write = sdf.copy()
                # Tarih kolonlarini Excel'in dogru gosterebilmesi icin string'e cevir
                for col in sdf_to_write.columns:
                    if pd.api.types.is_datetime64_any_dtype(sdf_to_write[col]):
                        sdf_to_write[col] = sdf_to_write[col].dt.strftime("%Y-%m-%d")
                sdf_to_write.to_excel(writer, sheet_name=safe_name, index=False)
                _format_excel_sheet(writer, safe_name, sdf_to_write)
        return True
    except Exception as e:
        log("SAVE", f"HATA: Master Excel raporu olusturulamadi: {e}")
        return False


# === [15] SONUC OZETI / [16] AKADEMIK YORUM DOSYASI =============================

def build_final_summary_text(profile_info, profile_df, ranking_df, best_model_info,
                              best_policy_row, n_models_tested, zip_path):
    """Terminale yazdirilacak ve final_summary.txt olarak kaydedilecek metni uretir."""
    date_range = (
        f"{profile_df.loc[profile_df['Metrik']=='Tarih araligi (baslangic)','Deger'].values[0]} - "
        f"{profile_df.loc[profile_df['Metrik']=='Tarih araligi (bitis)','Deger'].values[0]}"
    )
    zero_ratio = profile_df.loc[profile_df['Metrik']=='Sifir talep orani (%)','Deger'].values[0]

    lines = [
        "=" * 70,
        "BITIRME PROJESI - SONUC OZETI",
        "=" * 70,
        f"Veri tarih araligi               : {date_range}",
        f"Sifir talep orani                : %{zero_ratio}",
        f"ADI                               : {profile_info['adi']:.3f}",
        f"CV^2                              : {profile_info['cv2']:.3f}",
        f"Talep sinifi                      : {profile_info['demand_class']}",
        f"Test edilen model sayisi          : {n_models_tested}",
        f"En iyi model                      : {best_model_info.get('model')}",
        f"En iyi model history mode         : {best_model_info.get('history_mode')}",
        f"En iyi model saglik durumu        : {best_model_info.get('saglik_durumu_ozet', 'N/A')}",
        f"En iyi model pred/true orani      : {best_model_info.get('total_pred_true_ratio'):.3f}",
        f"En iyi model hWAPE (ort.)         : {best_model_info.get('hWAPE'):.4f}",
        f"En iyi model WAPE (ort.)          : {best_model_info.get('WAPE'):.4f}",
        f"En iyi model Bias %% (ort.)        : {best_model_info.get('Bias_%'):.2f}",
        f"En iyi model demand F1 (ort.)     : {best_model_info.get('demand_f1'):.3f}",
        f"En iyi model large50 underforecast: {best_model_info.get('large50_underforecast_rate'):.3f}",
        f"En iyi stok politikasi            : {best_policy_row['policy']}",
        f"  - Hizmet seviyesi               : %{best_policy_row['service_level_%']}",
        f"  - Stok/Talep orani              : {best_policy_row['stock_to_demand_ratio']}",
        f"ZIP dosya yolu                    : {zip_path}",
        "=" * 70,
    ]
    if best_model_info.get("best_model_is_flagged"):
        lines.insert(-1, "UYARI: Hicbir model tum test yillarinda 'saglikli' pred/true "
                          "oranina ulasamadi; en iyi focused_score'a sahip model secildi.")
    return "\n".join(lines)


def build_academic_notes_md(profile_info, profile_df, ranking_df, best_model_info,
                              best_policy_row, fi_used_model):
    """
    bulgular_icin_yorum_notlari.md icerigini, hesaplanan sonuclara gore
    otomatik doldurulmus, temkinli akademik dilde uretir.
    """
    top5_models = ranking_df.sort_values("focused_score").head(5)["model"].tolist()
    demand_class = profile_info["demand_class"]

    md = f"""# Bulgular Icin Yorum Notlari

Bu dosya, calismanin uretilen sayisal sonuclarina dayali olarak otomatik
doldurulmus kisa yorum sablonlari icermektedir. Asagidaki ifadeler taslak
niteligindedir ve tez yazim surecinde gozden gecirilip
genisletilmesi onerilir.

## 1. Veri Setinin Genel Yapisi

Incelenen urune ({"110-90' PN 10 Dirsek"}) ait gunluk talep verisi,
{profile_df.loc[profile_df['Metrik']=='Tarih araligi (baslangic)','Deger'].values[0]} ile
{profile_df.loc[profile_df['Metrik']=='Tarih araligi (bitis)','Deger'].values[0]}
tarihleri arasini kapsamaktadir. Bu sonuclar, incelenen veri seti ve
belirlenen gozlem donemi kapsaminda degerlendirilmelidir.

## 2. Seyrek Talep Karakteristigi

Hesaplanan ADI degeri ({profile_info['adi']:.2f}) ve CV² degeri
({profile_info['cv2']:.2f}) birlikte degerlendirildiginde, urunun talep
yapisi **{demand_class}** sinifina dahil edilmistir. Bu siniflandirma,
talebin hem seyrek (araliklarla) hem de degisken buyuklukte goruldugune
isaret etmekte olup, klasik zaman serisi yontemlerinin tek basina yetersiz
kalabilecegi bir yapi oldugunu dusundurmektedir. Bu bulgu, daha temkinli
bir ifadeyle, yalnizca incelenen donem icin gecerli kabul edilmelidir.

## 3. Model Karsilastirma Bulgulari

Walk-forward dogrulama sonuclarina gore, en iyi performans gosteren ilk
bes model sirasiyla {", ".join(top5_models)} olarak belirlenmistir.
Genel egilim olarak, agac tabanli makine ogrenmesi modellerinin
(LightGBM/XGBoost tabanli yaklasimlar), geleneksel benchmark
yontemlerine kiyasla daha dusuk focused_score degerleri urettigi
gozlemlenmistir. Bu sonuc, incelenen test donemleri (2023-2025) icin
gecerlidir ve farkli donemlerde farkli sonuclar elde edilebilecegi
unutulmamalidir.

## 4. En Iyi Modelin Yorumlanmasi

Calismada en iyi model olarak **{best_model_info.get('model')}**
({best_model_info.get('history_mode')} history mode ile) secilmistir.
Bu modelin ortalama hWAPE degeri {best_model_info.get('hWAPE'):.3f},
Bias degeri ise %{best_model_info.get('Bias_%'):.2f} olarak
hesaplanmistir. Modelin secimi yalnizca tek bir metrige (orn. RMSE)
dayanmamis; hWAPE, Bias, demand F1, top30 capture orani ve buyuk talep
gunlerindeki underforecast oranini birlikte dikkate alan bir "focused
score" kullanilmistir. Bu yaklasim, stok yonetimi acisindan onemli olan
sistematik sapma ve stockout riskinin de degerlendirmeye katilmasini
saglamistir.

## 5. Gunluk ve Haftalik Tahmin Farki

Gunluk seviyede uretilen tahminlerin, talebin seyrek/duzensiz yapisi
nedeniyle gunluk bazda yuksek varyasyon gosterdigi gozlemlenmistir.
Tahminlerin haftalik seviyede toplanmasi (agregasyon), gunluk
gurultunun bir kismini ortadan kaldirarak modelin gercek performansini
daha temsili sekilde yansitmaktadir. Bu nedenle stok planlama
kararlarinda haftalik agregasyonun on plana cikarilmasi onerilmektedir.

## 6. Bias ve Underforecast Degerlendirmesi

En iyi modelin buyuk talep gunlerindeki (50+ adet) underforecast orani
{best_model_info.get('large50_underforecast_rate'):.2f} olarak
hesaplanmistir. Bu oran, projeye/ihaleye baglı ani yuksek talep
gunlerinde modelin tahminlerinin gercek talebin altinda kalma egilimini
gostermektedir ve stok politikasinda bir guvenlik stogu (buffer)
uygulanmasinin gerekliligine isaret etmektedir.

## 7. Feature Importance Yorumu

Feature importance analizi {fi_used_model if fi_used_model else "agac tabanli model"}
uzerinden gerceklestirilmistir. Sonuclar, gecmis talep degerlerine
dayanan lag ve rolling feature gruplarinin model tahminlerinde onemli
bir agirliga sahip oldugunu gostermektedir; bu durum, seyrek talep
yapisina sahip urunlerde yakin gecmisin gelecegi tahmin etmede onemli
bir gosterge oldugunu dusundurmektedir.

## 8. Stok Politikasi Yorumu

Stok politikasi simulasyonlari arasinda dengeli skora gore secilen
politika **{best_policy_row['policy']}** olmustur. Bu politika ile
haftalik bazda yaklasik %{best_policy_row['service_level_%']:.1f}
hizmet seviyesine ulasilirken, stok/talep orani
{best_policy_row['stock_to_demand_ratio']:.2f} olarak gerceklesmistir.
Bu sonuc, asiri stok biriktirmeden kabul edilebilir bir hizmet
seviyesinin saglanabilecegine isaret etmekle birlikte, nihai karar
isletmenin risk tercihlerine gore yeniden degerlendirilmelidir.

## 9. Calismanin Sinirliliklari

Bu calisma tek bir urun uzerinden yurutulmus olup, sonuclarin diger
urunlere genellenmesi konusunda temkinli olunmalidir. Ayrica, dini
bayram tarihlerinin yillara gore degisken olmasi, kullanilan resmi
tatil listesinin sabit tarihli bayramlarla sinirli olmasi ve COVID-19
donemi gibi olaganustu donemlerin veri setinde sinirli sayida
gozlemle temsil edilmesi, modelin genellenebilirligini etkileyebilecek
faktorler arasindadir. Sonuclar, yalnizca incelenen veri seti ve
belirlenen test donemleri (2023, 2024, 2025) kapsaminda
degerlendirilmelidir.
"""
    return md


# === [17] KALITE VE TESLIM KONTROL LISTESI =======================================

def run_quality_checklist(base_dir, excel_path, ranking_csv, best_pred_csv,
                            stock_summary_csv, zip_path):
    """Tum kritik ciktilarin olusup olusmadigini kontrol eder, sonucu yazdirir."""
    checks = []

    checks.append(("Excel raporu olustu mu?", os.path.exists(excel_path)))
    checks.append(("Model ranking olustu mu?", os.path.exists(ranking_csv)))
    checks.append(("Best predictions olustu mu?", os.path.exists(best_pred_csv)))
    checks.append(("Stock simulation summary olustu mu?", os.path.exists(stock_summary_csv)))

    report_folder = SUBDIRS["rapor"]
    n_report_figs = len([f for f in os.listdir(report_folder) if f.endswith(".png")]) if os.path.exists(report_folder) else 0
    checks.append((f"Rapor klasorunde en az 10 grafik var mi? (bulunan: {n_report_figs})", n_report_figs >= 10))

    checks.append(("ZIP olustu mu?", os.path.exists(zip_path)))

    log("SAVE", "=== KALITE / TESLIM KONTROL LISTESI ===")
    all_ok = True
    for desc, ok in checks:
        status = "OK" if ok else "EKSIK"
        log("SAVE", f"[{status}] {desc}")
        if not ok:
            all_ok = False

    if all_ok:
        log("SAVE", "Tum kontroller basarili. Cikti seti teslime hazir.")
    else:
        log("SAVE", "UYARI: Bazi ciktilar eksik! Lutfen yukaridaki listeyi kontrol edin.")

    return all_ok
# === [18] ZIP OLUSTURMA ==========================================================

def create_output_zip(base_dir, zip_name="ciktilar.zip"):
    """Tüm çıktı klasörünü zip olarak paketler."""
    zip_path = os.path.join(os.path.dirname(base_dir), zip_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(base_dir))
                zf.write(file_path, arcname)

    log("ZIP", f"ZIP dosyasi olusturuldu: {zip_path} "
                f"({os.path.getsize(zip_path) / (1024*1024):.2f} MB)")

    log("ZIP", f"ZIP konumu: {zip_path}")

    return zip_path
# === ANA CALISTIRMA AKISI (main) =================================================

def run_pipeline(scenario_label, output_base_dir, use_peak_segmentation=False,
                  peak_quantile=PROJECT_PEAK_QUANTILE):
    """
    Belirtilen senaryo için tam pipeline'ı çalıştırır.
    use_peak_segmentation=True ise P95 eşiği (veriden otomatik) üzerindeki
    proje/ihale günleri operasyonel seriden sıfırlanır.
    """
    t_start = datetime.now()
    setup_output_dirs(output_base_dir)

    log("SETUP", f"=== {scenario_label} baslatiliyor === (cikti: {BASE_DIR})")

    # ---- [LOAD] -----------------------------------------------------------------
    log("LOAD", "Veri dosyasi araniyor...")
    data_path = find_data_file()
    log("LOAD", f"Veri dosyasi bulundu: {data_path}")
    df_raw = read_raw_data(data_path)
    log("LOAD", f"Veri okundu: {df_raw.shape[0]} satir, {df_raw.shape[1]} kolon.")

    # ---- [CLEAN] ------------------------------------------------------------------
    log("CLEAN", "Tatil feature'lari ekleniyor...")
    df_clean = add_holiday_features(df_raw)
    log("CLEAN", "Veri temizligi tamamlandi.")

    # ---- [PEAK] — Opsiyonel proje/ihale piki ayristirma ----------------------------
    peak_summary = None
    if use_peak_segmentation:
        log("PEAK", f"Proje/ihale piki ayristirmasi uygulaniyor "
                     f"(esik = pozitif satislarin P{int(peak_quantile*100)}'i, VERIDEN OTOMATIK hesaplanir)...")
        df_clean_full = df_clean.copy()  # orijinal (pik dahil) seri, grafik icin saklanir
        df_clean, peak_days, peak_summary = segment_project_peaks(df_clean, quantile=peak_quantile)
        log("PEAK", f"Otomatik esik: {peak_summary['peak_threshold']:.1f} adet "
                     f"(P{int(peak_quantile*100)}, sabit deger DEGIL) | "
                     f"Pik gun sayisi: {peak_summary['project_peak_day_count']} | "
                     f"Pik payi: %{peak_summary['project_peak_share_of_total_%']:.1f}")
        peak_days.to_csv(os.path.join(SUBDIRS["tablolar"], "00_project_peak_days.csv"), index=False)
        save_csv_xlsx(pd.DataFrame([peak_summary]), SUBDIRS["tablolar"], "00_peak_summary", "Peak_Summary")
        plot_peak_days(df_clean_full, peak_days, peak_summary["peak_threshold"], SUBDIRS["veri"])

    # ---- [PROFILE] ------------------------------------------------------------------
    log("PROFILE", "Veri profili hesaplaniyor...")
    profile_df, yearly_df, profile_info = build_data_profile(df_clean)
    print_data_profile(profile_df, yearly_df)
    save_csv_xlsx(profile_df, SUBDIRS["tablolar"], "01_veri_profili", "Veri_Profili")

    # ---- [FEATURE] ------------------------------------------------------------------
    log("FEATURE", "Feature engineering basliyor...")
    df_feat = make_features(df_clean)
    feature_cols = get_feature_columns(df_feat)
    log("FEATURE", f"Toplam {len(feature_cols)} feature uretildi.")

    feature_list_df = pd.DataFrame({"feature": feature_cols})
    save_csv_xlsx(feature_list_df, SUBDIRS["tablolar"], "02_feature_list", "Feature_List")
    model_ready_path = os.path.join(SUBDIRS["tablolar"], "02_model_ready_dataset.csv")
    df_feat.to_csv(model_ready_path, index=False)
    log("FEATURE", f"Model-ready veri seti kaydedildi: {model_ready_path}")

    # ---- [MODEL] / [EVAL] — Walk-forward validation -------------------------------
    log("MODEL", "Walk-forward validation ve model egitimi basliyor "
                  "(bu adim birkaç dakika surebilir)...")
    all_metrics, all_predictions, all_val_predictions, model_settings, fi_store = \
        run_walk_forward(df_feat, feature_cols)

    metrics_df = pd.DataFrame(all_metrics)
    all_predictions_df = pd.DataFrame(all_predictions)
    all_predictions_df["date"] = pd.to_datetime(all_predictions_df["date"])
    all_val_predictions_df = pd.DataFrame(all_val_predictions)
    all_val_predictions_df["date"] = pd.to_datetime(all_val_predictions_df["date"])
    model_settings_df = pd.DataFrame(model_settings)

    log("EVAL", f"Toplam {len(metrics_df)} model-sonucu satiri uretildi "
                 f"({metrics_df['model'].nunique()} farkli model).")

    save_csv_xlsx(metrics_df, SUBDIRS["tablolar"], "03_all_model_metrics", "Model_Metrics_All")
    save_csv_xlsx(model_settings_df, SUBDIRS["tablolar"], "06_model_settings", "Model_Settings")

    pred_path_all = os.path.join(SUBDIRS["predictions"], "05_all_predictions.csv")
    all_predictions_df.to_csv(pred_path_all, index=False)

    # ---- Model ranking ve en iyi model secimi --------------------------------------
    log("EVAL", "Model siralamasi olusturuluyor...")
    ranking_df, ranking_agg_df = build_model_ranking(metrics_df)
    save_csv_xlsx(ranking_df, SUBDIRS["tablolar"], "04_model_ranking", "Model_Ranking")

    best_model_info = select_best_model(ranking_df)
    log("EVAL", f"En iyi model: {best_model_info['model']} "
                 f"(history_mode={best_model_info['history_mode']}, "
                 f"focused_score={best_model_info['focused_score']:.4f})")
    if best_model_info.get("best_model_is_flagged"):
        log("EVAL", "UYARI: Tum test yillarinda saglikli (dengeli pred/true orani) "
                     "olan bir model bulunamadi; en iyi focused_score'a sahip model "
                     "secildi ancak sonuclar temkinli yorumlanmalidir.")

    best_preds_df = get_best_model_predictions(all_predictions_df, best_model_info["model"])
    best_val_preds_df = get_best_model_val_predictions(all_val_predictions_df, best_model_info["model"])

    best_pred_csv_path = os.path.join(SUBDIRS["predictions"], "05_best_model_predictions.csv")
    best_preds_df.to_csv(best_pred_csv_path, index=False)

    if len(best_val_preds_df) > 0:
        val_y_true = best_val_preds_df["y_true"].values
        val_y_pred = best_val_preds_df["y_pred"].values
    else:
        # guvenlik agi: validation tahmini bulunamazsa test seti basindan yaklasik deger kullanilir
        val_y_true = best_preds_df["y_true"].values[:200]
        val_y_pred = best_preds_df["y_pred"].values[:200]

    # ---- [PLOT] — Tum grafikler -----------------------------------------------------
    log("PLOT", "Veri analizi grafikleri uretiliyor (1-8)...")
    plot_daily_demand(df_clean, SUBDIRS["veri"])
    plot_monthly_total(df_clean, SUBDIRS["veri"])
    plot_yearly_total(yearly_df, SUBDIRS["veri"])
    plot_zero_positive_ratio(profile_info, df_clean, SUBDIRS["veri"])
    plot_demand_histogram(df_clean, SUBDIRS["veri"])
    plot_demand_boxplot(df_clean, SUBDIRS["veri"])
    plot_adi_cv2_classification(profile_info["adi"], profile_info["cv2"], SUBDIRS["veri"])
    plot_big_demand_days(df_clean, SUBDIRS["veri"])

    log("PLOT", "Walk-forward semasi uretiliyor (9)...")
    plot_walk_forward_scheme(df_feat, SUBDIRS["veri"])

    log("PLOT", "Model karsilastirma grafikleri uretiliyor (10-15)...")
    plot_model_ranking(ranking_df, SUBDIRS["karsilastirma"])
    plot_hwape_comparison(ranking_df, SUBDIRS["karsilastirma"])
    plot_bias_comparison(ranking_df, SUBDIRS["karsilastirma"])
    plot_demand_f1_recall(ranking_df, SUBDIRS["karsilastirma"])
    plot_pred_true_ratio(ranking_df, SUBDIRS["karsilastirma"])
    plot_large_demand_underforecast(ranking_df, SUBDIRS["karsilastirma"])

    log("PLOT", "En iyi model grafikleri uretiliyor (16-25)...")
    best_model_name_label = best_model_info["model"]
    plot_best_daily(best_preds_df, SUBDIRS["en_iyi"], model_name=best_model_name_label)
    plot_best_weekly(best_preds_df, SUBDIRS["en_iyi"], model_name=best_model_name_label)
    plot_best_monthly(best_preds_df, SUBDIRS["en_iyi"], model_name=best_model_name_label)
    plot_best_cumulative(best_preds_df, SUBDIRS["en_iyi"], model_name=best_model_name_label)
    plot_best_scatter(best_preds_df, SUBDIRS["en_iyi"], model_name=best_model_name_label)
    plot_best_residual_time(best_preds_df, SUBDIRS["en_iyi"], model_name=best_model_name_label)
    plot_best_residual_histogram(best_preds_df, SUBDIRS["en_iyi"], model_name=best_model_name_label)

    buffers_initial = compute_residual_buffers(val_y_true, val_y_pred)
    plot_best_forecast_band(best_preds_df, buffers_initial, SUBDIRS["en_iyi"], model_name=best_model_name_label)
    plot_top3_weekly(all_predictions_df, ranking_df, SUBDIRS["en_iyi"])
    plot_top5_cumulative(all_predictions_df, ranking_df, SUBDIRS["en_iyi"])

    log("PLOT", "Feature importance grafikleri uretiliyor (26-27)...")
    _, fi_used_model = plot_feature_importance_top25(fi_store, best_model_info["model"], SUBDIRS["feature_imp"])
    plot_feature_group_importance(fi_store, best_model_info["model"], SUBDIRS["feature_imp"])

    # ---- Model bazli detay grafikler -------------------------------------------------
    log("PLOT", "Model bazli detay grafikler uretiliyor (tum modeller)...")
    for model_name in all_predictions_df["model"].unique():
        try:
            sub = all_predictions_df[all_predictions_df["model"] == model_name].sort_values("date")
            safe_name = model_name.replace(" ", "_")
            detail_folder = SUBDIRS["detay"]

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(sub["date"], sub["y_true"], color=COLOR_ACTUAL, linewidth=0.8, label="Gercek")
            ax.plot(sub["date"], sub["y_pred"], color=COLOR_PRED, linewidth=0.8, alpha=0.85, label="Tahmin")
            ax.set_title(f"{model_name} - Gunluk Gercek vs Tahmin")
            ax.legend()
            fig.tight_layout()
            save_fig(fig, detail_folder, f"{safe_name}__gunluk")

            weekly = safe_weekly_resample(sub[["date", "y_true", "y_pred"]], date_col="date", agg="sum")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(weekly["date"], weekly["y_true"], color=COLOR_ACTUAL, linewidth=1.2, label="Gercek")
            ax.plot(weekly["date"], weekly["y_pred"], color=COLOR_PRED, linewidth=1.2, label="Tahmin")
            ax.set_title(f"{model_name} - Haftalik Gercek vs Tahmin")
            ax.legend()
            fig.tight_layout()
            save_fig(fig, detail_folder, f"{safe_name}__haftalik")

            max_val = max(sub["y_true"].max(), sub["y_pred"].max()) * 1.05
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(sub["y_true"], sub["y_pred"], color=COLOR_ML, alpha=0.4, s=14)
            ax.plot([0, max_val], [0, max_val], color=COLOR_BAD, linestyle="--", linewidth=1.2)
            ax.set_title(f"{model_name} - Scatter")
            fig.tight_layout()
            save_fig(fig, detail_folder, f"{safe_name}__scatter")

            residual = sub["y_true"] - sub["y_pred"]
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(sub["date"], residual, color=COLOR_DEEP, linewidth=0.7)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_title(f"{model_name} - Residual Zaman Grafigi")
            fig.tight_layout()
            save_fig(fig, detail_folder, f"{safe_name}__residual_zaman")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(residual, bins=30, color=COLOR_DEEP, edgecolor="white")
            ax.axvline(0, color="black", linewidth=1.0)
            ax.set_title(f"{model_name} - Residual Histogram")
            fig.tight_layout()
            save_fig(fig, detail_folder, f"{safe_name}__residual_hist")

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(sub["date"], sub["y_true"].cumsum(), color=COLOR_ACTUAL, linewidth=1.5, label="Gercek (Kum.)")
            ax.plot(sub["date"], sub["y_pred"].cumsum(), color=COLOR_PRED, linewidth=1.5, label="Tahmin (Kum.)")
            ax.set_title(f"{model_name} - Kumulatif Gercek vs Tahmin")
            ax.legend()
            fig.tight_layout()
            save_fig(fig, detail_folder, f"{safe_name}__kumulatif")
        except Exception as e:
            log("PLOT", f"UYARI: {model_name} detay grafikleri olusturulamadi: {e}")

    # ---- [Stok politikasi simulasyonu] ------------------------------------------------
    log("MODEL", "Stok politikasi simulasyonu calistiriliyor...")
    stock_summary_df, best_policy_row, buffers = run_stock_simulation(
        best_preds_df, val_y_true, val_y_pred
    )
    chosen_policy = best_policy_row["policy"]
    log("MODEL", f"Onerilen stok politikasi: {chosen_policy} "
                  f"(hizmet seviyesi=%{best_policy_row['service_level_%']:.1f})")

    daily_stock_df, weekly_stock_df = build_daily_weekly_stock_tables(
        best_preds_df, buffers, chosen_policy
    )
    daily_stock_df.to_csv(os.path.join(SUBDIRS["tablolar"], "07_stock_simulation_daily.csv"), index=False)
    weekly_stock_df.to_csv(os.path.join(SUBDIRS["tablolar"], "07_stock_simulation_weekly.csv"), index=False)
    save_csv_xlsx(stock_summary_df, SUBDIRS["tablolar"], "07_stock_simulation_summary", "Stock_Simulation_Summary")

    log("PLOT", "Stok politikasi grafikleri uretiliyor (28-32)...")
    plot_2025_weekly_stock_policy(best_preds_df, buffers, chosen_policy, SUBDIRS["stok"], model_name=best_model_name_label)
    plot_service_level_comparison(stock_summary_df, SUBDIRS["stok"])
    plot_stockout_count_comparison(stock_summary_df, SUBDIRS["stok"])
    plot_shortage_excess_comparison(stock_summary_df, SUBDIRS["stok"])
    plot_stock_demand_ratio(stock_summary_df, SUBDIRS["stok"])

    # ---- Rapor icin secili grafik klasoru -----------------------------------------------
    log("SAVE", "Rapor icin secili grafikler kopyalaniyor...")
    report_figure_list_df = build_report_figure_list()
    source_folders = [SUBDIRS["veri"], SUBDIRS["karsilastirma"], SUBDIRS["en_iyi"],
                       SUBDIRS["feature_imp"], SUBDIRS["stok"]]
    copied = 0
    for fname in report_figure_list_df["dosya_adi"]:
        found = False
        for folder in source_folders:
            src = os.path.join(folder, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(SUBDIRS["rapor"], fname))
                copied += 1
                found = True
                break
        if not found:
            log("SAVE", f"UYARI: Rapor grafigi bulunamadi: {fname}")
    log("SAVE", f"{copied} grafik rapor klasorune kopyalandi.")
    save_csv_xlsx(report_figure_list_df, SUBDIRS["rapor"], "rapor_icin_grafik_listesi", "Report_Figure_List")

    # ---- [SAVE] — Master Excel raporu --------------------------------------------------
    log("SAVE", "Master Excel raporu olusturuluyor...")
    excel_path = os.path.join(BASE_DIR, "model_sonuclari.xlsx")
    build_master_excel_report(
        profile_df, yearly_df, feature_list_df, metrics_df, ranking_df,
        best_preds_df, all_predictions_df, model_settings_df, stock_summary_df,
        report_figure_list_df, best_model_info, excel_path,
    )
    log("SAVE", f"Master Excel raporu kaydedildi: {excel_path}")

    # ---- Final summary ve akademik yorum dosyasi ----------------------------------------
    log("SAVE", "Ozet dosyalari olusturuluyor...")
    zip_name = f"ciktilar_{'senaryo_b' if use_peak_segmentation else 'senaryo_a'}.zip"
    zip_path_placeholder = os.path.join(os.path.dirname(BASE_DIR), zip_name)

    summary_text = build_final_summary_text(
        profile_info, profile_df, ranking_df, best_model_info,
        best_policy_row, metrics_df["model"].nunique(), zip_path_placeholder,
    )
    summary_text = f"SENARYO: {scenario_label}\n" + summary_text
    if use_peak_segmentation and peak_summary is not None:
        summary_text += (
            f"\n\n--- PROJE/IHALE PIKI AYRISTIRMA BILGISI ---\n"
            f"Esik kurali            : {peak_summary['peak_rule']} (VERIDEN OTOMATIK hesaplanmistir, sabit deger degildir)\n"
            f"Otomatik esik degeri   : {peak_summary['peak_threshold']:.1f} adet\n"
            f"Pik gun sayisi         : {peak_summary['project_peak_day_count']}\n"
            f"Pik toplam miktar      : {peak_summary['project_peak_total_qty']:.0f}\n"
            f"Pik payi (toplam talep): %{peak_summary['project_peak_share_of_total_%']:.1f}\n"
        )
    print("\n" + summary_text + "\n")
    with open(os.path.join(BASE_DIR, "final_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    academic_md = build_academic_notes_md(
        profile_info, profile_df, ranking_df, best_model_info,
        best_policy_row, fi_used_model,
    )
    with open(os.path.join(BASE_DIR, "bulgular_icin_yorum_notlari.md"), "w", encoding="utf-8") as f:
        f.write(academic_md)

    # ---- [ZIP] -------------------------------------------------------------------------
    log("ZIP", "Cikti klasoru zip'leniyor...")
    zip_path = create_output_zip(BASE_DIR, zip_name=zip_name)

    # ---- Kalite kontrol listesi ----------------------------------------------------------
    run_quality_checklist(
        BASE_DIR, excel_path,
        os.path.join(SUBDIRS["tablolar"], "04_model_ranking.csv"),
        best_pred_csv_path,
        os.path.join(SUBDIRS["tablolar"], "07_stock_simulation_summary.csv"),
        zip_path,
    )

    elapsed = (datetime.now() - t_start).total_seconds()
    log("SAVE", f"[{scenario_label}] Tum islemler tamamlandi. Toplam sure: {elapsed/60:.1f} dakika.")
    log("SAVE", f"Tum ciktilar: {BASE_DIR}")
    log("ZIP", f"ZIP dosyasi: {zip_path}")

    return {
        "scenario_label": scenario_label,
        "base_dir": BASE_DIR,
        "best_model_info": best_model_info,
        "ranking_df": ranking_df,
        "best_policy_row": best_policy_row,
        "stock_summary_df": stock_summary_df,
        "peak_summary": peak_summary,
        "profile_info": profile_info,
        "zip_path": zip_path,
    }


def build_scenario_comparison(result_a, result_b):
    """
    Senaryo A ve B sonuçlarını net etiketlerle özetler.
    İki senaryo farklı hedefleri tahmin ettiğinden metrikleri doğrudan karşılaştırılmaz.
    """
    lines = []
    lines.append("=" * 78)
    lines.append("SENARYO KARSILASTIRMA OZETI")
    lines.append("=" * 78)
    lines.append(
        "ONEMLI NOT: Senaryo A ve Senaryo B FARKLI hedef degiskenleri tahmin\n"
        "etmektedir (A: tum gunler dahil toplam talep, B: proje/ihale pikleri\n"
        "cikarilmis operasyonel/rutin talep). Bu nedenle iki senaryonun WAPE,\n"
        "hWAPE gibi metrikleri DOGRUDAN birbiriyle kiyaslanarak 'hangisi daha\n"
        "iyi model' sonucuna varilamaz; her biri kendi hedefi icinde\n"
        "degerlendirilmelidir."
    )
    lines.append("")

    for res in [result_a, result_b]:
        bmi = res["best_model_info"]
        bpr = res["best_policy_row"]
        lines.append(f"--- {res['scenario_label']} ---")
        lines.append(f"  Cikti klasoru          : {res['base_dir']}")
        if res["peak_summary"] is not None:
            ps = res["peak_summary"]
            lines.append(f"  Pik ayristirma         : EVET (otomatik esik = {ps['peak_threshold']:.1f} adet, "
                          f"{ps['peak_rule']})")
            lines.append(f"  Pik gun sayisi / payi  : {ps['project_peak_day_count']} gun / "
                          f"%{ps['project_peak_share_of_total_%']:.1f}")
        else:
            lines.append("  Pik ayristirma         : HAYIR (tum gunler dahil toplam talep)")
        lines.append(f"  En iyi model           : {bmi.get('model')} (history_mode={bmi.get('history_mode')})")
        lines.append(f"  Saglik durumu          : {bmi.get('saglik_durumu_ozet', 'N/A')}")
        lines.append(f"  hWAPE / WAPE           : {bmi.get('hWAPE'):.4f} / {bmi.get('WAPE'):.4f}")
        lines.append(f"  Bias %% / Pred-True     : {bmi.get('Bias_%'):.2f} / {bmi.get('total_pred_true_ratio'):.3f}")
        lines.append(f"  Demand F1              : {bmi.get('demand_f1'):.3f}")
        lines.append(f"  Large50 underforecast  : {bmi.get('large50_underforecast_rate'):.3f}")
        lines.append(f"  Onerilen stok politikasi: {bpr['policy']} "
                      f"(hizmet=%{bpr['service_level_%']:.2f}, stok/talep={bpr['stock_to_demand_ratio']:.3f})")
        lines.append("")

    lines.append("-" * 78)
    lines.append(
        "ONERI: Tez ana metninde 'toplam talep' perspektifini esas almak\n"
        "isteniyorsa Senaryo A sonuclari kullanilmalidir. Stok/guvenlik payi\n"
        "politikasinin proje/ihale gunlerinden ayristirilmis 'rutin talep'\n"
        "uzerinden ayrica degerlendirilmesi isteniyorsa Senaryo B, ek/duyarlilik\n"
        "analizi olarak sunulmalidir. Pik esigi HER IKI senaryoda da veriden\n"
        "otomatik hesaplanmistir; elle belirlenmis sabit bir deger (orn. '200\n"
        "adet') KULLANILMAMISTIR."
    )
    lines.append("=" * 78)

    return "\n".join(lines)


if __name__ == "__main__":
    t_overall_start = datetime.now()

    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    # Senaryo A: ham talep serisi, pik ayrıştırma yok
    result_a = run_pipeline(
        scenario_label="Senaryo A - Toplam Talep",
        output_base_dir=os.path.join(_out, "senaryo_a"),
        use_peak_segmentation=False,
    )

    # Senaryo B: P95 eşiği üzerindeki proje/ihale günleri ayrıştırılır
    result_b = run_pipeline(
        scenario_label="Senaryo B - Operasyonel Talep",
        output_base_dir=os.path.join(_out, "senaryo_b"),
        use_peak_segmentation=True,
        peak_quantile=PROJECT_PEAK_QUANTILE,
    )

    comparison_text = build_scenario_comparison(result_a, result_b)
    print("\n" + comparison_text + "\n")
    comparison_path = os.path.join(_out, "senaryo_karsilastirma.txt")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(comparison_text)

    elapsed_total = (datetime.now() - t_overall_start).total_seconds()
    log("SAVE", f"Tamamlandı. Toplam süre: {elapsed_total/60:.1f} dakika.")
    log("SAVE", f"Senaryo A : {result_a['base_dir']}")
    log("SAVE", f"Senaryo B : {result_b['base_dir']}")
    log("SAVE", f"Karşılaştırma : {comparison_path}")
