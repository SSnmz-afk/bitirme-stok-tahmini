# Veri Klasörü

## Gerçek Veri

Gerçek firma verisi (**`gunluk_ml_dataset.csv`**) gizlilik nedeniyle bu repoda
bulunmamaktadır. Pipeline'ı çalıştırmak için bu dosyayı `data/` klasörüne koyun.

## Örnek Veri

**`sample_gunluk_ml_dataset.csv`** — Gerçek verinin yapısını gösteren, rastgele
üretilmiş 365 günlük anonim örnek veridir. Gerçek satış rakamlarını içermez.

---

## Beklenen Sütunlar

| Sütun | Tür | Zorunlu | Açıklama |
|---|---|---|---|
| `date` | YYYY-MM-DD | ✅ | Tarih |
| `sales_qty` | float | ✅ | Günlük satış miktarı (adet) |
| `is_dini_bayram` | 0/1 | — | Dini bayram günü |
| `bayram_oncesi_3g` | 0/1 | — | Bayram öncesi 3 gün |
| `bayram_sonrasi_3g` | 0/1 | — | Bayram sonrası 3 gün |
| `ay_uretim_plani` | float | — | Aylık üretim planı |
| `stok_proxy` | float | — | Stok vekil değişkeni |
| `is_covid_period` | 0/1 | — | Covid dönemi bayrağı |

Opsiyonel sütunlar yoksa pipeline otomatik olarak 0 ile doldurur.
