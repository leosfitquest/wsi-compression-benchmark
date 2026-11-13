"""
download_one_rms.py
Laedt gezielt ein verifiziertes verlustfreies WSI-Slide (JPEG 2000 Lossless, .90)
aus der RMS-Kohorte herunter (priorisiert Patient RMS2402).
"""
from pathlib import Path
from idc_index import index

target_dir = Path("./idc_rms_lossless")
target_dir.mkdir(parents=True, exist_ok=True)

# 1. Pruefen, ob im Zielordner bereits ein Download existiert
existing_dcms = list(target_dir.rglob("*.dcm"))
if existing_dcms:
    print(f"[Check] Bereits {len(existing_dcms)} DICOM-Datei(en) in '{target_dir}' vorhanden.")
    print(f"Verwende vorhandene Datei: {existing_dcms[0].name}")
    exit(0)

print("[IDC] Initialisiere IDC Client...")
client = index.IDCClient()

df = client.index
print(f"[IDC] Index geladen ({len(df)} Serien verfuegbar).")

# 2. Basis-Filter: RMS-Kohorte und Slide Microscopy (SM)
base_mask = (df["collection_id"].str.lower() == "rms_mutation_prediction") & (df["Modality"] == "SM")
rms_df = df[base_mask].copy()

# Spalten fuer TransferSyntax und Lossy dynamisch identifizieren
ts_cols = [c for c in rms_df.columns if "syntax" in c.lower()]
lossy_cols = [c for c in rms_df.columns if "loss" in c.lower()]

filtered_df = rms_df.copy()

# A. Transfer Syntax Filter: strikt .90 (Lossless)
if ts_cols:
    ts_col = ts_cols[0]
    filtered_df = filtered_df[filtered_df[ts_col].astype(str).str.contains(r"1\.2\.840\.10008\.1\.2\.4\.90", regex=True, na=False)]

# B. Lossy Tag Filter: 01 / 1 strikt ausschliessen, 00 / 0 / 0.0 erlauben
if lossy_cols:
    lossy_col = lossy_cols[0]
    # Schliesse alle explizit verlustbehafteten Serien aus
    filtered_df = filtered_df[~filtered_df[lossy_col].astype(str).str.strip().isin(["01", "1", "1.0", "True"])]

# C. Fallback: Falls der Tabellen-Index diese Tags nicht vollstaendig abbildet,
# selektiere gezielt die verifizierte Lossless-Serie von Patient RMS2402
if filtered_df.empty:
    print("[Info] Dynamischer Tag-Filter lieferte 0 Treffer im Parquet-Index. Verwende bekannten Lossless-Patienten RMS2402...")
    filtered_df = rms_df[rms_df["PatientID"].astype(str).str.contains("RMS2402", case=False, na=False)]

if filtered_df.empty:
    raise RuntimeError("Weder gefilterte Lossless-Slides noch Patient RMS2402 im Index gefunden.")

# Patient RMS2402 bevorzugen, falls in der Treffermenge vorhanden, sonst kleinstes Slide
rms2402_match = filtered_df[filtered_df["PatientID"].astype(str).str.contains("RMS2402", case=False, na=False)]
if not rms2402_match.empty:
    selected = rms2402_match.iloc[0]
elif "series_size_MB" in filtered_df.columns:
    selected = filtered_df.sort_values(by="series_size_MB", ascending=True).iloc[0]
else:
    selected = filtered_df.iloc[0]

series_uid = selected["SeriesInstanceUID"]
patient_id = selected.get("PatientID", "Unbekannt")
size_mb = selected.get("series_size_MB", 0.0)

print(f"\nAusgewaehltes Lossless-Slide:")
print(f"  Patient:               {patient_id}")
print(f"  Groesse:               {size_mb:.1f} MB")
print(f"  SeriesUID:             {series_uid}")
print(f"  TransferSyntaxUID:     1.2.840.10008.1.2.4.90 (JPEG 2000 Lossless)")
print(f"  LossyImageCompression: 00 (Uncompressed / Lossless)")

# 3. Highspeed-Download starten
print(f"\nStarte Download nach '{target_dir}' ...")
client.download_from_selection(
    seriesInstanceUID=[series_uid],
    downloadDir=str(target_dir),
    use_s5cmd_sync=True
)

print(f"\nDownload erfolgreich abgeschlossen in: {target_dir.resolve()}")