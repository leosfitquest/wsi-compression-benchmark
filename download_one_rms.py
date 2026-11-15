"""
download_one_rms.py
Laedt gezielt ein verifiziertes verlustfreies WSI-Slide (JPEG 2000 Lossless .90
oder Uncompressed .2.1) aus der RMS-Kohorte herunter (priorisiert Patient RMS2402)
und verifiziert anschliessend den DICOM-Header der Level-0 VOLUME-Instanz.
"""
from pathlib import Path
from idc_index import index
import pydicom

LOSSLESS_TS = {
    "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless
    "1.2.840.10008.1.2.1",     # Explicit VR Little Endian (Uncompressed)
}


def verify_level0_volume(series_dir: Path) -> tuple[Path, pydicom.Dataset]:
    """Prueft alle DICOM-Dateien eines Ordners, waehlt die Level-0 VOLUME-Instanz
    und stellt sicher, dass TransferSyntaxUID und LossyImageCompression verlustfrei sind.
    """
    vols = []
    for p in series_dir.rglob("*.dcm"):
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        img_type = [str(x).upper() for x in getattr(ds, "ImageType", [])]
        if "VOLUME" in img_type:
            cols = int(getattr(ds, "TotalPixelMatrixColumns", 0))
            rows = int(getattr(ds, "TotalPixelMatrixRows", 0))
            vols.append((cols * rows, p, ds))

    if not vols:
        raise RuntimeError(f"Keine VOLUME-Instanz in '{series_dir}' gefunden.")

    _, path, ds = max(vols, key=lambda t: t[0])
    ts = str(ds.file_meta.TransferSyntaxUID).strip()
    lossy = str(ds.get("LossyImageCompression", "")).strip()

    if ts not in LOSSLESS_TS or lossy != "00":
        raise RuntimeError(
            f"Sicherheitsabbruch bei {path.name}: TS={ts}, LossyImageCompression={lossy!r} -> NICHT verlustfrei!"
        )
    return path, ds


target_dir = Path("./idc_rms_lossless")
target_dir.mkdir(parents=True, exist_ok=True)

# 1. Pruefen, ob im Zielordner bereits ein Download existiert
existing_dcms = list(target_dir.rglob("*.dcm"))
if existing_dcms:
    print(f"[Check] Bereits {len(existing_dcms)} DICOM-Datei(en) in '{target_dir}' vorhanden.")
    vol_path, vol_ds = verify_level0_volume(target_dir)
    print(f"Verifizierte Level-0 VOLUME-Datei: {vol_path.name}")
    print(f"  PatientID:             {vol_ds.get('PatientID', 'Unbekannt')}")
    print(f"  Abmessungen:           {vol_ds.TotalPixelMatrixColumns} x {vol_ds.TotalPixelMatrixRows} px")
    print(f"  TransferSyntaxUID:     {vol_ds.file_meta.TransferSyntaxUID}")
    print(f"  LossyImageCompression: {vol_ds.get('LossyImageCompression', 'FEHLT')}")
    exit(0)

print("[IDC] Initialisiere IDC Client...")
client = index.IDCClient()

df = client.index
print(f"[IDC] Index geladen ({len(df)} Serien verfuegbar).")

# 2. Basis-Filter: RMS-Kohorte und Slide Microscopy (SM)
base_mask = (df["collection_id"].str.lower() == "rms_mutation_prediction") & (df["Modality"] == "SM")
rms_df = df[base_mask].copy()

# TransferSyntax-Filter auf Serienebene: .4.90 (J2K Lossless) oder reine .1.2.1 (Uncompressed) zulassen,
# aber verlustbehaftete .4.91 und .4.50 strikt ausschliessen.
ts_cols = [c for c in rms_df.columns if "syntax" in c.lower()]
filtered_df = rms_df.copy()

if ts_cols:
    ts_col = ts_cols[0]
    ts_series = filtered_df[ts_col].astype(str)
    has_lossless = ts_series.str.contains(r"1\.2\.840\.10008\.1\.2\.4\.90|1\.2\.840\.10008\.1\.2\.1", regex=True, na=False)
    has_lossy = ts_series.str.contains(r"1\.2\.840\.10008\.1\.2\.4\.91|1\.2\.840\.10008\.1\.2\.4\.50", regex=True, na=False)
    filtered_df = filtered_df[has_lossless & (~has_lossy)]

if filtered_df.empty:
    raise RuntimeError("Keine verlustfreien RMS-Serien im IDC-Index gefunden.")

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
index_ts = selected.get(ts_cols[0], "Unbekannt") if ts_cols else "Unbekannt"

print(f"\nAusgewaehlte Serie im IDC-Index:")
print(f"  Patient:            {patient_id}")
print(f"  Groesse:            {size_mb:.1f} MB")
print(f"  SeriesUID:          {series_uid}")
print(f"  Serien-Syntax-List: {index_ts}")

# 3. Highspeed-Download starten
print(f"\nStarte Download nach '{target_dir}' ...")
client.download_from_selection(
    seriesInstanceUID=[series_uid],
    downloadDir=str(target_dir),
    use_s5cmd_sync=True
)

# 4. Strikte Header-Verifikation auf der heruntergeladenen Level-0 VOLUME-Instanz
vol_path, vol_ds = verify_level0_volume(target_dir)
print(f"\nDownload & Header-Audit erfolgreich abgeschlossen:")
print(f"  Level-0 VOLUME Datei:  {vol_path.name}")
print(f"  TransferSyntaxUID:     {vol_ds.file_meta.TransferSyntaxUID}")
print(f"  LossyImageCompression: {vol_ds.get('LossyImageCompression')}")