"""Header survey: one DICOM per series -> per-series metadata table (laterality, geometry, scanner)."""
import io, sys, zipfile, csv
from multiprocessing import Pool
import pydicom

ZIP = "rsna-knee-abnormality-detection.zip"
_ZF = None
TAGS = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "Laterality", "PatientID",
        "SliceThickness", "SpacingBetweenSlices", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
        "Rows", "Columns", "NumberOfFrames", "SeriesDescription", "ScanningSequence", "SequenceVariant",
        "ScanOptions", "MRAcquisitionType", "ContrastBolusAgent", "DiffusionBValue", "BodyPartExamined",
        "SoftwareVersions", "ReceiveCoilName", "PatientPosition", "PhotometricInterpretation"]


def one(job):
    global _ZF
    if _ZF is None:
        _ZF = zipfile.ZipFile(ZIP)
    study, series, name, n = job
    try:
        d = pydicom.dcmread(io.BytesIO(_ZF.read(name)), stop_before_pixels=True)
    except Exception as e:
        return [study, series, n, "ERR:" + type(e).__name__] + [""] * (len(TAGS) + 3)
    row = [study, series, n, d.file_meta.TransferSyntaxUID]
    for t in TAGS:
        v = getattr(d, t, "")
        if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
            v = "|".join(str(x) for x in v)
        row.append(str(v).replace(",", ";").replace("\n", " "))
    for t in ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"]:
        v = getattr(d, t, None)
        row.append("|".join(f"{float(x):.4f}" for x in v) if v is not None else "")
    return row


def main(split="train"):
    zf = zipfile.ZipFile(ZIP)
    first, count = {}, {}
    for name in zf.namelist():
        if not (name.startswith(split + "_series/") and name.endswith(".dcm")):
            continue
        _, st, se, _f = name.split("/")
        count[(st, se)] = count.get((st, se), 0) + 1
        first.setdefault((st, se), name)
    jobs = [(st, se, first[(st, se)], count[(st, se)]) for (st, se) in first]
    out = f"artifacts/private/v2/survey-{split}.csv"
    with Pool(48) as pool, open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["StudyInstanceUID", "SeriesInstanceUID", "n_files", "TransferSyntax"] + TAGS + ["IPP", "IOP", "PixelSpacing"])
        for row in pool.imap_unordered(one, jobs, chunksize=64):
            w.writerow(row)
    print("wrote", out, len(jobs))


if __name__ == "__main__":
    main(*sys.argv[1:])
