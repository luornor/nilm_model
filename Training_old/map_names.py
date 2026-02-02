import os
import re
import scipy.io as sio

ROOT = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Matlab_Data\Synthetic"

def clean(s: str) -> str:
    s = s.strip()
    # make filename/column-safe
    s = re.sub(r"[^\w\s\-\,\.]", "", s)
    s = s.replace(",", "")
    s = re.sub(r"\s+", "_", s)
    s = s[:60]  # keep short
    return s

code_to_descr = {}

for dirpath, _, filenames in os.walk(ROOT):
    for fn in filenames:
        if not fn.lower().endswith(".mat"):
            continue
        fp = os.path.join(dirpath, fn)
        try:
            m = sio.loadmat(fp, squeeze_me=True, struct_as_record=False)
        except Exception:
            continue

        # load_descr_short is like "1A0" or "2A0B0"
        short = m.get("load_descr_short", None)
        descr = m.get("load_descr", None)

        if short is None or descr is None:
            continue

        short = str(short)
        descr = str(descr)

        # Extract codes like A0, B0, etc from the short ID:
        # Examples:
        # "1A0" -> ["A0"]
        # "2A0B0" -> ["A0","B0"]
        codes = re.findall(r"[A-Z]0", short)

        for c in codes:
            # For single-load files, load_descr is exactly that load's official name.
            # We'll only set mapping from single-load acquisitions (starts with "1")
            if short.startswith("1") and len(codes) == 1:
                code_to_descr[c] = descr

print("Found mappings:", len(code_to_descr))
for k in sorted(code_to_descr.keys()):
    print(k, "->", code_to_descr[k])
