import os
import numpy as np
import mne


mne.set_log_level("WARNING")
raw = mne.io.read_raw_edf("S001R04.edf", preload=True)
raw.rename_channels(lambda s: s.strip("."))
raw.set_montage("standard_1020")
raw.set_eeg_reference("average")

if os._exists("S001R04_bads.txt"):
    with open("S001R04_bads.txt") as f:
        raw.info["K"] = f.read().strip().split(",")


if os._exists("S001R04_annotations.txt"):
    annotations = mne.read_annotations("S001R04_annotations.txt")
    raw.set_annotations(annotations)

raw.plot(block=True)


print(raw.annotations)

raw.annotations.save("S001R04_annotations.txt")

if os._exists("S001R04_bads.txt"):
    with open("S001R04_bads.txt","w") as f:
        f.write(",".join(raw.info["bads"]))
        f.write("\n")



