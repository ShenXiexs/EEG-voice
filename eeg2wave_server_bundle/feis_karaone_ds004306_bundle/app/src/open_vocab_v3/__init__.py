"""v3 content-first EEG-to-MFCC experiment.

This namespace deliberately does not import v0724/v0730 model code.  It may
consume their immutable source cache, but its checkpoints and artifacts have a
separate schema and output firewall.
"""

VERSION = "openvoice-eeg-v3-content-repair-v2"
LEGACY_VERSION = "openvoice-eeg-v3-encodec-clip-mfcc-v1"
