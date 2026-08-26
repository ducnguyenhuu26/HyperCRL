# PB-CWM

This repository contains the preference-based continual world-model (PB-CWM)
implementation and its matched Hopper screen. The old standalone HyperCRL
repository code and legacy Robosuite dependency are intentionally not part of
this checkout.

The canonical campaign is one fixed Hopper schedule, five baseline adapters,
RADIUS-PbCWM, three seeds, and four bounded worker processes:

```bash
bash scripts/setup_vast_global.sh
python scripts/run_hopper_campaign.py --dry-run --max-parallel 4
python scripts/run_hopper_campaign.py --max-parallel 4
```

`setup_vast_global.sh` installs the PB-CWM dependencies into the machine's
system Python exactly once, verifies CUDA allocation and synchronization, and
records a stamp before allowing later runs to skip installation. It does not
create or activate a virtual environment and does not install the removed root
requirements file.

See [pbcwm/README.md](pbcwm/README.md) for the protocol, metrics, and method
details.
