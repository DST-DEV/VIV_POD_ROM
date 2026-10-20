from pathlib import Path

import numpy as np

from viv_rom.pod.config import PODConfig
from viv_rom.spectral.config import SpectralConfig
from viv_rom.pod.pipeline import run as run_pod
from viv_rom.spectral.pipeline import run as run_spectral
from viv_rom.io.cfd.cfd_reader import HyReader
from viv_rom.io.results.export import save_result, load_result



# =============================================================================
# User configuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the *.1HY files
PITCH = 85
INCLINATION = 50
WIND_SPEED = 10  # free-stream wind speed [m/s]
FIELD = "combined"

# Load some data (to test)
reader = HyReader(DATA_DIR)
data = reader.retrieve_force_field(amplitude=1, pitch=PITCH,
                                   inclination=INCLINATION,
                                   wind_speed=WIND_SPEED,
                                   n_skip=1200, n_last=None)

# dy = np.diff(data.data.y)
# dy_unique = np.unique(np.round(dy, 3))
# print(f"N_unique: {len(dy_unique)}")

# =============================================================================
# Run Spectral Analysis
# =============================================================================

config = SpectralConfig(amplitude=1, pitch=PITCH, inclination=INCLINATION,
                        wind_speed=WIND_SPEED, data_dir=DATA_DIR,
                        field="both")
spectral_res = run_spectral(config=config)

# =============================================================================
# Run POD
# =============================================================================

config = PODConfig(amplitude=1, pitch=PITCH, inclination=INCLINATION,
                   wind_speed=WIND_SPEED, data_dir=DATA_DIR,
                   field="combined")
pod_res = run_pod(config=config)

exp_path = Path(__file__).parent / "copy.npz"
save_result(pod_res, config, exp_path,
            include_raw=False)

pod_res_loaded = load_result(exp_path)
print("")
