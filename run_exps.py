import subprocess
from tqdm import tqdm

jobs = [
    (2, "base.yaml"),
    (3, "cosine.yaml"),
    (4, "exp.yaml"),
]

processes = []

for gpu_id, config_file in tqdm(jobs):
    cmd = ["uv", "run", "main.py", f"--device-id={gpu_id}", f"--config={config_file}"]
    p = subprocess.Popen(cmd)
    processes.append(p)

for p in processes:
    p.wait()
print('Done')

