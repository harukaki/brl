import numpy as np

keys, values = np.load("dds_results/train_000.npy")
np.save("100_hash.py", (keys[:100], values[:100]))
