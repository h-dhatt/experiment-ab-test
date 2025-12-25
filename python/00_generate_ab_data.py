
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 20000

df = pd.DataFrame({
    "user_id": np.arange(1, n+1),
    "segment": rng.choice(["new","returning"], size=n, p=[0.6,0.4]),
    "device": rng.choice(["mobile","desktop"], size=n, p=[0.7,0.3]),
    "variant": rng.choice(["control","treatment"], size=n)
})

base = 0.085
seg = df["segment"].map({"new":0.9,"returning":1.2}).values
dev = df["device"].map({"mobile":0.95,"desktop":1.05}).values
lift = np.where(df["variant"].values=="treatment", 1.06, 1.00)

p = np.clip(base * seg * dev * lift, 0, 0.3)
df["converted"] = (rng.random(n) < p).astype(int)

df.to_csv("data/ab_events.csv", index=False)
print("Generated data/ab_events.csv")
