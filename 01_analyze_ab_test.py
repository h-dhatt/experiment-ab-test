
import pandas as pd
from math import sqrt
from scipy.stats import norm

df = pd.read_csv("data/ab_events.csv")

def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1/n1, x2/n2
    p_pool = (x1+x2)/(n1+n2)
    se = sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    z = (p2 - p1)/se
    pval = 2*(1 - norm.cdf(abs(z)))
    return p1, p2, z, pval

g = df.groupby("variant")["converted"].agg(["sum","count"])
x1, n1 = g.loc["control","sum"], g.loc["control","count"]
x2, n2 = g.loc["treatment","sum"], g.loc["treatment","count"]

p1, p2, z, pval = two_prop_ztest(x1,n1,x2,n2)
lift = (p2/p1 - 1)

print(f"Control:   {p1:.4f}")
print(f"Treatment: {p2:.4f}")
print(f"Lift: {lift*100:.2f}%")
print(f"P-value: {pval:.4g}")
