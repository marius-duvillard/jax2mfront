import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
file_nn = HERE / "relax_nn.res"
file_gt = HERE / "relax_gt.res"


def load_data(filename):
    df = pd.read_csv(
        filename,
        sep="\s+",
        comment="#",
        header=None,
        names=["time", "EXY", "SXY", "p"],
    )
    time = df["time"]
    strain = df["EXY"]
    stress = df["SXY"]
    p = df["p"]
    return time, strain, stress, p


time1, strain1, stress1, p1 = load_data(file_nn)
time2, strain2, stress2, p2 = load_data(file_gt)

plt.figure(figsize=(8, 6))
plt.plot(time1, stress1, label="NN", marker="o")
plt.plot(time2, stress2, label="GT", marker="x")
plt.xlabel("time")
plt.ylabel("Shear stress $S_{xy}$ [Pa]")
plt.title("Shear loading test")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_relax.png")

plt.figure(figsize=(8, 6))
# plt.plot(time1, p1, label="NN", marker="x")
plt.plot(time2, p2, label="GT", marker="x")
plt.xlabel("time")
plt.ylabel("p internal variable")
plt.grid(True)
plt.tight_layout()
plt.savefig("p.png")

fig, ax1 = plt.subplots(figsize=(8, 6))

# ---- Axe gauche : contrainte ----
# ax1.plot(time1, stress1, label="NN $S_{xy}$", marker="o")
ax1.plot(time1, stress1, label="NN $S_{xy}$", marker="x")
ax1.set_xlabel("time")
ax1.set_ylabel("Shear stress $S_{xy}$ [Pa]")
ax1.grid(True)

# ---- Axe droit : variable interne p ----
ax2 = ax1.twinx()
ax2.plot(time1, p1, label="NN $p$", linestyle="--")
ax2.set_ylabel("Internal variable $p$")

# ---- Légende combinée ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Shear loading test: stress and internal variable")
plt.tight_layout()
plt.savefig("comparison_relax_sp.png", dpi=300)

plt.figure(figsize=(8, 6))
plt.plot(strain1, stress1, label="NN", marker="o")
plt.plot(strain2, stress2, label="GT", marker="x")
plt.xlabel("Shear strain $E_{xy}$")
plt.ylabel("Shear stress $S_{xy}$ [Pa]")
plt.title("Shear stress vs Shear strain")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_relax_stress_strain.png", dpi=300)

fig, ax1 = plt.subplots(figsize=(8, 6))
# ---- Axe gauche : variable interne NN ----
ax1.plot(time1, p1, label="NN $p$", color="tab:blue", marker="x")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Internal variable $p$ (NN)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True, linestyle="--", alpha=0.4)

# ---- Axe droit : variable interne GT ----
ax2 = ax1.twinx()
ax2.plot(time2, p2, label="GT $p$", color="tab:orange", marker="o")
ax2.set_ylabel("Internal variable $p$ (GT)", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

# ---- Légende combinée ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Internal variable $p$ for NN and GT")
plt.tight_layout()
plt.savefig("p_comparison_sp.png", dpi=300)
plt.show()
