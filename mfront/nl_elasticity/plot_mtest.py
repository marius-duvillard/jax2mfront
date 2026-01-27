import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


HERE = Path(__file__).resolve().parent
file_nn = HERE / "shear_test_nn.res"
file_gt = HERE / "shear_test_gt.res"


def load_data(filename):
    df = pd.read_csv(
        filename,
        sep="\s+",
        comment="#",
        header=None,
        names=["time", "EXY", "SXY"],
    )
    time = df["EXY"]
    stress = df["SXY"]
    return time, stress


time1, stress1 = load_data(file_nn)
time2, stress2 = load_data(file_gt)


plt.figure(figsize=(8, 6))
plt.plot(time1, stress1, label="NN", marker="x")
plt.plot(time2, stress2, label="GT", marker="x")
plt.xlabel("Shear strain $E_{xy}$")
plt.ylabel("Shear stress $S_{xy}$ [Pa]")
plt.title("Shear loading test")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison.png")
