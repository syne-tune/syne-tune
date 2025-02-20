import os

import dill
from matplotlib import cm
from tqdm import tqdm

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from benchmarking.loop.baselines import (
    Methods,
)
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import get_metadata, load_experiments_df

from syne_tune.util import catchtime

rs_color = "black"
gp_color = "tab:orange"
gp_color2 = "tab:purple"
tpe_color = "red"
bore_color = "tab:green"
rea_color = "brown"
qr_color = "paleturquoise"
cqr_color = "tab:cyan"

fifo_style = "solid"
multifidelity_style = "dashed"
multifidelity_style2 = "dashdot"

show_seeds = False
marker_ours = "*"

cmap = cm.get_cmap("viridis")
method_styles = {}


@dataclass
class PlotArgs:
    xmin: float = None
    xmax: float = None
    ymin: float = None
    ymax: float = None


plot_range = {
    "fcnet-naval": PlotArgs(0, 3600, 0.0, 1e-3),
    "fcnet-parkinsons": PlotArgs(0, 3600, 0.005, 0.025),
    "fcnet-protein": PlotArgs(xmin=0, xmax=7200, ymin=0.22, ymax=0.3),
    "fcnet-slice": PlotArgs(0, 7200, 0.0, 0.0025),
    "nas201-ImageNet16-120": PlotArgs(2000, 36000, None, 0.8),
    "nas201-cifar10": PlotArgs(2000, 36000, 0.05, 0.1),
    "nas201-cifar100": PlotArgs(2000, 36000, 0.26, 0.35),
    "lcbench-bank-marketing": PlotArgs(2500, 36000, 82, 89),
    "lcbench-KDDCup09-appetency": PlotArgs(2500, 36000, 96, 100),
    "lcbench-christine": PlotArgs(500, 36000, 73.25, 75.5),
    "lcbench-albert": PlotArgs(500, 36000, 63, 66.5),
    "lcbench-airlines": PlotArgs(2500, 36000, 60, 65),
    "lcbench-Fashion-MNIST": PlotArgs(2500, 36000, 85, 90),
    "lcbench-covertype": PlotArgs(2500, 36000, 60, 80),
    "nas301-yahpo": PlotArgs(20000, 300000, 93.5, 97),
}

if __name__ == "__main__":
    x = np.linspace(0, 1)
    for i, (method, method_style) in enumerate(method_styles.items()):
        plt.plot(
            x,
            np.ones_like(x) * i,
            label=method,
            color=method_style.color,
            linestyle=method_style.linestyle,
            marker=method_style.marker,
        )
    plt.legend()
    plt.show()
