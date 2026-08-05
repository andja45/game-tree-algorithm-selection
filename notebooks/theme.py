import seaborn as sns
import matplotlib.pyplot as plt

CLASS_ORDER = ["Both", "AB", "MCTS", "Neither"]
CLASS_COLORS = {"Both": "#4c72b0", "AB": "#dd8452", "MCTS": "#55a868", "Neither": "#c44e52"}
NEG_COLOR = "#c44e52"
POS_COLOR = "#4c72b0"
BOX_FILL = "#d0e4f7"

LINEWIDTH = 2.5
MARKERSIZE = 9
MARKER = "o"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
