import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    d2_absolute_error_score,
)
from datetime import datetime
from pandas import read_json
import os


class ModelEvaluation:
    filepath = os.path.join('plots', datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f"))
    # filepath = f'E:\\Python\\test_doskutech\\plots\\{datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")}'

    def __init__(self) -> None:
        self.data = read_json(
            "https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json"
        )
        os.makedirs(self.filepath)

    def save_plots(self, filename, figure):
        filepath_plot = os.path.join(self.filepath, filename)
        figure.savefig(filepath_plot)

    def draw_bars(self):
        rows = self.data.sample(10)

        barWidth = 0.25
        fig, ax = plt.subplots(figsize=(12, 8))

        br1 = np.arange(len(rows["rb_corners"].tail(10)))
        br2 = [x + barWidth for x in br1]

        plt.bar(
            br1,
            rows["rb_corners"].tail(10),
            color="r",
            width=barWidth,
            edgecolor="grey",
            label="truth",
        )
        plt.bar(
            br2,
            rows["gt_corners"].tail(10),
            color="g",
            width=barWidth,
            edgecolor="grey",
            label="predicted",
        )

        plt.xlabel("Rooms", fontweight="bold", fontsize=15)
        plt.ylabel("Corners", fontweight="bold", fontsize=15)
        plt.xticks(
            [r + barWidth for r in range(len(rows["rb_corners"].tail(10)))],
            rows["name"].tail(10),
        )

        plt.legend()
        return fig

    def draw_subplots(self):
        x = np.arange(len(self.data["rb_corners"]))

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.plot(x, self.data["rb_corners"], linewidth=2, label="Truth")
        plt.plot(x, self.data["gt_corners"], linewidth=2.0, label="Predicted")

        plt.legend()

        return fig

    def draw_evaluations(self):
        evaluations_functions = [
            mean_squared_error,
            r2_score,
            mean_absolute_error,
            explained_variance_score,
            d2_absolute_error_score,
        ]

        evaluations = [
            evaluations_functions[i](self.data["rb_corners"], self.data["gt_corners"])
            for i in range(len(evaluations_functions))
        ]

        barWidth = 0.25
        figure, ax = plt.subplots(figsize=(12, 8))

        br1 = np.arange(len(evaluations))

        plt.bar(
            br1, evaluations, color="r", width=barWidth, edgecolor="grey", label="truth"
        )

        plt.xlabel("Metrics", fontweight="bold", fontsize=15)
        plt.ylabel("Values", fontweight="bold", fontsize=15)
        plt.xticks(
            [r for r in range(len(evaluations))],
            map(lambda x: x.__name__, evaluations_functions),
        )

        plt.legend()
        return figure

    def draw_plots(self):
        self.save_plots("sample-bars.png", self.draw_bars())
        self.save_plots("comparison-plots.png", self.draw_subplots())
        self.save_plots("evaluation-metrics.png", self.draw_evaluations())

        return self.filepath


print(ModelEvaluation().draw_plots())
