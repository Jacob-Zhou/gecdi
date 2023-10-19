from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_data(files, cut_off=0.35):
    data = []
    beta = 0.5
    for file_id, file in files.items():
        with open(file) as result:
            count_dict = {}
            for sent_id, line in enumerate(result):
                line = line.strip().split()
                seq_len = int(line[0])
                n_error = float(line[1])
                error_rate = float(line[2])
                error_rate_bin = cut_off if error_rate > cut_off else (
                    error_rate // 0.025) * 0.025
                error_rate_bin = f"{error_rate_bin:.2f}"
                tp = float(line[3])
                fp = float(line[4])
                fn = float(line[5])
                if error_rate_bin in count_dict:
                    count_dict[error_rate_bin] += Counter({
                        "tp": tp,
                        "fp": fp,
                        "fn": fn
                    })
                else:
                    count_dict[error_rate_bin] = Counter({
                        "tp": tp,
                        "fp": fp,
                        "fn": fn
                    })
            for k, v in count_dict.items():
                tp = v['tp']
                fp = v['fp']
                fn = v['fn']
                p = float(tp) / (tp + fp) if fp else 1.0
                r = float(tp) / (tp + fn) if fn else 1.0
                f = float((1 + (beta**2)) * p * r) / ((
                    (beta**2) * p) + r) if p + r else 0.0
                data.append([file_id, float(k), p, r, f])
    return pd.DataFrame(data, columns=["File", "ErrorRateBin", "P", "R", "F"])


def displot(data, name='matrix'):
    sns.set(style="white")

    _, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 30))

    # Draw the heatmap with the mask and correct aspect ratio
    # sns.histplot(data, x="ErrorRateBin", ax=ax[0])
    sns.lineplot(data, x="ErrorRateBin", y="P", ax=ax[0], hue="File")
    sns.lineplot(data, x="ErrorRateBin", y="R", ax=ax[1], hue="File")
    sns.lineplot(data, x="ErrorRateBin", y="F", ax=ax[2], hue="File")
    # g.set_axis_labels("Error rate", "F0.5")
    # plt.margins(0, 0)
    # plt.subplots_adjust(left=0.04, bottom=0., right=0.96, top=1.)
    plt.savefig(f'{name}.png')
    plt.close()


if __name__ == "__main__":
    seq_len, n_error, error_rate, p, r, f = [], [], [], [], [], []
    # bins
    data = load_data(
        {
            "Baseline":
            "exp/bart.with-weight-decay.0.seed-shuffle-data/pred/bea19.dev/baseline/bea19.dev.beam-12.pred.sent_ana",
            "Heuristic-0.2":
            "exp/bart.with-weight-decay.0.seed-shuffle-data/pred/bea19.dev/heuristic.prob.fix_bug/bea19.dev.penalty-0.2.beam-12.pred.sent_ana",
            "LM":
            "exp/bart.with-weight-decay.0.seed-shuffle-data/pred/bea19.dev/lm-gpt2.uncertainty-2.eps.debug.1/bea19.dev.penalty-0.9.beam-12.pred.sent_ana",
            "GED":
            "exp/bart.with-weight-decay.0.seed-shuffle-data/pred/bea19.dev/seq_wrapper.last.error_only/bea19.dev.penalty-0.5.beam-12.pred.sent_ana"
        }, 0.20)
    displot(data)