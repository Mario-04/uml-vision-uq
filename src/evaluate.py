import torch
import torch.nn.functional as F

def brier_score_multiclass(logits, labels):
    probs = F.softmax(logits, dim=1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)

    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))


def reliability_curve(confidences, accuracies, n_bins: int = 15):
    """Bin top-class predictions by confidence and return calibration stats.

    Args:
        confidences: (N,) tensor of predicted-class probabilities (max softmax).
        accuracies:  (N,) tensor (bool or 0/1), whether each prediction was correct.
        n_bins:      number of equal-width bins over [0, 1].

    Returns a dict with per-bin accuracy/confidence/count (length n_bins) and the
    scalar Expected Calibration Error (ECE).
    """
    confidences = confidences.detach().cpu().float()
    accuracies = accuracies.detach().cpu().float()

    edges = torch.linspace(0, 1, n_bins + 1)
    # Assign each sample to a bin; interior edges keep indices in [0, n_bins - 1].
    bin_idx = torch.bucketize(confidences, edges[1:-1].contiguous())

    bin_count = torch.zeros(n_bins)
    bin_conf = torch.zeros(n_bins)
    bin_acc = torch.zeros(n_bins)

    bin_count.index_add_(0, bin_idx, torch.ones_like(confidences))
    bin_conf.index_add_(0, bin_idx, confidences)
    bin_acc.index_add_(0, bin_idx, accuracies)

    nonempty = bin_count > 0
    bin_conf[nonempty] /= bin_count[nonempty]
    bin_acc[nonempty] /= bin_count[nonempty]

    weights = bin_count / bin_count.sum()
    ece = torch.sum(weights * (bin_acc - bin_conf).abs()).item()

    return {
        "bin_conf": bin_conf,
        "bin_acc": bin_acc,
        "bin_count": bin_count,
        "ece": ece,
    }


def plot_reliability_diagram(curve, save_path=None, title: str = "Reliability diagram"):
    """Plot a reliability diagram from the output of `reliability_curve`.

    If `save_path` is given the figure is written there and closed; otherwise it
    is shown. Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    bin_acc = curve["bin_acc"]
    bin_conf = curve["bin_conf"]
    nonempty = (curve["bin_count"] > 0).numpy()

    n_bins = len(bin_acc)
    edges = torch.linspace(0, 1, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).numpy()
    width = 1.0 / n_bins

    bin_acc = bin_acc.numpy()
    bin_conf = bin_conf.numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.bar(
        centers[nonempty], bin_acc[nonempty], width=width,
        edgecolor="black", color="tab:blue", alpha=0.85, label="Accuracy",
    )
    # Gap between confidence and accuracy (over-/under-confidence).
    gap = bin_conf[nonempty] - bin_acc[nonempty]
    ax.bar(
        centers[nonempty], gap, width=width, bottom=bin_acc[nonempty],
        edgecolor="black", color="tab:red", alpha=0.35, label="Gap",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} (ECE = {curve['ece']:.4f})")
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig


@torch.no_grad()
def evaluate_full(
    model,
    dataloader,
    device: str | torch.device = "cuda",
    n_bins: int = 15,
    reliability_path=None,
):
    model.eval()

    total_loss = 0
    total_brier = 0
    total_correct = 0
    total_samples = 0

    all_conf = []
    all_correct = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)

        loss = F.cross_entropy(logits, labels)
        brier = brier_score_multiclass(logits, labels)

        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        correct = preds == labels

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_brier += brier.item() * batch_size
        total_correct += correct.sum().item()
        total_samples += batch_size

        all_conf.append(conf.cpu())
        all_correct.append(correct.cpu())

    curve = reliability_curve(torch.cat(all_conf), torch.cat(all_correct), n_bins=n_bins)

    if reliability_path is not None:
        plot_reliability_diagram(curve, save_path=reliability_path)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "brier": total_brier / total_samples,
        "ece": curve["ece"],
    }
