from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .data import load_cifar10, load_svhn


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
    """Deterministic evaluation: a single forward pass with dropout disabled."""
    # Force deterministic inference even if MC dropout was enabled earlier.
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout(False)
    model.eval()

    total_loss = 0
    total_brier = 0
    total_correct = 0
    total_samples = 0

    all_conf = []
    all_correct = []
    all_entropy = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)

        loss = F.cross_entropy(logits, labels)
        brier = brier_score_multiclass(logits, labels)

        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        correct = preds == labels

        # Predictive entropy of the single softmax (per-sample OOD score). Shannons formula.
        entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_brier += brier.item() * batch_size
        total_correct += correct.sum().item()
        total_samples += batch_size

        all_conf.append(conf.cpu())
        all_correct.append(correct.cpu())
        all_entropy.append(entropy.cpu())

    curve = reliability_curve(torch.cat(all_conf), torch.cat(all_correct), n_bins=n_bins)

    if reliability_path is not None:
        plot_reliability_diagram(curve, save_path=reliability_path)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "brier": total_brier / total_samples,
        "ece": curve["ece"],
        # Per-sample uncertainty for OOD detection (CIFAR-10 vs SVHN).
        "entropy": torch.cat(all_entropy),
    }


@torch.no_grad()
def evaluate_mc_dropout(
    model,
    dataloader,
    device: str | torch.device = "cuda",
    n_samples: int = 30,
    n_bins: int = 15,
    reliability_path=None,
):
    """MC dropout evaluation: `n_samples` (T) stochastic forward passes per input.

    Dropout layers are kept active (via ``model.enable_mc_dropout``) while
    BatchNorm uses its running statistics. The per-pass softmax probabilities are
    averaged into a predictive distribution; accuracy, loss, Brier and ECE are
    computed on that mean. Similar to ``evaluate_full``.

    Two uncertainty scores are also returned per sample (useful for OOD
    detection): the predictive entropy of the mean distribution and the summed
    across-pass class variance.
    """
    if not hasattr(model, "enable_mc_dropout"):
        raise TypeError(
            "Model does not support MC dropout (missing enable_mc_dropout)."
        )

    model.enable_mc_dropout(True)
    model.eval()  # BatchNorm in eval mode; dropout stays active via the switch

    total_loss = 0.0
    total_brier = 0.0
    total_correct = 0
    total_samples = 0

    all_conf = []
    all_correct = []
    all_entropy = []
    all_variance = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # (T, B, C) softmax probabilities across the stochastic passes.
        probs_t = torch.stack(
            [torch.softmax(model(images), dim=1) for _ in range(n_samples)],
            dim=0,
        )
        mean_probs = probs_t.mean(dim=0)  # (B, C) predictive distribution

        conf, preds = mean_probs.max(dim=1)
        correct = preds == labels

        log_mean = torch.log(mean_probs.clamp_min(1e-12))
        loss = F.nll_loss(log_mean, labels)  # cross-entropy on the mean probs

        one_hot = torch.zeros_like(mean_probs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        brier = torch.mean(torch.sum((mean_probs - one_hot) ** 2, dim=1))

        # Uncertainty scores (per sample).
        entropy = -torch.sum(mean_probs * log_mean, dim=1)  # predictive entropy
        variance = probs_t.var(dim=0).sum(dim=1)            # summed class variance

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_brier += brier.item() * batch_size
        total_correct += correct.sum().item()
        total_samples += batch_size

        all_conf.append(conf.cpu())
        all_correct.append(correct.cpu())
        all_entropy.append(entropy.cpu())
        all_variance.append(variance.cpu())

    curve = reliability_curve(torch.cat(all_conf), torch.cat(all_correct), n_bins=n_bins)

    if reliability_path is not None:
        plot_reliability_diagram(
            curve, save_path=reliability_path,
            title=f"MC dropout (T={n_samples})",
        )

    entropy = torch.cat(all_entropy)
    variance = torch.cat(all_variance)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "brier": total_brier / total_samples,
        "ece": curve["ece"],
        "n_samples": n_samples,
        "mean_entropy": entropy.mean().item(),
        "mean_variance": variance.mean().item(),
        # Per-sample arrays for downstream OOD detection (CIFAR-10 vs SVHN).
        "entropy": entropy,
        "variance": variance,
    }


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def ood_detection(id_scores, ood_scores):
    """Score far-OOD detection from per-sample uncertainty scores.

    OOD is the positive class and ``scores`` are uncertainties (higher => more
    likely OOD), so a good detector assigns higher scores to ``ood_scores`` than
    to ``id_scores``.

    Args:
        id_scores:  (N_id,) uncertainty on in-distribution samples (CIFAR-10).
        ood_scores: (N_ood,) uncertainty on out-of-distribution samples (SVHN).

    Returns a dict with the ROC curve arrays (``fpr``, ``tpr``) for plotting and
    the scalars ``auroc``, ``aupr`` and ``fpr95`` (FPR at 95% TPR).
    """
    id_scores = _to_numpy(id_scores)
    ood_scores = _to_numpy(ood_scores)

    scores = np.concatenate([ood_scores, id_scores])
    labels = np.concatenate([np.ones(len(ood_scores)), np.zeros(len(id_scores))])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    # FPR at the first threshold where TPR reaches 95%.
    fpr95 = float(fpr[np.searchsorted(tpr, 0.95)])

    return {"fpr": fpr, "tpr": tpr, "auroc": auroc, "aupr": aupr, "fpr95": fpr95}


def plot_ood_roc(curves, save_path=None, title: str = "OOD ROC (CIFAR-10 vs SVHN)"):
    """Overlay ROC curves for one or more OOD detectors.

    Args:
        curves: mapping ``{model_name: result}`` where each ``result`` is the
            output of ``ood_detection``. Add a key (e.g. an ensemble) to draw an
            extra curve -- no other change needed.

    If ``save_path`` is given the figure is written there and closed. Returns the
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    for name, r in curves.items():
        ax.plot(r["fpr"], r["tpr"], label=f"{name} (AUROC={r['auroc']:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig


def evaluate_ood(
    model,
    device,
    run_dir,
    n_samples: int = 30,
    batch_size: int = 256,
    seed: int = 42,
):
    """Far-OOD detection: CIFAR-10 (ID) vs SVHN (OOD).

    Scores the same trained model two ways - deterministic (dropout off,
    baseline) and MC-dropout (dropout on, ``n_samples`` passes) - using
    predictive entropy as the per-sample uncertainty. Prints AUROC/AUPR/FPR@95%
    per model and saves an overlaid ROC curve to ``run_dir``. Returns the
    ``{name: ood_detection result}`` dict.
    """
    id_loader = load_cifar10(batch_size=batch_size, seed=seed)[2]
    ood_loader = load_svhn(batch_size=batch_size, seed=seed)[2]

    # Baseline: deterministic single-pass predictive entropy.
    id_base = evaluate_full(model, id_loader, device)["entropy"]
    ood_base = evaluate_full(model, ood_loader, device)["entropy"]

    # MC-dropout: predictive entropy of the mean over T stochastic passes.
    id_mc = evaluate_mc_dropout(model, id_loader, device, n_samples=n_samples)["entropy"]
    ood_mc = evaluate_mc_dropout(model, ood_loader, device, n_samples=n_samples)["entropy"]

    curves = {
        "Baseline (entropy)": ood_detection(id_base, ood_base),
        f"MC-dropout T={n_samples} (entropy)": ood_detection(id_mc, ood_mc),
    }

    print("Far-OOD detection (CIFAR-10 ID vs SVHN OOD):")
    print(f"{'model':<32}{'AUROC':>8}{'AUPR':>8}{'FPR@95':>8}")
    for name, r in curves.items():
        print(f"{name:<32}{r['auroc']:>8.3f}{r['aupr']:>8.3f}{r['fpr95']:>8.3f}")

    roc_path = Path(run_dir) / "ood_roc.png"
    plot_ood_roc(curves, save_path=roc_path)
    print(f"ROC curve saved to {roc_path}")
    return curves
