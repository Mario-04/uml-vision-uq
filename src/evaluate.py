import torch
import torch.nn.functional as F

def brier_score_multiclass(logits, labels):
    probs = F.softmax(logits, dim=1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)

    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))

@torch.no_grad()
def evaluate_full(model, dataloader, device="cuda"):
    model.eval()

    total_loss = 0
    total_brier = 0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)

        loss = F.cross_entropy(logits, labels)
        brier = brier_score_multiclass(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_brier += brier.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "brier": total_brier / total_samples,
    }