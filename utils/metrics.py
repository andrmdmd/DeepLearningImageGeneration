from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute(self, preds, labels, losses):
        preds = preds.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall = recall_score(labels, preds, average="macro", zero_division=0)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(labels, preds, labels=range(self.num_classes))

        acc = (preds == labels).mean()
        avg_loss = sum(losses) / len(losses)

        return {
            "accuracy": acc,
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conf_matrix": conf_matrix,
        }
