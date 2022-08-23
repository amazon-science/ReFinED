from dataclasses import dataclass

# add weak match for QA and EL


@dataclass
class Metrics:
    num_gold_spans: int
    tp: int
    fp: int
    fn: int
    gold_entity_in_cand: int = 0
    num_docs: int = 0

    def __add__(self, other):
        return Metrics(
            num_gold_spans=self.num_gold_spans + other.num_gold_spans,
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            gold_entity_in_cand=self.gold_entity_in_cand + other.gold_entity_in_cand,
            num_docs=self.num_docs + other.num_docs,
        )

    def get_summary(self):
        p = self.get_precision()
        r = self.get_recall()
        f1 = self.get_f1()
        accuracy = self.get_accuracy()
        gold_recall = self.get_gold_recall()
        result = f"f1: {f1:.4f}\naccuracy: {accuracy:.4f}\ngold_recall: {gold_recall:.4f}\np: {p:.4f}\nr: " \
                 f"{r:.4f}\nnum_gold_spans: {self.num_gold_spans}"
        return result

    def get_precision(self):
        return self.tp / (self.tp + self.fp + 1e-8 * 1.0)

    def get_recall(self):
        return self.tp / (self.tp + self.fn + 1e-8 * 1.0)

    def get_f1(self):
        p = self.get_precision()
        r = self.get_recall()
        return 2.0 * p * r / (p + r + 1e-8)

    def get_accuracy(self):
        return 1.0 * self.tp / (self.num_gold_spans + 1e-8)

    def get_gold_recall(self):
        return 1.0 * self.gold_entity_in_cand / (self.num_gold_spans + 1e-8)

    @classmethod
    def zeros(cls):
        return Metrics(num_gold_spans=0, tp=0, fp=0, fn=0)
