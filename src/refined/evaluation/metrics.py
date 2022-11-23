from dataclasses import dataclass, field

# add weak match for QA and EL
from typing import List, Any


@dataclass
class Metrics:
    el: bool  # flags whether the metrics are for entity linking (EL) or entity disambiguation (ED)
    num_gold_spans: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_md: int = 0
    fp_md: int = 0
    fn_md: int = 0
    gold_entity_in_cand: int = 0
    num_docs: int = 0
    example_errors: List[Any] = field(default_factory=list)
    example_errors_md: List[Any] = field(default_factory=list)

    def __add__(self, other: 'Metrics'):
        return Metrics(
            el=self.el,
            num_gold_spans=self.num_gold_spans + other.num_gold_spans,
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tp_md=self.tp_md + other.tp_md,
            fp_md=self.fp_md + other.fp_md,
            fn_md=self.fn_md + other.fn_md,
            gold_entity_in_cand=self.gold_entity_in_cand + other.gold_entity_in_cand,
            num_docs=self.num_docs + other.num_docs,
            example_errors=self.example_errors + other.example_errors,
            example_errors_md=self.example_errors_md + other.example_errors_md
        )

    def get_summary(self):
        p = self.get_precision()
        r = self.get_recall()
        f1 = self.get_f1()
        accuracy = self.get_accuracy()
        gold_recall = self.get_gold_recall()
        result = f"\n****************\n" \
                 f"************\n" \
                 f"f1: {f1:.4f}\naccuracy: {accuracy:.4f}\ngold_recall: {gold_recall:.4f}\np: {p:.4f}\nr: " \
                 f"{r:.4f}\nnum_gold_spans: {self.num_gold_spans}\n" \
                 f"************\n"
        if self.el:
            # MD results only make sense for when EL mode is enabled
            result += f"*******MD*****\n" \
                      f"MD_f1: {self.get_f1_md():.4f}, (p: {self.get_precision_md():.4f}," \
                      f" r: {self.get_recall_md():.4f})" \
                      f"\n*****************\n"
        return result

    def get_precision(self):
        return self.tp / (self.tp + self.fp + 1e-8 * 1.0)

    def get_recall(self):
        return self.tp / (self.tp + self.fn + 1e-8 * 1.0)

    def get_f1(self):
        p = self.get_precision()
        r = self.get_recall()
        return 2.0 * p * r / (p + r + 1e-8)

    def get_precision_md(self):
        return self.tp_md / (self.tp_md + self.fp_md + 1e-8 * 1.0)

    def get_recall_md(self):
        return self.tp_md / (self.tp_md + self.fn_md + 1e-8 * 1.0)

    def get_f1_md(self):
        # Note that MD results only make sense for when EL mode is enabled as gold `spans` and `md_spans` may differ.
        p = self.get_precision_md()
        r = self.get_recall_md()
        return 2.0 * p * r / (p + r + 1e-8)

    def get_accuracy(self):
        return 1.0 * self.tp / (self.num_gold_spans + 1e-8)

    def get_gold_recall(self):
        return 1.0 * self.gold_entity_in_cand / (self.num_gold_spans + 1e-8)

    @classmethod
    def zeros(cls, el: bool):
        return Metrics(num_gold_spans=0, tp=0, fp=0, fn=0, el=el)
