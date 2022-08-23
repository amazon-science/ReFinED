import sys

import torch
from dataset_reading.dataset_factory import Datasets
from dataset_reading.document_dataset import DocDataset
from doc_preprocessing.dataclasses import BatchedElementsTns
from evaluation.evaluation import eval_all
from model_components.refined_model import RefinedModel
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import AdamW, get_linear_schedule_with_warmup

from refined.processor import Refined

code_dir = "/data/tayoola/2021_code/TomED"

sys.path.append(code_dir)


def main():
    el = True

    data_dir = "/data/tayoola/2021_data"
    model_dir = "/data/tayoola/data/trained_models/21_march"
    datasets_dir = "/data/tayoola/datasets"
    debug = False
    device = "cuda:0"
    refined = Refined(
        model_dir=model_dir,
        data_dir=data_dir,
        debug=debug,
        requires_redirects_and_disambig=True,
        backward_coref=True,
        device=device,
        use_cpu=False,
    )
    refined.preprocessor.max_candidates = 30

    refined.model = RefinedModel.from_pretrained(
        "/data/tayoola/2021_code/TomED/best_model_2", refined.preprocessor
    )
    refined.model.to(refined.device)
    _ = refined.model.eval()

    datasets = Datasets(preprocessor=refined.preprocessor, datasets_path=datasets_dir)
    dataset = DocDataset(
        docs=datasets.get_aida_docs(split="train", include_gold_label=True),
        preprocessor=refined.preprocessor,
    )

    dataloader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=dataset.collate
    )

    _ = refined.model.train()
    model = refined.model

    ed_dropout = 0.05
    et_dropout = 0.10

    model.entity_disambiguation.dropout.p = ed_dropout
    model.entity_typing.dropout.p = et_dropout

    gradient_accumulation_steps = 4

    epochs = 30

    if el:
        optimizer = AdamW(
            [
                {"params": model.get_md_params(), "lr": 5e-5},
                {"params": model.get_et_params(), "lr": 5e-3},
                {"params": model.get_desc_params(), "lr": 5e-5},
                {"params": model.get_ed_params(), "lr": 5e-3},
                {"params": model.get_parameters_not_to_scale(), "lr": 5e-5},
            ],
            lr=5e-3,  # not used I think
            eps=1e-8,
        )
    else:
        optimizer = AdamW(
            [
                {"params": model.get_et_params(), "lr": 5e-3},
                {"params": model.get_desc_params(), "lr": 5e-5},
                {"params": model.get_ed_params(), "lr": 5e-3},
                {"params": model.get_parameters_not_to_scale(), "lr": 5e-5},
            ],
            lr=5e-3,  # not used I think
            eps=1e-8,
        )

    total_steps = len(dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=total_steps / gradient_accumulation_steps
    )

    model.train()
    for epoch_num in trange(epochs):
        for step, batch in enumerate(dataloader):
            batch = BatchedElementsTns(
                *[x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            )
            (
                md_loss,
                _,
                et_loss,
                _,
                ed_loss,
                ed_logits,
                _,
                _,
                description_loss,
                _,
                kg_loss,
                kg_scores,
                final_loss,
                final_score,
                _,
            ) = model(
                token_ids=batch.token_id_values,
                token_acc_sums=batch.token_acc_sum_values,
                entity_mask=batch.entity_mask_values,
                class_targets=batch.class_target_values,
                attention_mask=batch.attention_mask_values,
                token_type_ids=batch.token_type_values,
                candidate_entity_targets=batch.candidate_target_values,
                candidate_pem_values=batch.pem_values,
                candidate_classes=batch.candidate_class_values,
                candidate_pme_values=batch.pme_values,
                entity_index_mask_values=batch.entity_index_mask_values,
                spans=None,
                batch_elements=batch.batch_elements,
                cand_ids=batch.candidate_qcode_values,
                ner_labels=batch.ner_labels,
                cand_desc=batch.candidate_desc,
                candidate_features=batch.candidate_features,
                batch_elements_included=torch.arange(
                    batch.entity_index_mask_values.size(0)
                ).unsqueeze(-1),
            )

            loss = ed_loss + et_loss + (description_loss * 0.01)
            if el:
                loss += md_loss * 0.01
            if gradient_accumulation_steps >= 1:
                loss = loss / gradient_accumulation_steps

            if step % 50 == 0:
                print(f"ed_loss: {ed_loss}")
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        print(f"epoch_num: {epoch_num}")

        refined.preprocessor.max_candidates = 30
        eval_all(refined=refined, datasets_dir=datasets_dir, ed_threshold=0.25, el=el)
        refined.preprocessor.max_candidates = 30


if __name__ == "__main__":
    main()
