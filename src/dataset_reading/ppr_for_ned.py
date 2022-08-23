import json

from tqdm.auto import tqdm


def eval_model_ppr():
    refined.model.eval()
    total_ents = 0
    agree = 0
    with open("/Users/tayoola/Downloads/aida_test_ppr.jsonl", "r") as f:
        for line_idx, line in tqdm(enumerate(f)):
            line = json.loads(line)
            text = line["doc_text"]
            spans = []
            y_gold = set()
            for span in line["spans"]:
                start = span["start"]
                ln = span["end"] - span["start"]
                gold_qcode = refined.preprocessor.map_title_to_qcode(span["gold_titles"][0])
                candidates_qcodes = {
                    refined.preprocessor.map_title_to_qcode(c) for c in span["candidates"]
                }
                candidates_qcodes -= {None}
                if gold_qcode is None or gold_qcode in refined.preprocessor.disambiguation_qcodes:
                    continue
                spans.append(
                    Span(start=start, ln=ln, text=span["test"], pruned_candidates=candidates_qcodes)
                )
                y_gold.add(
                    (
                        line_idx,
                        text[span["start"] : span["start"] + span["length"]],
                        span["start"],
                        qcode,
                    )
                )

            predicted_spans = refined.process_text(text, spans)
            y_pred = {
                (
                    line_idx,
                    span.text,
                    span.start,
                    span.pred_entity_id[0].get("wikidata_qcode", "Q0"),
                )
                for span in predicted_spans
            }
            agree += len(y_pred & y_gold)

        # fix the pred Q0
        print(f"accuracy: {agree / total_ents:.3f}")
        refined.model.train()
        return agree / total_ents
