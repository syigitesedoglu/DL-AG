import argparse
import json
import joblib
import sys
# Hata çözümü: 'datasets' yerine 'evaluate' kütüphanesinden metrik yüklüyoruz.
from evaluate import load as load_metric


OPTS = None

tasks = ['sst2', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte']

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}


def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for AdvGLUE.')
    parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
    parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')
    parser.add_argument('--out-file', '-o', metavar='eval.json',
                        help='Write accuracy metrics to file (default is stdout).')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def evaluate(adv_glue, predictions):
    results = {}
    scores = {}
    
    for task_name in predictions:
        metric = load_metric("glue", task_name if task_name != 'mnli-mm' else 'mnli')
        label_list = [data['label'] for data in adv_glue[task_name]]
        
        # HATA ÇÖZÜMÜ: predictions[task_name] zaten temiz bir liste olduğu için, 
        # onu doğrudan kullanıyoruz ve sözlükten listeye çevirme adımını atlıyoruz.
        pred_list = predictions[task_name]
        
        # Önceki kod bloğu (Hata veren kısım) kaldırılmıştır:
        # pred_dict = predictions[task_name]
        # pred_list = [pred_dict[str(i)] for i in range(len(label_list))] 
        
        
        # Tahmin listesi ve orijinal etiket listesi uzunluğunu kontrol edin
        assert len(label_list) == len(pred_list), f"Hata: {task_name} için etiket uzunluğu ({len(label_list)}) ile tahmin uzunluğu ({len(pred_list)}) eşleşmiyor."
        
        # Metrik hesaplama
        results[task_name] = metric.compute(predictions=pred_list, references=label_list)
        
        # ... geri kalan kod (task_scores, mnli hesaplamaları vb.) aynı kalır ...
        task_scores = list(results[task_name].values())
        task_score = sum(task_scores) / len(task_scores)
        scores[task_name] = task_score
        
    if 'mnli' in scores and 'mnli-mm' in scores:
        scores['mnli'] = (scores['mnli'] + scores['mnli-mm']) / 2
        del scores['mnli-mm']
        
    scores = list(scores.values())
    results['score'] = sum(scores) / len(scores)

    return results


def main():
    with open(OPTS.data_file) as f:
        dataset = json.load(f)
    with open(OPTS.pred_file) as f:
        preds = json.load(f)

    results = evaluate(dataset, preds)
    if OPTS.out_file:
        with open(OPTS.out_file, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    OPTS = parse_args()
    main()
