from transformers import BertTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from datasets import Dataset, load_dataset
import timeit, evaluate

model = BartForConditionalGeneration.from_pretrained("Ayaka/bart-base-cantonese")
tokenizer = BertTokenizer.from_pretrained("Ayaka/bart-base-cantonese")
dataset = load_dataset("raptorkwok/cantonese-traditional-chinese-parallel-corpus")
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
bert = evaluate.load("bertscore")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, max_length=200, label_pad_token_id=-100, return_tensors ="pt")

batch_size = 16

def dataset_loader(split):
    global dataset
    data = dataset[split]["translation"]
    yue = []
    zh = []
    for row in data:
        yue.append(row["yue"])
        zh.append(row["zh"])
    batch = Dataset.from_dict({"yue":yue, "zh":zh})
    return batch
        
def batch_tokenizer(batch):
    inputs = tokenizer(batch["yue"], padding="max_length", truncation=True, max_length=200)
    outputs = tokenizer(batch["zh"], padding="max_length", truncation=True, max_length=200)
    batch["input_ids"] = inputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    return batch

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = [s.replace(" ", "") for s in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = [s.replace(" ", "") for s in tokenizer.batch_decode(labels_ids, skip_special_tokens=True)]
    ref_str = [[row] for row in label_str]
    bleu_score = bleu.compute(predictions=pred_str, references=ref_str, tokenize="zh")["score"]
    chrf_score = chrf.compute(predictions=pred_str, references=ref_str, word_order=2)["score"]
    bert_score = bert.compute(predictions=pred_str, references=label_str, lang="zh")["f1"]
    for i in range(0, 10):
        print(f"pred :{pred_str[i]}\nlabel:{label_str[i]}\n\n")
    return {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "bert": sum(bert_score)/len(bert_score)
    }

def train(trainset, evalset):
    training_args = Seq2SeqTrainingArguments(
        output_dir="./ayaka9-0/",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        do_train=True,
        evaluation_strategy="steps",
        do_eval=True,
        logging_steps=500,
        save_steps=500,
        eval_steps=500,
        warmup_steps=0,
        overwrite_output_dir=True,
        save_total_limit=10,
        fp16=False, 
        #load_best_model_at_end=True,
        #metric_for_best_model="eval_chrf"
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #callbacks=EarlyStoppingCallback,
        #compute_metrics=compute_metrics,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=evalset
    )
    #trainer.remove_callback(EarlyStoppingCallback)
    #trainer.add_callback(EarlyStoppingCallback)
    trainer.train()

if __name__ == "__main__":
    trainset = dataset_loader("train").map(batch_tokenizer, batched=True, batch_size=batch_size)
    evalset = dataset_loader("validation").map(batch_tokenizer, batched=True, batch_size=batch_size)
    start = timeit.default_timer() #measuring train time
    train(trainset, evalset)
    end = timeit.default_timer()
    print(f"train time: {end-start}s")