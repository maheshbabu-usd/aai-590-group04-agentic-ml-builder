
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate

def train_text_classifier(dataset_name_or_path, model_name="bert-base-uncased", num_epochs=3):
    # Load dataset
    # Assumes dataset has 'text' and 'label' columns
    try:
        dataset = load_dataset(dataset_name_or_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Model
    num_labels = len(dataset['train'].unique('label'))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"] if "test" in tokenized_datasets else tokenized_datasets["train"], # Fallback
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("final_model")
    print("Model saved to final_model directory")

if __name__ == "__main__":
    # train_text_classifier("imdb")
    pass
