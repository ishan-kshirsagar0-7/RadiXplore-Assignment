# Here I realized that a better approach to train the NER model would be to also include labels apart from just "Project" ones.
# That way, we can ensure the model doesn't just mislabel any proper noun as a project.

import time
import numpy as np
import evaluate
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoTokenizer
)
from datasets import DatasetDict
import torch

def train_multiclass_model(dataset_path, model_checkpoint, output_dir, label_names):
    print(f"Loading dataset from: {dataset_path}")
    tokenized_datasets = DatasetDict.load_from_disk(dataset_path)
    
    print(f"Loading tokenizer for '{model_checkpoint}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    print("Loading pre-trained model for multi-class token classification...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB2", mode="strict")
        
        per_class_results = {
            f"{key}_precision": value['precision'] for key, value in results.items() if isinstance(value, dict)
        }
        per_class_results.update({
            f"{key}_recall": value['recall'] for key, value in results.items() if isinstance(value, dict)
        })
        per_class_results.update({
            f"{key}_f1": value['f1'] for key, value in results.items() if isinstance(value, dict)
        })

        final_results = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        final_results.update(per_class_results)
        return final_results

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting multi-class model training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_duration = round(end_time - start_time)

    print("--- Training Summary ---")
    print(f"Training complete.")
    print(f"Total training time: {training_duration} seconds")

    final_model_path = f"{output_dir}/best-model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path) 
    print(f"Best multi-class model and tokenizer saved to: {final_model_path}")
    print("-------------------------")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("-" * 50)
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print("-" * 50)
    
    DATASET_DIR = 'data/ner_dataset_multiclass'
    MODEL_CHECKPOINT = "distilbert-base-cased"
    MODEL_OUTPUT_DIR = "models/ner-multiclass"

    entity_types = ["PROJECT", "ORGANIZATION", "LOCATION", "PROSPECT"]
    label_names = ["O"]
    for entity in entity_types:
        label_names.append(f"B-{entity}")
        label_names.append(f"I-{entity}")
    
    train_multiclass_model(
        dataset_path=DATASET_DIR,
        model_checkpoint=MODEL_CHECKPOINT,
        output_dir=MODEL_OUTPUT_DIR,
        label_names=label_names
    )