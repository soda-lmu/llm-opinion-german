import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from src.bert.custom_trainer import CustomTrainer
from src.bert.data_prep import (
    get_annotated_answers,
    get_answer_df,
    get_classid2trainid,
    get_label2str_dict,
    prepare_test_dataset,
    prepare_train_dataset,
    split_llm_train_test,
    split_train_test_df,
)
from src.bert.db_loss import DBloss
from src.bert.metrics import define_metrics, get_compute_metrics_function
from src.logger import setup_logger
from src.paths import CODING_DIR, LOGS_DIR, MODELS_DIR, TRAINING_OUT_DIR
from src.utils import load_lookup_data

logger = setup_logger("bert_logger")

class BertClassifier:
    def __init__(self, id2label=None, model_name=None, device='cuda',loss_type=None):


        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-german-cased",
            padding=True,
            truncation=True,
            max_length=512,
            problem_type="multi_label_classification",
        )
        if model_name: #was already trained and saved
            self.config = self.load_config(model_name)
            self.id2label = self.config['id2label']
            self.label2id = self.config['label2id']
            
            self.load_custom_config(model_name)

            self.num_labels = len(self.label2id)
            self.class_mode = 'coarse' if len(self.label2id) < 20 else 'fine'
            self.model_path = os.path.join(MODELS_DIR, model_name)
            self.model_name = model_name
            self.label2str = get_label2str_dict(self.label2id,class_mode=self.class_mode)
            self.load_model()
            self.evaluator_threshold= 0.5 
            self.evaluator = self.get_eval_trainer(0.5)

        else:#not trained
            self.id2label = id2label
            self.label2id = {v: k for k, v in id2label.items()}
            self.num_labels = len(id2label)
            self.class_mode = 'coarse' if len(self.label2id) < 20 else 'fine'
            self.label2str = get_label2str_dict(self.label2id,class_mode=self.class_mode)
            self.model_path=None #not trained yet
            self.load_model()
            self.custom_config = {'loss_type':loss_type, 'class_freq':None, 'train_num':None}



        self.device = device
        if self.device != "cpu":
            self.model.to(self.device)
        

    def load_model(self):
        if self.model_path is None:
            print('loading the not fine tuned model')
        model_source = os.path.join(MODELS_DIR, self.model_path) if self.model_path else "bert-base-german-cased"
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type="multi_label_classification"
        )

    def load_config(self,model_name):
        config_path = os.path.join(MODELS_DIR,model_name, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    
    def train(self, splits, compute_metrics, report_to=None, training_args_dict=None):
        from transformers import DataCollatorWithPadding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        labels = np.array(splits["train"]['labels'])
        class_freq = np.sum(labels, axis=0) + 1 
        class_freq= class_freq.tolist()
        train_num = len(labels)
        print(f"Class frequencies {class_freq}")
        print(f"Total training samples {train_num}")
        self.custom_config['class_freq'] = class_freq 
        self.custom_config['train_num'] = train_num
        if training_args_dict is None:
            training_args_dict = {
                "output_dir": TRAINING_OUT_DIR,
                "learning_rate": 2e-5,
                "per_device_train_batch_size": 32,
                "per_device_eval_batch_size": 32,
                "num_train_epochs": 15,
                "weight_decay": 0.01,
                "eval_strategy": "steps",
                "save_strategy": "steps",
                'logging_steps': 100,
                "load_best_model_at_end": True,
                'fp16': True,
            }
        if report_to:
            training_args_dict["report_to"] = report_to

        training_args = TrainingArguments(**training_args_dict)
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=splits["train"],
            eval_dataset=splits["validation"],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            compute_metrics=compute_metrics,
            loss_type=self.custom_config['loss_type'], 
            class_freq=class_freq,
            train_num=train_num
        )
        self.trainer.train()

        return self.trainer

    def train_old(self, splits, compute_metrics, report_to=None, training_args_dict=None):
        from transformers import DataCollatorWithPadding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        if training_args_dict is None:
            training_args_dict = {
                "output_dir": TRAINING_OUT_DIR,
                "learning_rate": 2e-5,
                "per_device_train_batch_size": 32,
                "per_device_eval_batch_size": 32,
                "num_train_epochs": 15,
                "weight_decay": 0.01,
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
                'logging_steps': 100,
                "load_best_model_at_end": True,
                'fp16': True,
            }
        if report_to:
            training_args_dict["report_to"] = report_to

        training_args = TrainingArguments(**training_args_dict)
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=splits["train"],
            eval_dataset=splits["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()

        return self.trainer

    def get_eval_trainer(self,threshold,save_classification_report=False):
        metrics = define_metrics()

        compute_metrics = get_compute_metrics_function(metrics,target_names=self.label2str.values(),threshold=threshold,save_classification_report=save_classification_report)
        training_args = TrainingArguments(
            disable_tqdm=True,
            output_dir='./results',   
            per_device_eval_batch_size=32,
            report_to=None
        )
        return CustomTrainer(
            model=self.model,
            args=training_args,                  
            compute_metrics=compute_metrics,
            loss_type=self.custom_config['loss_type'], 
            class_freq=self.custom_config['class_freq'],
            train_num=self.custom_config['train_num']
        )
    
    def evaluate_df(self,dataset_df, x_col, y_col, prefix='eval',threshold=0.5):
        self.evaluator = self.get_eval_trainer(threshold,save_classification_report=True)
        dataset=prepare_test_dataset(dataset_df,self.tokenizer,x_col,y_col)
        eval_results = self.evaluator.evaluate(dataset,metric_key_prefix=prefix)
        return eval_results
    
    def predict_text(self,text,threshold=0.5):
        if self.evaluator_threshold != threshold:
            self.evaluator = self.get_eval_trainer(threshold)
            self.evaluator_threshold = threshold
        tokenized_text = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        tokenized_ds = Dataset.from_dict({k: [v[0]] for k, v in tokenized_text.items()})
        eval_results = self.evaluator.predict(tokenized_ds)
        predictions =  (self.sigmoid(eval_results.predictions) > threshold).astype(int)[0].tolist()#.reshape(-1)
        probabilities = self.sigmoid(eval_results.predictions)[0].round(3).tolist()
        predicted_class_probs = [probabilities[i] for i in np.nonzero(predictions)[0] ]
        if len(predicted_class_probs) == 0:
            print('No class predicted,so taking the highest prob class as prediction')
            decision='argmax'
            predicted_class_idx = np.argmax(self.sigmoid(eval_results.predictions)[0].round(3))
            predicted_class_probs=probabilities[predicted_class_idx]
            label_names = [self.label2str[predicted_class_idx] ]
        else:
            decision='threshold'
            label_names = [self.label2str[i] for i in np.nonzero(predictions)[0] ]

        result={
            'pred_label_names':label_names,
            'pred_label_probs':predicted_class_probs,
            'probabilities':probabilities,
            'predictions':predictions,
            'threshold':threshold,
            'decision':decision
        }
        return result 
    
    def get_not_classified_files(self, folder_path,n=None):
        classification_key = f'classification_{self.class_mode}'
        json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
        not_classified_files = []
        for json_file in json_files:
            full_path = os.path.join(folder_path, json_file)
            with open(full_path, 'r') as file:
                try:
                    data = json.load(file)
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")
                #print(data)
                #print(f"Checking {full_path}")
                if classification_key in data and self.model_name in [elt['model_name'] for elt in data[classification_key]]:
                    print(f"Skipping {full_path} as it is already classified by {self.model_name}")
                    continue
                else:
                    not_classified_files.append(json_file)


        if n==None:
            pass
        else:
            not_classified_files = not_classified_files[:n]
        
        print(f"Classifying {len(not_classified_files)} files")
        return not_classified_files




    def classify_experiment_folder(self, folder_path, batch_size=16, save=True, threshold=0.5,n=None):

        json_files = self.get_not_classified_files(folder_path,n)

        if json_files == []:
            print("All files are already classified")
            return
        print(f"Classifying {len(json_files)} files")
        for i in tqdm(range(0, len(json_files), batch_size)):
            batch_files = json_files[i:i + batch_size]
            texts = []
            paths = []

            for json_file in batch_files:
                full_path = os.path.join(folder_path, json_file)
                
                with open(full_path, 'r') as file:
                    data = json.load(file)
                    texts.append(data['output'])
                    paths.append(full_path)

            
            # Batch prediction
            tokenized_texts = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            tokenized_ds = Dataset.from_dict({k: v for k, v in tokenized_texts.items()})
            eval_results = self.evaluator.predict(tokenized_ds)
            predictions = (self.sigmoid(eval_results.predictions) > threshold).astype(int).tolist()
            probabilities = self.sigmoid(eval_results.predictions).round(3).tolist()
            
            for idx, json_file in enumerate(batch_files):
                full_path = paths[idx]
                with open(full_path, 'r') as file:
                    data = json.load(file)
                
                result = {
                    'pred_label_names': [self.label2str[i] for i in np.nonzero(predictions[idx])[0]],
                    'pred_label_probs': [probabilities[idx][i] for i in np.nonzero(predictions[idx])[0]],
                    'probabilities': probabilities[idx],
                    'predictions': predictions[idx],
                    'threshold': threshold,
                    'decision': 'threshold' if any(predictions[idx]) else 'argmax'
                }

                if not any(predictions[idx]):
                    max_idx = np.argmax(probabilities[idx])
                    result['pred_label_names'] = [self.label2str[max_idx]]
                    result['pred_label_probs'] = [probabilities[idx][max_idx]]
                    result['decision'] = 'argmax'

                classification_key = f'classification_{self.class_mode}'
                if classification_key in data:
                    if isinstance(data[classification_key], list):
                        data[classification_key].append({'model_name': self.model_name, 'result': result})
                    else:
                        data[classification_key] = [{'model_name': self.model_name, 'result': result}]
                else:
                    data[classification_key] = [{'model_name': self.model_name, 'result': result}]

                if save:
                    with open(full_path, 'w') as file:
                        json.dump(data, file)
                    #print(f"Classification results saved to {full_path}")
                else:
                    results.append((json_file, result))
        
        if not save:
            return pd.DataFrame(results, columns=['file', 'classification_result'])


    def load_custom_config(self,model_name):
        
        config_path = os.path.join(MODELS_DIR,model_name, "custom_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.custom_config = config
            self.loss_type = self.custom_config['loss_type']
            self.class_freq = self.custom_config['class_freq']
            self.train_num = self.custom_config['train_num']
        else:
            self.custom_config = {
                'loss_type': 'default',
                'class_freq': None,
                'train_num': None
            }
            print(f"Custom config not found at {config_path},skipping")

    def save_model(self, model_path):
        self.model_path = os.path.join(MODELS_DIR, model_path)
        self.trainer.save_model(self.model_path)

        config_path = os.path.join(self.model_path, "custom_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.custom_config, f, ensure_ascii=False, indent=4)
        print(f"Model and the custom config saved to {self.model_path}")


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
