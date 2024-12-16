import os
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

import wandb
from src.bert.bert_classifier import BertClassifier
from src.bert.data_prep import (
    get_annotated_answers,
    get_answer_df,
    prepare_test_dataset,
    prepare_train_dataset,
    split_llm_train_test,
    split_train_test_df,
)
from src.bert.metrics import define_metrics, get_compute_metrics_function
from src.bert.training_multilabel import get_label2str_dict
from src.paths import CODING_DIR

# Define the space of hyperparameters to search
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'weight_decay': hp.uniform('weight_decay', 0.0, 0.1),
    #'num_train_epochs': hp.quniform('num_train_epochs', 10, 15, 1),
}

def objective(params):
    training_args_dict = {
            "output_dir": "./training_results",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "num_train_epochs": 1,
            "weight_decay": 0.01,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            'logging_steps':300,
            "load_best_model_at_end": True,
            'fp16':True,
        }
    training_args_dict.update(params)
    print('new param combination')
    print('training_args_dict',training_args_dict)


    wandb.init(name=f"llm", project="hyperopt_bert")
    answer_df = get_annotated_answers()
    train_df, test_df = split_llm_train_test(answer_df, test_size=0.2)
    #train_df= train_df.sample(1000)
    #test_df= test_df.head(200)

    metrics = define_metrics()
    classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).upperclass_id.unique())) }    
    # Initialize model
    clf = BertClassifier(id2label=classid2trainid)

    
    splits=prepare_train_dataset(train_df,clf.tokenizer)

    label2str = get_label2str_dict(clf.label2id,class_mode='coarse')
    compute_metrics = get_compute_metrics_function(metrics,target_names=label2str.values())



    metrics = define_metrics()
    compute_metrics = get_compute_metrics_function(metrics, target_names=label2str.values())
    clf.train(splits, compute_metrics, report_to="wandb")


    eval_metrics= clf.evaluate_df(test_df,x_col= "text",y_col='labels',prefix='test')
    print(eval_metrics)
    return {'loss': eval_metrics['test_loss'], 'status': STATUS_OK}

# Run the optimization
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)

print(best)