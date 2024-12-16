import argparse
import os 
import pandas as pd
from src.paths import CODING_DIR, LOGS_DIR
from src.bert.metrics import define_metrics, get_compute_metrics_function
from src.bert.data_prep import (
    get_annotated_answers, get_answer_df, get_classid2trainid,
    get_label2str_dict, prepare_test_dataset, prepare_train_dataset,
    split_llm_train_test, split_train_test_df
)
from bert_classifier import BertClassifier
# python src/bert/train.py --class_mode coarse --dataset mixed --model_name bert_classifier --loss_type dbloss
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_mode", type=str, help='Class mode: "coarse" or "fine"')
    parser.add_argument("--dataset", type=str, default='gles', help='llm or gles dataset')
    parser.add_argument('--model_name', type=str, default=None, help='The name of the model')
    parser.add_argument('--loss_type', type=str, default=None, help='The type of loss to use: default, focal, dbloss,resample ')
    args = parser.parse_args()
    class_mode = args.class_mode
    print('args', args)
    loss_type = 'default' if args.loss_type is None else args.loss_type
    # Load and split dataset as per the original script logic
    if args.dataset == 'gles':
        #parser.add_argument("--test_wave", type=int, default=20, help='Wave number to use for testing, rest is for training')
        i = 12#args.wave 
        answer_df = get_answer_df(class_mode)
        train_df, test_df = split_train_test_df(answer_df, i)
        train_df=train_df.sample(frac=0.1)
    elif args.dataset == 'llm':
        answer_df = get_annotated_answers()
        train_df, test_df = split_llm_train_test(answer_df, test_size=0.2)
    elif args.dataset == 'mixed':
        i = 12
        answer_df = get_answer_df(class_mode,drop_duplicates=True)
        train_df, test_df = split_train_test_df(answer_df, i)
        train_df=train_df.sample(frac=0.1)

        llm_answer_df = get_annotated_answers()
        llm_train_df, llm_test_df = split_llm_train_test(llm_answer_df, test_size=0.2)
        train_df_combined = pd.concat([train_df, llm_train_df])
        test_df_combined = pd.concat([test_df, llm_test_df])
    else:
        print('Dataset not implemented error')
        raise NotImplementedError
    print('Class mode:', class_mode)


    
    metrics = define_metrics()
    if class_mode == 'coarse':
        classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).upperclass_id.unique())) }  
    else:
        classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).subclassid.unique())) }  
    # Initialize model
    clf = BertClassifier(id2label=classid2trainid,loss_type=loss_type)

    
    splits=prepare_train_dataset(train_df_combined,clf.tokenizer)


    label2str = get_label2str_dict(clf.label2id,class_mode)
    compute_metrics = get_compute_metrics_function(metrics,target_names=label2str.values())



    metrics = define_metrics()
    compute_metrics = get_compute_metrics_function(metrics, target_names=label2str.values())
    clf.train(splits, compute_metrics)

    i = 12
    answer_df = get_answer_df(class_mode,drop_duplicates=True)
    train_df, test_df = split_train_test_df(answer_df, i)
    llm_answer_df = get_annotated_answers()
    llm_train_df, llm_test_df = split_llm_train_test(llm_answer_df, test_size=0.2)
    train_df_combined = pd.concat([train_df, llm_train_df])
    test_df_combined = pd.concat([test_df, llm_test_df])

    print('eval on test_df_combined',test_df_combined.shape)
    survey_eval_metrics= clf.evaluate_df(test_df,x_col= "text",y_col='labels',prefix='test')
    print(survey_eval_metrics)

    print('eval on llm_test_df',llm_test_df.shape)
    
    llm_eval_metrics= clf.evaluate_df(llm_test_df,x_col= "text",y_col='labels',prefix='test')
    print(llm_eval_metrics)

    if args.model_name is None:
        from datetime import datetime

        model_name = f"bert_{args.dataset}_{class_mode}_{args.loss_type}" + datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        model_name= args.model_name

    clf.save_model(model_name)

if __name__ == "__main__":
    main()
