import os
import pickle

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer

from src.paths import CODING_DIR, MODELS_DIR, RAW_DATA_DIR,PROJECT_DIR,TRAIN_DATA_DIR,TEXT_GEN_DIR
from src.utils import load_lookup_data

def get_annotated_answers():
    df=pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'annotated_answers.csv'))
    df= df[df.label.apply(type)  == str]
    df = df[~df.label.str.contains("-1")]
    df.label = df.label.astype(str)
    
    # llm_refusal_df= pd.read_csv(os.path.join(RAW_DATA_DIR,'llm_refusal.csv'))
    # llm_refusal_df.label = llm_refusal_df.label.astype(str)

    # df = pd.concat([df,llm_refusal_df])
    df = df[df.label.str.len()!=0]
    df = df[df['label'].str.contains('\d')]
    df['labels_list'] =df.label.str.split(",").apply(lambda x: [int(elt) for elt in x ])
    classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).upperclass_id.unique())) }    
    #convert labels to binarized format
    df = df.rename(columns={'output':'text'}) #kpx_840s
    classes = list(classid2trainid.keys())
    mlb = MultiLabelBinarizer(classes=classes)
    sparse_matrix = mlb.fit_transform(df['labels_list']).astype(float).tolist()   
    df['labels'] = sparse_matrix
    df['wave'] = 'synthetic'

    return df
    
def split_llm_train_test(df,test_size=0.2,random_state=42):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def split_train_test_df(df,test_wave):
    """
    split df into train and test based on test_wave
    """
    train_df = df[(df.wave != test_wave) | (df.wave=='synthetic')]
    test_df = df[df.wave == test_wave]
    return train_df, test_df

# def get_answer_df3(class_mode,drop_duplicates=False):
#     """
#     option: class_mode: 'coarse' or 'fine' ; if you want to use clustered labels, num_classes = 15

#     get open ended answers and assigned labels (min 1, max 3 labels per answer) for all waves 10-21 
#     filter no response rows
#     drop duplicate text answers to avoid overfitting
    
#     output: df[text, labels, wave_id]
#     """
#     coding_840s_path = os.path.join(RAW_DATA_DIR,r"ZA7957_6838_v2.0.0.csv") # BERT classification for open ended answers 
#     df_coding_840s = pd.read_csv(coding_840s_path, sep=';', encoding='iso-8859-1')
#     answer_waves=[]
#     print('77')
#     for i in range(12,13):
#         regexstr=f"lfdn|kp{i}_840_c1|kp{i}_840_c2|kp{i}_840_c3|kp{i}_840s"
#         wave_i_df=df_coding_840s.filter(regex=regexstr, axis=1).dropna().rename(columns=lambda x: x.replace(f"kp{i}_840", "kpx_840")).reset_index(drop=True)
#         wave_i_df['wave']=i
#         answer_waves.append(wave_i_df)
#     wave10_21_answer_df=pd.concat(answer_waves, axis=0)
#     wave10_21_answer_df = wave10_21_answer_df[(wave10_21_answer_df.kpx_840_c1.ge(0)) | (wave10_21_answer_df.kpx_840_c1.isin([-99, -98]))]
#     if drop_duplicates: # for training, it makes sense to train on unique answers 
#         wave10_21_answer_df= wave10_21_answer_df.drop_duplicates(subset='kpx_840s')
#     wave10_21_answer_df.kpx_840_c2 = wave10_21_answer_df.kpx_840_c2.mask(wave10_21_answer_df.kpx_840_c2 < 0, 0).astype(int) 
#     wave10_21_answer_df.kpx_840_c3 = wave10_21_answer_df.kpx_840_c3.mask(wave10_21_answer_df.kpx_840_c3 < 0, 0).astype(int) 
#     wave10_21_answer_df.kpx_840_c1 = wave10_21_answer_df.kpx_840_c1.astype(int) 

#     if class_mode == 'fine':
#         labels_list = wave10_21_answer_df.filter(regex='kpx_840_c1|kpx_840_c2|kpx_840_c3').apply(lambda x: list(x[x != 0].astype(int)), axis=1)
#         classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).subclassid.unique())) }    

#     if class_mode == 'coarse':
#         df= pd.read_csv(os.path.join(CODING_DIR,'map.csv'))
#         lookup= dict(zip(df.subclassid,df.upperclass_id))
#         for col in wave10_21_answer_df.filter(like='kpx_840_c').columns:
#             wave10_21_answer_df[col] = wave10_21_answer_df[col].map(lookup)
        
#         llm_refusal_df= pd.read_csv(os.path.join(RAW_DATA_DIR,'llm_refusal.csv')).rename({'output':'kpx_840s','label':'kpx_840_c1'},axis=1)
#         llm_refusal_df['wave']= 'synthetic'
#         wave10_21_answer_df = pd.concat([wave10_21_answer_df,llm_refusal_df])
        
#         labels_list = wave10_21_answer_df.filter(regex='kpx_840_c1|kpx_840_c2|kpx_840_c3').apply(lambda x: list(x[x.notna()].astype(int)), axis=1)
#         classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).upperclass_id.unique())) }    


#     wave10_21_answer_df['labels_list']= labels_list
#     #convert labels to binarized format
#     wave10_21_answer_df = wave10_21_answer_df.rename(columns={'kpx_840s':'text'}) #kpx_840s
#     classes = list(classid2trainid.keys())
#     mlb = MultiLabelBinarizer(classes=classes)
#     sparse_matrix = mlb.fit_transform(labels_list).astype(float).tolist()   
#     wave10_21_answer_df['labels'] = sparse_matrix

#     mlb.classes_= np.array(classes)
#     mlb_path = os.path.join(MODELS_DIR, "mlb.pkl")
#     with open(mlb_path, 'wb') as f:
#         pickle.dump(mlb, f)
#     return wave10_21_answer_df


def get_answer_df(class_mode,drop_duplicates=False):
    """
    option: class_mode: 'coarse' or 'fine' ; if you want to use clustered labels, num_classes = 15

    get open ended answers and assigned labels (min 1, max 3 labels per answer) for all waves 10-21 
    filter no response rows
    drop duplicate text answers to avoid overfitting
    
    output: df[text, labels, wave_id]
    """
    coding_840s_path = os.path.join(RAW_DATA_DIR,r"ZA7957_6838_v2.0.0.csv") # BERT classification for open ended answers 
    df_coding_840s = pd.read_csv(coding_840s_path, sep=';', encoding='iso-8859-1')
    answer_waves=[]

    for i in range(10,22):
        regexstr=f"lfdn$|kp{i}_840_c1|kp{i}_840_c2|kp{i}_840_c3|kp{i}_840s"
        wave_i_df=df_coding_840s.filter(regex=regexstr, axis=1).dropna().rename(columns=lambda x: x.replace(f"kp{i}_840", "kpx_840")).reset_index(drop=True)
        wave_i_df['wave']=i
        answer_waves.append(wave_i_df)
    wave10_21_answer_df=pd.concat(answer_waves, axis=0)
    wave10_21_answer_df = wave10_21_answer_df[(wave10_21_answer_df.kpx_840_c1.ge(0)) | (wave10_21_answer_df.kpx_840_c1.isin([-99, -98,-72]))]
    if drop_duplicates: # for training, it makes sense to train on unique answers 
        wave10_21_answer_df= wave10_21_answer_df.drop_duplicates(subset='kpx_840s')
    wave10_21_answer_df.kpx_840_c2 = wave10_21_answer_df.kpx_840_c2.mask(wave10_21_answer_df.kpx_840_c2 < 0, 0).astype(int) 
    wave10_21_answer_df.kpx_840_c3 = wave10_21_answer_df.kpx_840_c3.mask(wave10_21_answer_df.kpx_840_c3 < 0, 0).astype(int) 
    wave10_21_answer_df.kpx_840_c1 = wave10_21_answer_df.kpx_840_c1.astype(int) 

    if class_mode == 'fine':
        labels_list = wave10_21_answer_df.filter(regex='kpx_840_c1|kpx_840_c2|kpx_840_c3').apply(lambda x: list(x[x != 0].astype(int)), axis=1)
        classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).subclassid.unique())) }    

    if class_mode == 'coarse':
        df= pd.read_csv(os.path.join(CODING_DIR,'map.csv'))
        lookup= dict(zip(df.subclassid,df.upperclass_id))
        for col in wave10_21_answer_df.filter(like='kpx_840_c').columns:
            wave10_21_answer_df[col] = wave10_21_answer_df[col].map(lookup)
        
        # llm_refusal_df= pd.read_csv(os.path.join(RAW_DATA_DIR,'llm_refusal.csv')).rename({'output':'kpx_840s','label':'kpx_840_c1'},axis=1)
        # llm_refusal_df['wave']= 'synthetic'
        # wave10_21_answer_df = pd.concat([wave10_21_answer_df,llm_refusal_df])
        
        labels_list = wave10_21_answer_df.filter(regex='kpx_840_c1|kpx_840_c2|kpx_840_c3').apply(lambda x: list(x[x.notna()].astype(int)), axis=1)
        classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).upperclass_id.unique())) }    


    wave10_21_answer_df['labels_list']= labels_list
    #convert labels to binarized format
    wave10_21_answer_df = wave10_21_answer_df.rename(columns={'kpx_840s':'text'}) #kpx_840s
    classes = list(classid2trainid.keys())
    mlb = MultiLabelBinarizer(classes=classes)
    sparse_matrix = mlb.fit_transform(labels_list).astype(float).tolist()   
    wave10_21_answer_df['labels'] = sparse_matrix
    
    mlb.classes_= np.array(classes)
    mlb_path = os.path.join(MODELS_DIR, "mlb.pkl")
    with open(mlb_path, 'wb') as f:
        pickle.dump(mlb, f)
    return wave10_21_answer_df


# def get_answer_df2(df,class_mode,drop_duplicates=False):
#     """
#     option: class_mode: 'coarse' or 'fine' ; if you want to use clustered labels, num_classes = 15

#     get open ended answers and assigned labels (min 1, max 3 labels per answer) for all waves 10-21 
#     filter no response rows
#     drop duplicate text answers to avoid overfitting
    
#     output: df[text, labels, wave_id]
#     """
#     wave10_21_answer_df=df
#     wave10_21_answer_df = wave10_21_answer_df[(wave10_21_answer_df.kpx_840_c1.ge(0)) | (wave10_21_answer_df.kpx_840_c1.isin([-99, -98]))]
#     if drop_duplicates: # for training, it makes sense to train on unique answers 
#         wave10_21_answer_df= wave10_21_answer_df.drop_duplicates(subset='kpx_840s')
#     wave10_21_answer_df.kpx_840_c2 = wave10_21_answer_df.kpx_840_c2.mask(wave10_21_answer_df.kpx_840_c2 < 0, 0).astype(int) 
#     wave10_21_answer_df.kpx_840_c3 = wave10_21_answer_df.kpx_840_c3.mask(wave10_21_answer_df.kpx_840_c3 < 0, 0).astype(int) 
#     wave10_21_answer_df.kpx_840_c1 = wave10_21_answer_df.kpx_840_c1.astype(int) 

#     if class_mode == 'fine':
#         labels_list = wave10_21_answer_df.filter(regex='kpx_840_c1|kpx_840_c2|kpx_840_c3').apply(lambda x: list(x[x != 0].astype(int)), axis=1)
#         classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).subclassid.unique())) }    

#     if class_mode == 'coarse':
#         df= pd.read_csv(os.path.join(CODING_DIR,'map.csv'))
#         lookup= dict(zip(df.subclassid,df.upperclass_id))
#         for col in wave10_21_answer_df.filter(like='kpx_840_c').columns:
#             wave10_21_answer_df[col] = wave10_21_answer_df[col].map(lookup)
        
#         llm_refusal_df= pd.read_csv(os.path.join(RAW_DATA_DIR,'llm_refusal.csv')).rename({'output':'kpx_840s','label':'kpx_840_c1'},axis=1)
#         llm_refusal_df['wave']= 'synthetic'
#         wave10_21_answer_df = pd.concat([wave10_21_answer_df,llm_refusal_df])
        
#         labels_list = wave10_21_answer_df.filter(regex='kpx_840_c1|kpx_840_c2|kpx_840_c3').apply(lambda x: list(x[x.notna()].astype(int)), axis=1)
#         classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).upperclass_id.unique())) }    


#     wave10_21_answer_df['labels_list']= labels_list
#     #convert labels to binarized format
#     wave10_21_answer_df = wave10_21_answer_df.rename(columns={'kpx_840s':'text'}) #kpx_840s
#     classes = list(classid2trainid.keys())
#     mlb = MultiLabelBinarizer(classes=classes)
#     sparse_matrix = mlb.fit_transform(labels_list).astype(float).tolist()   
#     wave10_21_answer_df['labels'] = sparse_matrix

#     mlb.classes_= np.array(classes)
#     mlb_path = os.path.join(MODELS_DIR, "mlb.pkl")
#     with open(mlb_path, 'wb') as f:
#         pickle.dump(mlb, f)
#     return wave10_21_answer_df


def get_classid2trainid(wave10_21_answer_df):
    c1_unique_cls= set(wave10_21_answer_df.kpx_840_c1.dropna().unique() )
    c2_unique_cls= set(wave10_21_answer_df.kpx_840_c2.dropna().unique() )
    c3_unique_cls= set(wave10_21_answer_df.kpx_840_c3.dropna().unique() )
    classes= sorted(c1_unique_cls.union(c2_unique_cls).union(c3_unique_cls))
    classid2trainid = {int(classname):idx  for idx, classname in enumerate(classes)}
    
    return classid2trainid



def tokenize_function(df,tokenizer):
    return tokenizer(df["text"], padding="max_length", truncation=True)

def prepare_train_dataset(df,tokenizer):

    dataset = Dataset.from_pandas(df[["text", "labels"]])
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x,tokenizer), batched=True)

    train_testvalid = tokenized_datasets.train_test_split(test_size=0.2)

    splits = DatasetDict(
        {
            "train": train_testvalid["train"],  # %80
            "validation": train_testvalid["test"],  # %20
        }
    )
    print('train splits',splits)
    return splits

def prepare_test_dataset(df, tokenizer, x_col, y_col):
    if x_col not in df.columns :
        raise ValueError(f"DataFrame must contain '{x_col}'")
    if y_col is None:
        df_renamed = df[[x_col]].rename(columns={x_col: "text"})
    else:
        df_renamed = df[[x_col, y_col]].rename(columns={x_col: "text", y_col: "labels"})

    dataset = Dataset.from_pandas(df_renamed)
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    return tokenized_datasets

def get_label2str_dict(label2id,class_mode):
        print('label2id',label2id)
        if class_mode=='fine':
            lookup_dict= load_lookup_data('first_most_imp_coding_list.json')
        elif class_mode=='coarse':
            df= pd.read_csv(os.path.join(CODING_DIR,'map.csv'))
            lookup_dict= dict(zip(df.upperclass_id,df.upperclass_name))
        print('lookup_dict',lookup_dict)

        label2str= {int(k): lookup_dict.get(v, v) for k, v in label2id.items()}
        print('label2str',label2str)
        return label2str



df= get_annotated_answers()
print('hello')