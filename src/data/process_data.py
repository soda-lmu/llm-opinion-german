import re
import pandas as pd
import numpy as np
import os
from src.data.read_data import get_wave_df_dict, load_lookup_data, load_raw_survey_data, read_stata_file
from src.paths import (
    CODING_DIR,
    PROJECT_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROMPT_DIR,
    GLES_DIR,
)
from sklearn.preprocessing import MultiLabelBinarizer


coding_list_dict = load_lookup_data(
    os.path.join(CODING_DIR, "first_most_imp_coding_list.json")
)
leaning_party_dict = load_lookup_data(
    os.path.join(CODING_DIR, "leaning_party_dict.json")
)
gender_dict = load_lookup_data(os.path.join(CODING_DIR, "gender_dict.json"))
ostwest_dict = load_lookup_data(os.path.join(CODING_DIR, "ostwest_dict.json"))
schulabschluss_dict = load_lookup_data(
    os.path.join(CODING_DIR, "schulabschluss_dict.json")
)
berufabschluss_lookup = pd.read_csv(
    os.path.join(CODING_DIR, "berufabschluss_lookup.csv")
)
berufabschluss_dict = load_lookup_data(
    os.path.join(CODING_DIR, "berufabschluss_dict.json")
)


edu_lookup = pd.read_csv(os.path.join(CODING_DIR, "education_lookup.csv"))
month_names_dict = load_lookup_data(os.path.join(CODING_DIR, "month_names_german.json"))


def process_open_ended(wave_open_ended_df, df_coding_840s, wave_number):
    """
    
    """


    i=wave_number
    regexstr=f"lfdn|kp{i}_840_c1|kp{i}_840_c2|kp{i}_840_c3|kp{i}_840s"
    wave_i_df=df_coding_840s.filter(regex=regexstr, axis=1).dropna().rename(columns=lambda x: x.replace(f"kp{i}_840", "kpx_840")).reset_index(drop=True)
    wave_i_df['wave']=i

    wave_coding_df=wave_i_df
    wave_coding_df = wave_coding_df[(wave_coding_df.kpx_840_c1.ge(0)) | (wave_coding_df.kpx_840_c1.isin([-99, -98]))]
    wave_coding_df.kpx_840_c2 = wave_coding_df.kpx_840_c2.mask(wave_coding_df.kpx_840_c2 < 0, 0).astype(int) 
    wave_coding_df.kpx_840_c3 = wave_coding_df.kpx_840_c3.mask(wave_coding_df.kpx_840_c3 < 0, 0).astype(int) 
    wave_coding_df.kpx_840_c1 = wave_coding_df.kpx_840_c1.astype(int) 

    df= pd.read_csv(os.path.join(CODING_DIR,'map.csv'))
    lookup= dict(zip(df.subclassid,df.upperclass_id))
    for col in wave_coding_df.filter(like='kpx_840_c').columns:
        wave_coding_df[col] = wave_coding_df[col].map(lookup)
    
    
    labels_list = wave_coding_df.filter(regex='kpx_840_c1|kpx_840_c2|kpx_840_c3').apply(lambda x: list(x[x.notna()].astype(int)), axis=1)
    classid2trainid = {int(classname):idx  for idx, classname in enumerate(sorted(pd.read_csv(os.path.join(CODING_DIR,'map.csv')).upperclass_id.unique())) }    

    wave_coding_df["highest_prob_label"]=wave_coding_df['kpx_840_c1']
    wave_coding_df['labels_list']= labels_list
    #convert labels to binarized format
    wave_coding_df = wave_coding_df.rename(columns={'kpx_840s':'text'}) #kpx_840s
    classes = list(classid2trainid.keys())
    mlb = MultiLabelBinarizer(classes=classes)
    sparse_matrix = mlb.fit_transform(labels_list).astype(float).tolist()   
    wave_coding_df['labels'] = sparse_matrix


    wave_open_ended_df_merged = pd.merge(
        wave_coding_df,
        wave_open_ended_df[["lfdn", f"kp{wave_number}_840s"]],
        left_on="lfdn_od", #user id in the original data
        right_on="lfdn",
        how="left",
    )
    wave_open_ended_df_merged= wave_open_ended_df_merged.dropna(subset=[f"kp{wave_number}_840s"])
    wave_open_ended_df_merged = wave_open_ended_df_merged.drop(['lfdn_x','lfdn_y','text'], axis=1)
    wave_open_ended_df_merged = wave_open_ended_df_merged.rename(
        {
         f"kp{wave_number}_840s": "text",
         'lfdn_od':'lfdn'}, axis=1
    )
    return wave_open_ended_df_merged



def process_wave_data(wave_df, wave_open_ended_df_merged, wave_number):
    """
    apply coding, convert datatypes for further analysis 
    """
    wave_df = pd.merge(
        wave_df, wave_open_ended_df_merged, on="lfdn"
    )
    df_2320_lookup= edu_lookup[['lfdn', f'code_2320']] # schulabschluss values for wave_number
    df_2330_lookup= edu_lookup[['lfdn', f'code_2330']] # berufabschluss values for wave_number
    wave_df = wave_df.merge(df_2320_lookup, on="lfdn", how="left")
    wave_df = wave_df.merge(df_2330_lookup, on="lfdn", how="left")

    wave_df["leaning_party"] = wave_df[f"kp{wave_number}_2090a"].apply(
        lambda x: leaning_party_dict[x] if x in leaning_party_dict else x
    )
    wave_df["gender"] = wave_df["kpx_2280"].map(gender_dict)
    wave_df["age"] = pd.to_datetime(wave_df.field_start.iloc[0]).year - wave_df[
        "kpx_2290s"
    ].str.extract(r"(\d+)").astype(float)
    wave_df["age_group"] = pd.cut(
        wave_df["age"],
        bins=[18, 30, 45, 60, float("inf")],
        labels=["18-29 Years", "30-44 Years", "45-59 Years", "60 Years and Older"],
    )

    wave_df = wave_df[wave_df["age"].notna()]  # drop if age is nan
    wave_df.age = wave_df.age.astype(int)

    wave_df.ostwest = wave_df.ostwest.map(ostwest_dict)
    wave_df = wave_df[
        wave_df["ostwest"].str.contains("-") == False
    ]  # filter ostwest is empty
    wave_df = wave_df[
        wave_df["leaning_party"].str.contains("-") == False
    ]  # filter leaning_party is empty
    wave_df = wave_df[wave_df["code_2330"].notna()]
    wave_df = wave_df[wave_df["code_2320"].notna()]

    wave_df["schulabschluss_clause"] = wave_df["code_2320"].map(schulabschluss_dict)
    wave_df["berufabschluss_clause"] = wave_df["code_2330"].map(berufabschluss_dict)
    return wave_df


if __name__ == "__main__":
    pass