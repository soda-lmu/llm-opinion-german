import re
import pandas as pd
import numpy as np
import os
from data.read_data import  read_stata_file
from paths import (
    CODING_DIR,
    GLES_DIR,
)




def process_2320_data(dfs_dict):
    """
    assigns education level to each respondent based on the education level variables in the dataframes
    """
    a2_2320 = dfs_dict["a2"].filter(regex="lfdn$|kp(.*?)_2320", axis=1)
    w21_2320 = dfs_dict["21"].filter(regex="lfdn$|kp(.*?)_2320", axis=1)
    w1to9_2320 = dfs_dict["1to9"].filter(regex="lfdn$|kp(.*?)_2320", axis=1)
    all_2320 = pd.merge(w1to9_2320, w21_2320, on="lfdn", how="outer").merge(
        a2_2320, on="lfdn", how="outer"
    )

    for col in all_2320.filter(regex="kp(.*?)_2320", axis=1).columns:
        all_2320.loc[all_2320[col] < 0, col] = np.nan

    all_2320.sort_values(
        by=["kp1_2320", "kpa1_2320", "kp9_2320", "kp21_2320", "kpa2_2320"]
    )

    all_2320["code_2320"] = (
        all_2320["kp1_2320"]
        .combine_first(all_2320["kpa1_2320"])
        .combine_first(all_2320["kp9_2320"])
        .combine_first(all_2320["kpa2_2320"])
        .combine_first(all_2320["kp21_2320"])
    )

    all_2320["source_combined"] = np.nan
    all_2320.loc[(all_2320["kp1_2320"].notna()), "source_combined"] = "kp1_2320"
    all_2320.loc[(all_2320["kpa1_2320"].notna()), "source_combined"] = "kpa1_2320"
    all_2320.loc[(all_2320["kp9_2320"].notna()), "source_combined"] = "kp9_2320"
    all_2320.loc[(all_2320["kpa2_2320"].notna()), "source_combined"] = "kpa2_2320"
    all_2320.loc[(all_2320["kp21_2320"].notna()), "source_combined"] = "kp21_2320"

    return all_2320


def process_2330_data(dfs_dict):
    """
    assigns education level (berufabsc) to each respondent based on the education level variables in the dataframes
    """
    dfa2 = dfs_dict["a2"]
    df1to9 = dfs_dict["1to9"]

    w1to9_2330 = df1to9.filter(regex="lfdn$|kp(.*?)_2330", axis=1).astype(int)
    wa2_2330 = dfa2.filter(regex="lfdn$|kp(.*?)_2330", axis=1).astype(int)
    all_2330 = pd.merge(w1to9_2330, wa2_2330, on="lfdn", how="outer")

    for col in all_2330.filter(regex="kp(.*?)_2330", axis=1).columns:
        all_2330.loc[all_2330[col] < 0, col] = np.nan

    all_2330["code_2330"] = (
        all_2330["kp1_2330"]
        .combine_first(all_2330["kpa1_2330"])
        .combine_first(all_2330["kpa2_2330"])
    )

    all_2330["source_combined"] = "source"
    all_2330.loc[all_2330["kp1_2330"].notna(), "source_combined"] = "kp1_2330"
    all_2330.loc[all_2330["kpa1_2330"].notna(), "source_combined"] = "kpa1_2330"
    all_2330.loc[all_2330["kpa2_2330"].notna(), "source_combined"] = "kpa2_2330"

    return all_2330


def get_education_lookup(dfs_dict,save=False):
    """
    returns a lookup table with the schulabschluss and berufabschluss of each respondent
    """

    all_2320 = process_2320_data(dfs_dict)
    all_2330 = process_2330_data(dfs_dict)
    edu_lookup_2320_2330 = pd.merge(
        all_2320[["lfdn", "code_2320"]], all_2330[["lfdn", "code_2330"]], on="lfdn"
    )

    if save == True:
        edu_lookup_2320_2330.to_csv(
            os.path.join(CODING_DIR, "education_lookup.csv"), index=False
        )
    return edu_lookup_2320_2330


def get_2320_2330_lookups(dfs_dict): 
    cols_2330 = ['kp1_2330',
     'kp2_2330',
     'kp3_2330',
     'kp4_2330',
     'kpa1_2330',
     'kp5_2330',
     'kp6_2330',
     'kp7_2330',
     'kp8_2330',
     'kp9_2330',
     'kp10_2330',
     'kp11_2330',
     'kp12_2330',
     'kp13_2330',
     'kp14_2330',
     'kpa2_2330',
     'kp15_2330',
     'kp16_2330',
     'kp17_2330',
     'kp18_2330',
     'kp19_2330',
     'kp20_2330',
     'kp21_2330']

    cols_2320 = ['kp1_2320',
     'kp2_2320',
     'kp3_2320',
     'kp4_2320',
     'kpa1_2320', 
     'kp5_2320',
     'kp6_2320',
     'kp7_2320',
     'kp8_2320',
     'kp9_2320',
     'kp10_2320',
     'kp11_2320',
     'kp12_2320',
     'kp13_2320',
     'kp14_2320',
     'kpa2_2320',
     'kp15_2320',
     'kp16_2320',
     'kp17_2320',
     'kp18_2320',
     'kp19_2320',
     'kp20_2320',
     'kp21_2320']

    lfdn_list= list(set([lfdn_value for df in dfs_dict.values() for lfdn_value in df['lfdn'].values]))
    df_2320 =pd.DataFrame(index=lfdn_list,columns=cols_2320)
    df_2330 =pd.DataFrame(index=lfdn_list,columns=cols_2330)
    import numpy as np

    for key in dfs_dict.keys():
        cols_2320_key = dfs_dict[key].filter(regex='2320', axis=1).columns
        # Loop through each column and assign values with condition
        for col in cols_2320_key:
            print(key, col)
            values = np.where(dfs_dict[key][col].values < 0, np.nan, dfs_dict[key][col].values)
            df_2320.loc[dfs_dict[key]['lfdn'], col] = values

        cols_2330_key = dfs_dict[key].filter(regex='2330', axis=1).columns
        # Loop through each column and assign values with condition
        for col in cols_2330_key:
            print(key, col)
            values = np.where(dfs_dict[key][col].values < 0, np.nan, dfs_dict[key][col].values)
            df_2330.loc[dfs_dict[key]['lfdn'], col] = values
    df_2320 = df_2320.ffill(axis=1).infer_objects(copy=None)
    df_2330 = df_2330.ffill(axis=1).infer_objects(copy=None)
    df_2320.insert(0, 'lfdn', df_2320.index)
    df_2330.insert(0, 'lfdn', df_2330.index)
    return df_2320,df_2330


if __name__ == "__main__":
    wave_ids = [match.group(1) for fname in os.listdir(GLES_DIR) if fname.endswith('.dta') and (match := re.search(r'_w([^_]+)_s', fname))]
    print(wave_ids)
    wave_df_dict={}
    for wave_id in wave_ids:
        wave_df_dict[wave_id]=read_stata_file(os.path.join(GLES_DIR, f"ZA6838_w{wave_id}_sA_v6-0-0.dta"))
    edu_df=get_education_lookup(wave_df_dict)