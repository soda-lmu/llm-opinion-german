import argparse
import os
import time

import torch
#from dotenv import load_dotenv
from tqdm import tqdm

from src.data.process_data import process_open_ended, process_wave_data
from src.data.read_data import load_raw_survey_data
from src.experiment.experiment_utils import get_experiment_config
from src.experiment.HFTextGenerator import HFTextGenerator
from src.logger import setup_logger
from src.paths import (
    GENERATIONS_DIR,
    PROJECT_DIR,
    PROMPT_DIR,
)
from src.utils import format_prompt, get_experiment_log, save_experiment_log

logger = setup_logger("experiment_logger")


def experiment_setup(run_on_notebook=False, experiment_config_path=None):
    dotenv_path = os.path.join("experiment.env")
    #load_dotenv(dotenv_path)
    logger.info(os.environ.get("HF_HOME"))
    logger.info(os.environ.get("HUGGINGFACE_HUB_CACHE"))

    if run_on_notebook: 
        EXPERIMENT_CONFIG_PATH = experiment_config_path
    else:
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("config", type=str, help="The path to the experiment configuration file")
        args = parser.parse_args()
        logger.info(f"Using config: {args.config}")
        EXPERIMENT_CONFIG_PATH = args.config

    config = get_experiment_config(EXPERIMENT_CONFIG_PATH)
    logger.info(f"Using config: {config}")

    if config.get("experiment_results_folder") is None:
        config["experiment_results_folder"] = f"{config['model_name'].replace('/','-')}_{config['wave_number']}"

    config["prompt_fpath"] = os.path.join(PROMPT_DIR, config["prompt_fname"])
    config["experiment_dir"] = os.path.join(GENERATIONS_DIR,str(config["wave_number"]), config["experiment_results_folder"])

    return config


def run_experiment(
    wave_df_processed,
    model,
    experiment_dir,
    generation_config,
    remove_tag_fnc,
    batch_size=10,
):
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Saving logs to {experiment_dir}")
    logs_to_save = []
    saved_batch_count = 0
    for index, row in wave_df_processed.iterrows():
        completed_lfdn_ids = os.listdir(experiment_dir)
        filename = f"{row['lfdn']}.json"
        survey_wave = row.study  # wave id
        if filename in completed_lfdn_ids:
            logger.info(f'skipping {row["lfdn"]} as it already exists.')
        else:
            model_output = model.generate_response(
                row.formatted_prompt, generation_config, remove_tag_fnc
            )
            log = get_experiment_log(row, survey_wave, model_output)
            logs_to_save.append(log)

        if (index + 1) % batch_size == 0:
            for log in logs_to_save:
                save_experiment_log(log["user_id"], log, experiment_dir)
            saved_batch_count += 1
            logger.info(f"Saved {saved_batch_count} x 10 logs")
            logs_to_save = []  # Clear logs_to_save

    for log in logs_to_save:
        save_experiment_log(log["user_id"], log, experiment_dir)


def batch_iterator(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df[i : i + batch_size]


def run_experiment_batched(
    wave_df_processed,
    model,
    experiment_dir,
    generation_config,
    remove_tag_fnc,
    batch_size=16,
):
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Saving logs to {experiment_dir}")



    for batch in tqdm(
        batch_iterator(wave_df_processed, batch_size),
        total=len(wave_df_processed) // batch_size,
    ):
        start = time.time()
        formatted_prompts = batch.formatted_prompt.tolist()
        outputs_decoded_cleared = model.generate_batch_response(
            formatted_prompts, generation_config, remove_tag_fnc
        )
        batch.loc[:, "output"] = outputs_decoded_cleared
        dict_list = []
        for index, row in batch.iterrows():
            d = {
                "model": model.model.name,
                "prompt": row.formatted_prompt,
                "output": row["output"],
                "survey_wave": row.wave_number,
                "user_id": row["lfdn"],
            }
            d.update(generation_config)
            dict_list.append(d)
        for d in dict_list:
            save_experiment_log(d["user_id"], d, experiment_dir)

        logger.info(f"Time taken, batch: {time.time() - start}")


def main():
    config = experiment_setup()
    wave_number = config["wave_number"]
    model_name = config["model_name"]
    device = config["device"]
    quantization_config = config["quantization_config"]
    generation_config = config["generation_config"]
    prompt_fpath = config["prompt_fpath"]
    experiment_dir = config["experiment_dir"]
    batch_size = config["batch_size"]
    sample_size = config.get("sample_size")
    remove_tag_fnc = config.get("remove_tag_fnc")

    os.makedirs(experiment_dir, exist_ok=True)
    wave_df, wave_open_ended_df, df_coding_840s = load_raw_survey_data(wave_number)
    wave_open_ended_df = process_open_ended(
        wave_open_ended_df, df_coding_840s, wave_number
    )
    wave_df_processed = process_wave_data(wave_df, wave_open_ended_df, wave_number)
    wave_df_processed["formatted_prompt"] = wave_df_processed.apply(
        lambda row: format_prompt(prompt_fpath, row), axis=1
    )
    wave_df_processed['wave_number']=wave_number
    logger.info(f'wave_number: {wave_number}')
    if sample_size is not None:
        logger.warning(f"Sampling {sample_size} from the data")
        wave_df_processed = wave_df_processed.sample(sample_size, random_state=42)
    
    completed_lfdn_ids = os.listdir(experiment_dir)
    completed_lfdn_ids = [x.split(".")[0] for x in completed_lfdn_ids]
    wave_df_processed = wave_df_processed.loc[
        ~wave_df_processed["lfdn"].astype(str).isin(completed_lfdn_ids)
    ]
    logger.warning(f"Already {len(completed_lfdn_ids)} samples were generated.")
    logger.info(f"Remaining {len(wave_df_processed)} are being generated.")

    model = HFTextGenerator(model_name, device, quantization_config)
    logger.info("batch_size: {}".format(batch_size))
    if batch_size:
        logger.info("running batched")
        start = time.time()
        run_experiment_batched(
            wave_df_processed,
            model,
            experiment_dir,
            generation_config,
            remove_tag_fnc,
            batch_size,
        )
        logger.info(f"Time taken, run_experiment_batched: {time.time() - start}")

    else:
        # Run experiment
        start = time.time()
        run_experiment(
            wave_df_processed, model, experiment_dir, generation_config, remove_tag_fnc
        )
        logger.info(f"Time taken, run_experiment: {time.time() - start}")


if __name__ == "__main__":
    main()
