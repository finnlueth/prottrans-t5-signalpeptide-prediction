from src import (
    config,
    utils
    )
import requests
import pandas as pd


def download_file(url: str, name: str, path: str):
    """ Download file from url and save to file_path
    Args:
        url (_type_): _description_
        file_path (_type_): _description_
    """
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(path + name, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully to '{path}'.")
    else:
        print(f"Failed to download file to {path} .")


def download_all(path: str, urls: dict):
    """ Download all files to path
    Args:
        path (_type_): _description_
    """
    for name, url in urls.items():
        download_file(url=url, path=path + name, name=name)


def parse_file(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='UTF-8') as file:
        file = file.read()

    items = []
    file = file.splitlines()
    for index, line in enumerate(file):
        if line.startswith('>'):
            line = line[1:].split('|')
            data = {
                'Uniprot_AC': line[0],
                'Kingdom': line[1],
                'Type': line[2],
                'Partition_No': line[3],
                'Sequence': file[index+1],
                'Label': file[index+2]
            }
            items.append(
                pd.Series(
                    data=data,
                    index=data.keys()
                    )
                )
    return pd.DataFrame(items)


def process(df_data: pd.DataFrame) -> pd.DataFrame:
    df_data.Sequence = df_data.Sequence.apply(lambda x: " ".join([*(x)]))
    df_data.Label = df_data.Label.apply(lambda x: [config.label_encoding[x] for x in [*(x)]])
    # df_process_data['Mask'] = df_process_data.Annotation.apply(lambda x: [0 if item == -1 else 1 for item in x])
    df_data['Split'] = df_data['Partition_No'].map(int)
    # df_data['Split'] = df_data.Partition_No.apply(lambda x: "test" if x in ['4'] else "train")
    df_data.drop(columns=['Partition_No', 'Uniprot_AC', 'Kingdom', 'Type'], inplace=True)
    return df_data
