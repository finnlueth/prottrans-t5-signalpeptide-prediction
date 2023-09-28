from src import config
import requests

def download_file(url, file_path):
    """ Download file from url and save to file_path
    Args:
        url (_type_): _description_
        file_path (_type_): _description_
    """
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully to '{file_path}'.")
    else:
        print(f"Failed to download file to {file_path} .")


def download_all(path):
    """ Download all files to path
    Args:
        path (_type_): _description_
    """
    for name, url in urls.items():
        download_file(url=url, file_path=path + name)
        
        
def process(path):
    """ Process downloaded files
    Args:
        path (_type_): _description_
    """
    pass
