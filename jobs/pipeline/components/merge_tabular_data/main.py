# Requierments
import logging as log
import os
import sys
import pandas as pd
import shutil
from pathlib import Path
import fire

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Auxiliar method to fetch files
def get_file(f):
    f = Path(f)
    if f.is_file():
        return f
    else:
        files = list(f.iterdir())
        if len(files) == 1:
            return files[0]
        else:
            log.error(f"More than one file was found in directory: {','.join(files)}.")
            return (f"More than one file was found in directory: {','.join(files)}.", 500)

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    output_path
):
    """
    Component that ingest a pd.DataFrame object with filepaths and returns a compressed rar file with decrypted data.

    Args:
        input_path (rar): Folder where a .rar extension file is placed with several tabular (.csv or .xlsx) data files.
        output_path (pd.DataFrame): Folder where a single file in .csv extension is stored, combining all inputs.
    """
    # Create output paths
    Path('./data').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Extract rar data
    os.system(f"unrar e {get_file(input_path)} ./data")
    # Load all files
    lst = []
    for filename in os.listdir('./data'):
        _, ext = os.path.splitext(filename)
        if ext=='.csv':
            lst.append(
                pd.read_csv(
                    os.path.join('data', filename),
                    dtype='str'
                )
            )
        elif ext=='.xlsx':
            lst.append(
                pd.read_excel(
                    os.path.join('data', filename),
                    dtype='str'
                )
            )
    # Generate output
    df = pd.concat(lst, axis=0)
    df.to_csv(os.path.join(output_path, 'output.csv'), index=False)
    # Cleanup resources
    shutil.rmtree('./data')

if __name__=="__main__":
    fire.Fire(main)