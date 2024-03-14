import os
import random
from tqdm import tqdm

random.seed(0)

DATA_TYPE = list[tuple[str, str]]
 
def get_data(datapath: str,
             matching_table: dict[str, str | None]
             ) -> list[tuple[str, str]]:
    
    data: list[tuple[str, str]] = []    # list of [image_path, image_class]

    for label_key, label_value in tqdm(matching_table.items(), desc='find images'):
        if label_value is None:
            continue

        datapath_key = os.path.join(datapath, label_key)
        
        if not os.path.exists(datapath_key):
            raise FileNotFoundError(datapath_key)
        
        for image in os.listdir(datapath_key):
            if image.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                image_path = os.path.join(datapath_key, image)
                data.append((image_path, label_value))    

    random.shuffle(data)
    
    print(f'We have {len(data)} images in total')
    return data


def split_data(data: DATA_TYPE,
               train_rate: float=0.6,
               val_rate: float=0.1,
               test_rate: float=0.3
               ) -> tuple[DATA_TYPE, DATA_TYPE, DATA_TYPE]:
    """
    Split the data into training, validation, and testing sets according to the specified proportions.
    """
    # Check if the proportions sum to 1
    if train_rate + val_rate + test_rate != 1:
        raise ValueError(f'train + val + test rate must be equal to 1,',
                         f'but is equal to {train_rate + val_rate + test_rate}.')

    n = len(data)
    split_1 = int(n * train_rate)
    split_2 = int(n * (train_rate + val_rate))

    train_data = data[:split_1]
    val_data = data[split_1 : split_2]
    test_data = data[split_2:]

    return train_data, val_data, test_data


def save_data(data: DATA_TYPE, mode: str, datapath: str) -> None:
    """
    Save the data into a CSV file.
    The file is saved in the specified path and its name is {mode}_item.csv.
    """
    output = 'item,imagepath,label\n'
    for i, (path, label) in enumerate(data):
            output += f'{i},{path},{label}\n'
    
    dstpath = os.path.join(datapath, f'{mode}_item.csv')
    with open(file=dstpath, mode='w', encoding='utf8') as f:
        f.write(output)
        f.close()
    print(f'data save in {dstpath}')
        

if __name__ == '__main__':
    import sys
    from os.path import dirname as up
    sys.path.append(up(up(up(os.path.abspath(__file__)))))
    from data.process_real_data.get_matching_table import get_matching_table

    data = get_data(datapath=os.path.join('data', 'real_data', 'Photothe╠Çque'),
                    matching_table=get_matching_table())

    MODE = ['train', 'val', 'test']
    data_split = split_data(data)

    for i, mode in enumerate(MODE):
        save_data(data=data_split[i],
                  mode=MODE[i],
                  datapath=os.path.join('data', 'process_real_data'))
    
    save_data(data=data,
              mode='all',
              datapath=os.path.join('data', 'process_real_data'))