import os
import sys
import yaml
import copy
import random
from typing import Any
from itertools import product
from easydict import EasyDict
from dataclasses import dataclass
from os.path import dirname as up

sys.path.append(up(up(os.path.abspath(__file__))))

from config.utils import number_folder
from config.compare_experiments import compare_experiments


class Search:
    def __init__(self,
                 config_yaml_file: str = os.path.join('config', 'config.yaml'),
                 search_yaml_file: str = os.path.join('config', 'search.yaml'),
                 logspath: str = 'logs',
                 name: str = 'random_search',
                 ) -> None:
        """
        Initialize the Search class.

        Args:
            config_yaml_file (str): The path to the config YAML file. Defaults to 'config/config.yaml'.
            search_yaml_file (str): The path to the search YAML file. Defaults to 'config/search.yaml'.
            logspath (str): The path to the logs directory. Defaults to 'logs'.
            name (str): The name of the search. Defaults to 'random_search'.

        Raise:
            FileNotFoundError: If the config YAML file or the search YAML file does not exist.
            ValueError: If the items list is empty.
        """
        
        if not os.path.exists(config_yaml_file):
            raise FileNotFoundError(config_yaml_file)
        if not os.path.exists(search_yaml_file):
            raise FileNotFoundError(search_yaml_file)
        
        # Get yaml file
        self.config = self.__loadyaml(config_yaml_file)
        search = self.__loadyaml(search_yaml_file)
        self.logspath = logspath
        self.name = name

        # Get all items
        self.items: list[Item] = []
        self.__get_all_item(search)
        if self.items == []:
            raise ValueError(f'Empty items')

        print('\n---- ITEMS POSSIBILITIES ----')
        for item in self.items:
            print(item)

        # Get the list of the possibilities
        self.len: int = 1
        self.possibilities: list[int] = []
        for item in self.items:
            self.possibilities.append(len(item))
            self.len *= len(item)
        print(f'{self.possibilities = }')
        print(f'{self.len} possibilities')
        self.all_possibilities = self.__get_all_possibilities()

        # Create folder to logs the experiments
        folder_name = number_folder(path=self.logspath, name=f'{self.name}_')
        self.folder_name = os.path.join(self.logspath, folder_name)
        os.mkdir(self.folder_name)
        print(f'create folder: {self.folder_name}')

        # Get the suffle the index corresponding to the possibilities
        self.indexes: list[int] = list(range(0, len(self)))
        random.shuffle(self.indexes)
        self.index: int = -1

    def get_new_config(self) -> EasyDict:
        """
        Get a new configuration based on the current index value.

        Returns:
            EasyDict: A new configuration based on the current index value.
        """
        self.__update_index()
        config = copy.copy(self.config)
        possibility: list[int] = self.all_possibilities[self.indexes[self.index]]

        for item_number, item in enumerate(self.items):
            item.change_config(config, index_value=possibility[item_number])

        return EasyDict(config)
    
    def get_directory(self) -> str:
        """
        Get the logs path where the experiments will be saved.
        """
        return self.folder_name
    
    def compare_experiments(self) -> None:
        """
        Compare the experiments based on the specified hyperparameters.

        Raise:
            ValueError: If the end of the traversal is reached.
        """

        hyperparameters: dict[str, list[str]] = {}
        for item in self.items:
            hyperparameters[item.keys[-1]] = item.keys

        compare_experiments(csv_output='compare',
                            logs_path=self.folder_name,
                            hyperparameters=hyperparameters,
                            compare_on='val')
    
    def __update_index(self) -> None:
        """
        Update the index by incrementing it by 1.

        Raise:
            ValueError: If the index reaches the end of the object.
        """
        self.index += 1
        if self.index == len(self):
            raise ValueError('fin du parcours')
    
    def __loadyaml(self, yaml_file: str) -> EasyDict:
        """Load a YAML file.

        Args:
            yaml_file (str): The path to the YAML file.

        Raises:
            FileNotFoundError: If the specified YAML file does not exist.

        Returns:
            EasyDict: A dictionary-like object representing the contents of the YAML file.
                        Returns None if the YAML file is empty.
        """
        return EasyDict(yaml.safe_load(open(yaml_file, 'r')))
    
    def __get_all_item(self, search: EasyDict, keys: list[str] = []) -> None:
        """Get all Item class to search in.

        Args:
            search (EasyDict): The dictionary to search in.
            keys (list[str], optional): The list of keys representing the current path in the dictionary. Defaults to [].
        """
        for key, value in search.items():
            new_keys: list[str] = keys + [key]
            if type(value) == EasyDict:
                self.__get_all_item(value, new_keys)
            elif type(value) == list:
                self.items.append(Item(keys=new_keys, possibles_values=value))
    
    def __get_all_possibilities(self) -> list[tuple[int]]:
        """
        Get all possible combinations of values for each possibility.
        Returns:
            A list of tuples representing all possible combinations of values for each possibility.
            If there are no possibilities, an empty list is returned.
        """
        return list(product(*[range(x) for x in self.possibilities]))
    
    def __len__(self) -> int:
        """
        Returns the length of the object.
        """
        return self.len
    


@dataclass
class Item:
    keys: list[str]                 # keys path: config[key[0]][key[1]]... to get the item
    possibles_values: list[Any]     # possibles values for the item
    
    def get_value(self, index_value: int) -> Any:
        """ get a index_value of possibles_value"""

        if not (0 <= index_value < len(self)):
            raise ValueError('index_value out of range. '
                             f'{index_value = } but {len(self) = }')
        
        return self.possibles_values[index_value]
    
    def change_config(self, config: dict, index_value: int) -> None:
        aux: dict = config
        for key in self.keys[:-1]:
            aux = aux[key]
        aux[self.keys[-1]] = self.get_value(index_value)
    
    def __len__(self) -> int:
        return len(self.possibles_values)
    
    def __repr__(self) -> str:
        return f'{str(self.keys):<50} : {self.possibles_values}'


if __name__ == '__main__':
    search = Search(config_yaml_file=os.path.join('config', 'config.yaml'),
                    search_yaml_file=os.path.join('config', 'search.yaml'),
                    logspath='logs',
                    name='random_search')
    
    for i in range(len(search)):
        config = search.get_new_config()
        print(i, config.learning.learning_rate, config.learning.adv.learning_rate_adversary, config.learning.adv.alpha)
    
  
    # item = Item(keys=['learning', 'learning_rate'],
    #             possibles_values=[0.1, 0.2, 0.3])
    # config_path = os.path.join('config', 'config.yaml')
    # config = EasyDict(yaml.safe_load(open(config_path, 'r')))

    # item.change_config(config, index_value=2)
    # print(config.learning.learning_rate)