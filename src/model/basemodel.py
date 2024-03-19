from typing import Iterator

import torch
from torch import nn
from torch.nn import Parameter


class Model(nn.Module):
    """ BaseModel with method for get parameter more easily """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the BaseModel object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
    
    def get_parameters(self) -> Iterator[Parameter]:
        """
        Get parameters.
        
        Returns:
            Iterator[Parameter]: The parameters of the model.
        """
        return self.parameters()
    
    def get_learned_parameters(self) -> Iterator[Parameter]:
        """
        Get learned parameters.

        Returns:
            Iterator[Parameter]: An iterator over the learned parameters.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param
    
    def get_name_learned_parameters(self) -> Iterator[tuple[str, Parameter]]:
        """
        Get the names and parameters of the learnable parameters.

        Args:
            self: The instance of the BaseModel class.

        Returns:
            An iterator yielding tuples of the form (name, parameter) where the parameter is learnable.
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def get_number_parameters(self) -> int:
        """
        Get the number of parameters.

        Args:
            self: The BaseModel instance.

        Returns:
            int: The total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def get_number_learnable_parameters(self) -> int:
        """Get the number of learned parameters.

        Args:
            self: The instance of the BaseModel class.

        Returns:
            int: The number of learned parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_dict_learned_parameters(self) -> dict[str, Parameter]:
        """
        Get learned parameters in a dictionary to save it.
        
        Returns:
            A dictionary containing the learned parameters.
        """
        state_dict: dict[str, Parameter] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state_dict[name] = param

        return state_dict
    
    def load_dict_learnable_parameters(self,
                                       state_dict: dict[str, Parameter],
                                       strict: bool=True,
                                       verbose: bool=False
                                       ) -> None:
        """Load only learnable parameters from a dictionary.

        Args:
            state_dict (dict[str, Parameter]): A dictionary that contains the name and the parameters.
            strict (bool, optional): If True, raise an error if there are some matching errors. Defaults to True.
            verbose (bool, optional): If True and strict=False, print a warning error if there are some matching errors. Defaults to False.

        Raises:
            ValueError: If strict is True and there are matching errors.

        Returns:
            None: This method does not return anything.

        """
        loaded_keys = []
        for name, param in self.named_parameters():
            # if name in keys
            if name in state_dict.keys():
                if param.requires_grad:
                    with torch.no_grad():
                        param.copy_(state_dict[name])
                    loaded_keys.append(name)
                elif strict or verbose:
                    error_message = f'parameter:{name} is in state_dict.keys ' + \
                                    f'but this param has requires_grad=False'
                    print_error_message(error_message, strict, verbose)
            
            # if name not in keys but param is requires_grad
            elif param.requires_grad:
                error_message = f'parameter:{name} not in state_dict.keys ' + \
                                f'but this param has requires_grad=True'
                print_error_message(error_message, strict, verbose)

        # if loaded_keys != state_dict.keys()   (we know that: state_dict.keys() c loaded_keys)
        if len(loaded_keys) != len(state_dict):
            missing_keys = [key for key in state_dict.keys() if key not in loaded_keys]
            error_message = f"the following parameter wasn't loaded: {missing_keys}"
            print_error_message(error_message, strict, verbose)
        
        print('model weigths were successfully loaded')
    


def print_error_message(error_message: str, strict: bool, verbose: bool) -> None:
    """
    Print an error message.

    Args:
        error_message (str): The error message to be printed.
        strict (bool): If True, raise a KeyError with the error message.
        verbose (bool): If True, print a warning with the error message.

    Raise:
        KeyError: If strict is True, raise a KeyError with the error message.
    """
    if strict:
        raise KeyError(error_message)
    if verbose:
        print(Warning(error_message))
