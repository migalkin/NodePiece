from typing import List, Dict, Union
from pathlib import Path
from collections import namedtuple
import warnings
import os
import time
import json
import torch
import numpy as np
import pickle
import traceback

class ImproperCMDArguments(Exception): pass
class MismatchedDataError(Exception): pass
class BadParameters(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "Unexpected value of parameter {0}".format(dErrorArguments))
        self.dErrorArguments = dErrorArguments

tosave = namedtuple('ObjectsToSave','fname obj')

class FancyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self

class Timer:
    """ Simple block which can be called as a context, to know the time of a block. """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

def default_eval(y_pred, y_true):
    """
        Expects a batch of input

        :param y_pred: tensor of shape (b, nc)
        :param y_true: tensor of shape (b, 1)
    """
    return torch.mean((torch.argmax(y_pred, dim=1) == y_true).float())

def compute_mask(t: Union[torch.Tensor, np.array], padding_idx=0):
    """
    compute mask on given tensor t
    :param t: either a tensor or a nparry
    :param padding_idx: the ID used to represented padded data
    :return: a mask of the same shape as t
    """
    if type(t) is np.ndarray:
        mask = np.not_equal(t, padding_idx)*1.0
    else:
        mask = torch.ne(t, padding_idx).float()
    return mask

# Transparent, and simple argument parsing FTW!
def convert_nicely(arg, possible_types=(bool, float, int, str)):
    """ Try and see what sticks. Possible types can be changed. """
    for data_type in possible_types:
        try:

            if data_type is bool:
                # Hard code this shit
                if arg in ['T', 'True', 'true']: return True
                if arg in ['F', 'False', 'false']: return False
                raise ValueError
            else:
                proper_arg = data_type(arg)
                return proper_arg
        except ValueError:
            continue
    # Here, i.e. no data type really stuck
    warnings.warn(f"None of the possible datatypes matched for {arg}. Returning as-is")
    return arg


def parse_args(raw_args: List[str], compulsory: List[str] = (), compulsory_msg: str = "",
               types: Dict[str, type] = None, discard_unspecified: bool = False):
    """
        I don't like argparse.
        Don't like specifying a complex two liner for each every config flag/macro.

        If you maintain a dict of default arguments, and want to just overwrite it based on command args,
        call this function, specify some stuff like

    :param raw_args: unparsed sys.argv[1:]
    :param compulsory: if some flags must be there
    :param compulsory_msg: what if some compulsory flags weren't there
    :param types: a dict of confignm: type(configvl)
    :param discard_unspecified: flag so that if something doesn't appear in config it is not returned.
    :return:
    """

    # parsed_args = _parse_args_(raw_args, compulsory=compulsory, compulsory_msg=compulsory_msg)
    #
    # # Change the "type" of arg, anyway

    parsed = {}

    while True:

        try:                                        # Get next value
            nm = raw_args.pop(0)
        except IndexError:                          # We emptied the list
            break

        # Get value
        try:
            vl = raw_args.pop(0)
        except IndexError:
            raise ImproperCMDArguments(f"A value was expected for {nm} parameter. Not found.")

        # Get type of value
        if types:
            try:
                parsed[nm] = types[nm](vl)
            except ValueError:
                raise ImproperCMDArguments(f"The value for {nm}: {vl} can not take the type {types[nm]}! ")
            except KeyError:                    # This name was not included in the types dict
                if not discard_unspecified:     # Add it nonetheless
                    parsed[nm] = convert_nicely(vl)
                else:                           # Discard it.
                    continue
        else:
            parsed[nm] = convert_nicely(vl)

    # Check if all the compulsory things are in here.
    for key in compulsory:
        try:
            assert key in parsed
        except AssertionError:
            raise ImproperCMDArguments(compulsory_msg + f"Found keys include {[k for k in parsed.keys()]}")

    # Finally check if something unwanted persists here
    return parsed


def mt_save_dir(parentdir: Path, _newdir: bool = False):
    """
            Function which returns the last filled/or newest unfilled folder in a particular dict.
            Eg.1
                parentdir is empty dir
                    -> mkdir 0
                    -> cd 0
                    -> return parentdir/0

            Eg.2
                ls savedir -> 0, 1, 2, ... 9, 10, 11
                    -> mkdir 12 && cd 12 (if newdir is True) else 11
                    -> return parentdir/11 (or 12)

            ** Usage **
            Get a path using this function like so:
                parentdir = Path('runs')
                savedir = save_dir(parentdir, _newdir=True)

        :param parentdir: pathlib.Path object of the parent directory
        :param _newdir: bool flag to save in the last dir or make a new one
        :return: None
    """

    # Check if the dir exits
    try:
        assert parentdir.exists(), f'{parentdir} does not exist. Making it'
    except AssertionError:
        parentdir.mkdir(parents=False)

    # List all folders within, and convert them to ints
    existing = sorted([int(x) for x in os.listdir(parentdir) if x.isdigit()], reverse=True)

    if not existing:
        # If no subfolder exists
        parentdir = parentdir / '0'
        parentdir.mkdir()
    elif _newdir:
        # If there are subfolders and we want to make a new dir
        parentdir = parentdir / str(existing[0] + 1)
        parentdir.mkdir()
    else:
        # There are other folders and we dont wanna make a new folder
        parentdir = parentdir / str(existing[0])

    return parentdir


def mt_save(savedir: Path, message: str = None, message_fname: str = None, torch_stuff: list = None, pickle_stuff: list = None,
            numpy_stuff: list = None, json_stuff: list = None):
    """

        Saves bunch of diff stuff in a particular dict.

        NOTE: all the stuff to save should also have an accompanying filename, and so we use tosave named tuple defined above as
            tosave = namedtuple('ObjectsToSave','fname obj')

        ** Usage **
        # say `encoder` is torch module, and `traces` is a python obj (dont care what)
        parentdir = Path('runs')
        savedir = save_dir(parentdir, _newdir=True)
        save(
                savedir,
                torch_stuff = [tosave(fname='model.torch', obj=encoder)],
                pickle_stuff = [tosave('traces.pkl', traces)]
            )


    :param savedir: pathlib.Path object of the parent directory
    :param message: a message to be saved in the folder alongwith (as text)
    :param torch_stuff: list of tosave tuples to be saved with torch.save functions
    :param pickle_stuff: list of tosave tuples to be saved with pickle.dump
    :param numpy_stuff: list of tosave tuples to be saved with numpy.save
    :param json_stuff: list of tosave tuples to be saved with json.dump
    :return: None
    """

    assert savedir.is_dir(), f'{savedir} is not a directory!'

    # Commence saving shit!
    if message:
        with open(savedir / 'message.txt' if message_fname is None else savedir / message_fname, 'w+') as f:
            f.write(message)

    for data in torch_stuff or ():
        try:
            torch.save(data.obj, savedir / data.fname)
        except:
            traceback.print_exc()

    for data in pickle_stuff or ():
        try:
            pickle.dump(data.obj, open(savedir / data.fname, 'wb+'))
        except:
            traceback.print_exc()

    for data in numpy_stuff or ():
        try:
            np.save(savedir / data.fname, data.obj)
        except:
            traceback.print_exc()

    for data in json_stuff or ():
        # Filter out: only things belonging to specific types should be saved.
        saving_data = {k: v for k, v in data.obj.items() if type(v) in [str, int, float, np.int, np.float, np.long, bool]}
        try:
            json.dump(saving_data, open(savedir / data.fname, 'w+'))
        except:
            traceback.print_exc()


class SimplestSampler:
    """
        Given X and Y matrices (or lists of lists),
            it returns a batch worth of stuff upon __next__
    :return:
    """

    def __init__(self, data, bs: int = 64):

        try:
            assert len(data["x"]) == len(data["y"])
        except AssertionError:

            raise MismatchedDataError(f"Length of x is {len(data['x'])} while of y is {len(data['y'])}")

        self.x = data["x"]
        self.y = data["y"]
        self.n = len(self.x)
        self.bs = bs  # Batch Size

    def __len__(self):
        return self.n // self.bs - (1 if self.n % self.bs else 0)

    def __iter__(self):
        self.i, self.iter = 0, 0
        return self

    def __next__(self):
        if self.i + self.bs >= self.n:
            raise StopIteration

        _x, _y = self.x[self.i:self.i + self.bs], self.y[self.i:self.i + self.bs]
        self.i += self.bs

        return _x, _y


