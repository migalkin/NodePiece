# -*- coding: utf-8 -*-

"""The **Co**mpletion **D**atasets **Ex**tracted from Wikidata and Wikipedia (CoDEx) datasets from [safavi2020]_.

- GitHub Repository: https://github.com/tsafavi/codex
- Paper: https://arxiv.org/abs/2009.07810
"""

#from docdata import parse_docdata
import logging
import os
import requests
import shutil
import pathlib
import pystow

from pathlib import Path
from pykeen.datasets.base import PathDataSet
from typing import Optional, Mapping, Any, Union
from pystow.utils import name_from_url
from urllib.request import urlretrieve
logger = logging.getLogger(__name__)

BASE_URL = 'https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/'
SMALL_VALID_URL = f'{BASE_URL}/codex-s/valid.txt'
SMALL_TEST_URL = f'{BASE_URL}/codex-s/test.txt'
SMALL_TRAIN_URL = f'{BASE_URL}/codex-s/train.txt'

MEDIUM_VALID_URL = f'{BASE_URL}/codex-m/valid.txt'
MEDIUM_TEST_URL = f'{BASE_URL}/codex-m/test.txt'
MEDIUM_TRAIN_URL = f'{BASE_URL}/codex-m/train.txt'

LARGE_VALID_URL = f'{BASE_URL}/codex-l/valid.txt'
LARGE_TEST_URL = f'{BASE_URL}/codex-l/test.txt'
LARGE_TRAIN_URL = f'{BASE_URL}/codex-l/train.txt'

PYKEEN_MODULE: pystow.Module = pystow.module('pykeen')
#: A path representing the PyKEEN data folder
#PYKEEN_HOME: Path = PYKEEN_MODULE.base
#: A subdirectory of the PyKEEN data folder for datasets, defaults to ``~/.data/pykeen/datasets``
PYKEEN_DATASETS: Path = PYKEEN_MODULE.get('datasets')
# If GitHub ever gets upset from too many downloads, we can switch to
# the data posted at https://github.com/pykeen/pykeen/pull/154#issuecomment-730462039

def _urlretrieve(url: str, path: str, clean_on_failure: bool = True, stream: bool = True) -> None:
    """Download a file from a given URL.
    :param url: URL to download
    :param path: Path to download the file to
    :param clean_on_failure: If true, will delete the file on any exception raised during download
    :param stream: If true, use :func:`requests.get`. By default, use ``urlretrieve``.
    :raises Exception: If there's a problem wih downloading via :func:`requests.get` or copying
        the data with :func:`shutil.copyfileobj`
    :raises KeyboardInterrupt: If the user quits during download
    """
    if not stream:
        logger.info('downloading from %s to %s', url, path)
        urlretrieve(url, path)  # noqa:S310
    else:
        # see https://requests.readthedocs.io/en/master/user/quickstart/#raw-response-content
        # pattern from https://stackoverflow.com/a/39217788/5775947
        try:
            with requests.get(url, stream=True) as response, open(path, 'wb') as file:
                logger.info('downloading (streaming) from %s to %s', url, path)
                shutil.copyfileobj(response.raw, file)
        except (Exception, KeyboardInterrupt):
            if clean_on_failure:
                os.remove(path)
            raise

class UnpackedRemoteDataset(PathDataSet):
    """A dataset with all three of train, test, and validation sets as URLs."""

    def __init__(
        self,
        training_url: str,
        testing_url: str,
        validation_url: str,
        cache_root: Optional[str] = None,
        stream: bool = True,
        force: bool = False,
        eager: bool = False,
        create_inverse_triples: bool = False,
        load_triples_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize dataset.
        :param training_url: The URL of the training file
        :param testing_url: The URL of the testing file
        :param validation_url: The URL of the validation file
        :param cache_root:
            An optional directory to store the extracted files. Is none is given, the default PyKEEN directory is used.
            This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to ``~/.pykeen``.
        :param stream: Use :mod:`requests` be used for download if true otherwise use :mod:`urllib`
        :param force: If true, redownload any cached files
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param load_triples_kwargs: Arguments to pass through to :func:`TriplesFactory.from_path`
            and ultimately through to :func:`pykeen.triples.utils.load_triples`.
        """
        self.cache_root = self._help_cache(cache_root)

        self.training_url = training_url
        self.testing_url = testing_url
        self.validation_url = validation_url

        training_path = os.path.join(self.cache_root, name_from_url(self.training_url))
        testing_path = os.path.join(self.cache_root, name_from_url(self.testing_url))
        validation_path = os.path.join(self.cache_root, name_from_url(self.validation_url))

        for url, path in [
            (self.training_url, training_path),
            (self.testing_url, testing_path),
            (self.validation_url, validation_path),
        ]:
            if os.path.exists(path) and not force:
                continue
            _urlretrieve(url, path, stream=stream)

        super().__init__(
            training_path=training_path,
            testing_path=testing_path,
            validation_path=validation_path,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
            #load_triples_kwargs=load_triples_kwargs,
        )

    def _help_cache(self, cache_root: Union[None, str, pathlib.Path]) -> pathlib.Path:
        """Get the appropriate cache root directory.
        :param cache_root: If none is passed, defaults to a subfolder of the
            PyKEEN home directory defined in :data:`pykeen.constants.PYKEEN_HOME`.
            The subfolder is named based on the class inheriting from
            :class:`pykeen.datasets.base.Dataset`.
        :returns: A path object for the calculated cache root directory
        """
        if cache_root is None:
            cache_root = PYKEEN_DATASETS
        cache_root = pathlib.Path(cache_root) / self.__class__.__name__.lower()
        cache_root.mkdir(parents=True, exist_ok=True)
        logger.debug('using cache root at %s', cache_root)
        return cache_root


#docs]@parse_docdata
class CoDExSmall(UnpackedRemoteDataset):
    """The CoDEx small dataset.

    ---
    name: CoDEx (small)
    citation:
        author: Safavi
        year: 2020
        link: https://arxiv.org/abs/2009.07810
        github: tsafavi/codex
    statistics:
        entities: 2034
        relations: 42
        training: 32888
        testing: 1828
        validation: 1827
        triples: 36543
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the `CoDEx <https://github.com/tsafavi/codex>`_ small dataset from [safavi2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        # GitHub's raw.githubusercontent.com service rejects requests that are streamable. This is
        # normally the default for all of PyKEEN's remote datasets, so just switch the default here.
        kwargs.setdefault('stream', False)
        super().__init__(
            training_url=SMALL_TRAIN_URL,
            testing_url=SMALL_TEST_URL,
            validation_url=SMALL_VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )



#[docs]@parse_docdata
class CoDExMedium(UnpackedRemoteDataset):
    """The CoDEx medium dataset.

    ---
    name: CoDEx (medium)
    citation:
        author: Safavi
        year: 2020
        link: https://arxiv.org/abs/2009.07810
        github: tsafavi/codex
    statistics:
        entities: 17050
        relations: 51
        training: 185584
        testing: 10311
        validation: 10310
        triples: 206205
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the `CoDEx <https://github.com/tsafavi/codex>`_ medium dataset from [safavi2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        kwargs.setdefault('stream', False)  # See comment in CoDExSmall
        super().__init__(
            training_url=MEDIUM_TRAIN_URL,
            testing_url=MEDIUM_TEST_URL,
            validation_url=MEDIUM_VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )



#[docs]@parse_docdata
class CoDExLarge(UnpackedRemoteDataset):
    """The CoDEx large dataset.

    ---
    name: CoDEx (large)
    citation:
        author: Safavi
        year: 2020
        link: https://arxiv.org/abs/2009.07810
        github: tsafavi/codex
    statistics:
        entities: 77951
        relations: 69
        training: 551193
        testing: 30622
        validation: 30622
        triples: 612437
    """

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the `CoDEx <https://github.com/tsafavi/codex>`_ large dataset from [safavi2020]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        kwargs.setdefault('stream', False)  # See comment in CoDExSmall
        super().__init__(
            training_url=LARGE_TRAIN_URL,
            testing_url=LARGE_TEST_URL,
            validation_url=LARGE_VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )



def _main():
    for cls in [CoDExSmall, CoDExMedium, CoDExLarge]:
        d = cls()
        d.summarize()


if __name__ == '__main__':
    _main()