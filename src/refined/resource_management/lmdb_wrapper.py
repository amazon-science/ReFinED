import logging
import os
import warnings
from typing import List, Any, Dict, Mapping
from typing import TypeVar

import lmdb
import ujson as json
from tqdm import tqdm

from refined.utilities.general_utils import batch_items

K = TypeVar('K')
V = TypeVar('V')


class LmdbImmutableDict(Mapping[K, V]):
    def __iter__(self):
        NotImplementedError()

    def __getitem__(self, key: K) -> V:
        if not key:
            raise KeyError(key)
        with self.env.begin() as txn:
            value = txn.get(self.encode(key))
        if value is None:
            raise KeyError(key)
        return self.decode(value)

    def encode(self, key: K) -> bytes:
        try:
            return key.encode("utf-8")
        except UnicodeEncodeError as err:
            warnings.warn(f'Unable to encode key {key}, err: {err}')

    def decode(self, value: bytes) -> Dict[Any, Any]:
        return json.loads(value.decode("utf-8"))

    def __len__(self) -> int:
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def __init__(self, path: str, write_mode: bool = False):
        if write_mode:
            self.path = f'{path}.incomplete'
            self.env = lmdb.open(self.path, max_dbs=1, readonly=False, create=True, writemap=True,
                                 subdir=False, map_size=1099511627776 * 2,
                                 meminit=False, map_async=True, mode=0o755,
                                 lock=False)
        else:
            self.path = path
            self.env = lmdb.open(self.path, max_dbs=1, readonly=True, create=True, writemap=False,
                                 subdir=False, map_size=1099511627776 * 2,
                                 meminit=False, map_async=True, mode=0o755,
                                 lock=False)

    def __contains__(self, key: K) -> bool:
        if not key:
            return False
        with self.env.begin() as txn:
            encoded_key = self.encode(key)
            value = txn.get(encoded_key) if encoded_key else None
        return value is not None

    def close(self) -> None:
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get(self, key: K, default_value=None) -> V:
        with self.env.begin() as txn:
            value = txn.get(self.encode(key))
        if value is None:
            return default_value
        return self.decode(value)

    def put(self, key: K, value: V):
        if key is not None and value is not None:
            with self.env.begin(write=True) as txn:
                txn.put(key=key.encode(), value=json.dumps(value).encode())

    def put_batch(self, keys: List[K], values: List[V]):
        with self.env.begin(write=True) as txn:
            for key, value in zip(keys, values):
                try:
                    txn.put(key=key.encode(), value=json.dumps(value).encode())
                except lmdb.Error as err:
                    logging.debug(f'skipping {key}, error: {err}')

    def write_to_compacted_file(self):
        """
        Writes memmap-based data structure to disk in compacted format
        and deletes original over-allocated file.
        Only call this method when this object is finished with.
        """
        self.env.copy(path=self.path.replace('.incomplete', ''), compact=True)
        os.remove(self.path)

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any], output_file_path: str) -> 'LmdbImmutableDict[Any, Any]':
        if os.path.exists(output_file_path):
            print(f'Skipping conversion as {output_file_path} already exists.')
        else:
            output_lmdb_dict = LmdbImmutableDict(output_file_path,
                                                 write_mode=True)
            for batch in tqdm(list(batch_items(input_dict.items(), n=250000)),
                              desc=f'Writing {output_file_path}'):
                keys, values = zip(*batch)
                output_lmdb_dict.put_batch(keys=keys, values=values)
            output_lmdb_dict.write_to_compacted_file()
        return cls(path=output_file_path, write_mode=False)
