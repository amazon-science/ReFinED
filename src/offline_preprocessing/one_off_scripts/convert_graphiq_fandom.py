import csv
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import *

from tqdm.auto import tqdm

csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class SourceStoreDocument:
    """
    Class to represent a document stored in the FOUNT source store.
    Params:
        text: the full document text
        url: document url
        language: language
        uploader: the system or user which uploaded the document to the source store
        title: document title (e.g. HTML page title)
        upload_date: epoch timestamp (s)
        archive_date: format is %Y%m, i.e. YYYYMM
        domain: url domain
        data_format_version: data format version if backward incompatible
                             changes are made update this field
    """

    text: str
    url: str
    language: str
    uploader: str
    title: str
    upload_date: int
    archive_date: str
    domain: str
    dataset: Optional[str] = None
    data_format_version: str = "2.0"

    def __post_init__(self):
        self.check_types()

    def check_types(self):
        """
        Check types for `SourceStoreDocument` fields. If the types are misaligned
        with what is expected, then a TypeError is raised.
        """

        if not all(
                [
                    isinstance(self.text, str),
                    isinstance(self.url, str),
                    isinstance(self.language, str),
                    isinstance(self.uploader, str),
                    isinstance(self.title, str),
                    isinstance(self.upload_date, int),
                    isinstance(self.archive_date, str),
                    isinstance(self.domain, str),
                    isinstance(self.data_format_version, str),
                ]
        ):
            raise TypeError(f"`SourceStoreDocument` type mismatch found when validating: {self}.")

        if not self.dataset and self.data_format_version == "2.0":
            raise TypeError(
                "SourceStoreDocument with version 2.0 and above must include dataset name."
            )

        if self.dataset and not isinstance(self.dataset, str):
            raise TypeError(f"`SourceStoreDocument` dataset field must be str.")

    def to_json(self) -> str:
        """
        Convert self (`SourceStoreDocument`) to JSON string.
        :return: json string
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_string: str) -> "SourceStoreDocument":
        """
        Parse JSON string as `SourceStoreDocument`.
        :param json_string: json string representing InputDocument
        :return: InputDocument object
        """
        kwargs = json.loads(json_string)
        if "__class__" in kwargs:
            kwargs.pop("__class__", None)
        return cls(**kwargs)


with open('FOUNT_fandom_sections_fount_format.json', 'w') as f_out:
    with open('FOUNT_fandom_sections.csv', 'r') as f:
        reader = csv.reader(f)
        for i, row in tqdm(enumerate(reader), total=1300000):
            if i == 0:
                continue
            fandom_unique, domain, character_title, text, url, title = row
            doc = SourceStoreDocument(
                text=text,
                url=url,
                language='en',
                uploader='Graphiq_fandom_1m',
                title=title,
                upload_date=int(time.time()),
                archive_date='202204',
                domain=domain,
                dataset='fandom_fictional'
            )
            f_out.write(doc.to_json() + '\n')


# split -l 50000 FOUNT_fandom_sections_fount_format.json
