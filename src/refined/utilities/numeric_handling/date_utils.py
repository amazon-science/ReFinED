import dateutil
import re
import copy

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from refined.data_types.base_types import Entity, Span


@dataclass
class Date:
    text: Optional[str] = None  # the text representing this date
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    offset: Optional[int] = None  # We split some date spans into multiple instances of the Date class (e.g. 1999 to
                                  # 2010) would be split into two separate dates. This is the character level offset of
                                  # this date instance relative to the start of the original span
    known_format: bool = True  # If True, we can identify if this date is day/month-first
                               # (e.g. "30/01/2021" is definitely day-first)

    def __post_init__(self):

        if self.year is None and self.month is None and self.day is None:
            raise AssertionError("All of year, month and day are None")

        if self.month is not None:
            assert 1 <= self.month <= 12, f"month should be between 1 and 12 but is {self.month}"

        if self.day is not None:
            assert 1 <= self.day <= 31, f"day should be between 1 and 31 but is {self.day}"

    def can_identify_format(self) -> bool:
        """
        Check if we can identify whether this dates is in day-first or month-first format
        """
        if self.day is None or self.month is None:
            return True

        # If the date string contains any letters then assume we can identify format properly
        if bool(re.search(r"[a-zA-Z]", self.text)):
            return True

        # If the day is 12 or under, it is impossible to identify if this is a US format date (month first)
        if self.day <= 12:
            return False

        return True

    def get_doc_format(self) -> Optional[str]:
        """
        Check if the date gives away whether dates in this document are UK style (day/month/year) or US style
         (month/day/year)
        :return: one of ["day_first", "month_first", None] indicating the style of date
        """
        if self.day is None or self.month is None:
            return None

        # If the date string contains any letters then does not give away format of number-only dates
        if bool(re.search(r"[a-zA-Z]", self.text)):
            return None

        # If day is 12 or under then this date doesn't reveal more general format
        if self.day <= 12:
            return None

        numbers_only = "".join([l for l in self.text if l.isnumeric()])

        if str(self.day) in numbers_only and numbers_only.index(str(self.day)) == 0:
            return "day_first"
        else:
            return "month_first"

    def to_string(self) -> Optional[str]:
        """
        Attempt to convert date to a string
        """
        if self.day is None and self.month is None and self.year is None:
            return None

        if self.day is not None and self.month is not None and self.year is not None:
            return f'[timepoint: ["{self.year}/{self.month}/{self.day}"]]'

        if self.day is None and self.month is not None and self.year is not None:
            return f'[timepoint: ["{self.year}/{self.month}"]]'

        if self.day is None and self.month is None and self.year is not None:
            return f'[timepoint: ["{self.year}"]]'

        if self.day is not None and self.month is not None and self.year is None:
            return f'[day of the year: ["{self.month}/{self.day}"]]'

        return None


class DateHandler:
    """
    Handles resolving the string of a date mention to an instance of the Date class
    """
    def __init__(self):

        self.parser = dateutil.parser.parser()
        self.start_regex = re.compile(r"^(early|late|mid|the start of|the end of|the middle of|the year)( |-)")
        self.split_regex = re.compile(r"\s?--\s?| - | to ")

    def resolve_dates(self, text: str) -> List[Date]:
        """
        Given a text input representing a date, resolve it to a list of Date instances
        :param text: the text representing the date
        :return: list of Date instances (returns a list as in some cases like "1988 to 1999" we will split the input
        into two separate Date instances
        """
        date_texts = self._split_date_text(text=text)

        dates = [self._get_date_from_text(text, offset) for text, offset in date_texts]

        return [date for date in dates if date is not None]

    def resolve_multiple(self, dates: List[Date]) -> List[Date]:
        """
        Given a list of dates all of which are from the same doc/page, check to see if can resolve any additional dates
        by identifying the format of dates in this page (i.e. are dates month first or day first format). For example,
        if a date like "30/01/2021" gives away that dates in this document are formatted with the day before the month,
        then can potentially resolve other dates in the doc such as "01/02/2021" where this is not clear from the
        individual date alone
        :param dates: list of Date instances, all of which should be from the same document/page
        :return: the same list of dates, potentially with extra dates within the list properly resolved
        """

        date_format = self._get_date_formats(dates=dates)

        if date_format is None:
            return dates

        for ix, date in enumerate(dates):

            if date.known_format:
                continue

            dates[ix] = self._get_date_from_text(date_text=date.text, offset=date.offset, date_format=date_format)

        return dates

    def _get_date_formats(self, dates: List[Date]) -> Optional[str]:
        """
        Given a list of dates, see if we can identify whether dates in the list are "month_first" or "day_first" format
        :param dates: list of Date instances, all of which should be from the same document/page
        :return: if possible, return the format of the input dates - one of ["day_first", "month_first", None]
        """
        date_formats = [date.get_doc_format() for date in dates]
        date_formats = list(set([f for f in date_formats if f is not None]))

        # If we have exactly one resolved format, then can assume this applies to all dates in this set
        if len(date_formats) == 1:
            date_format = date_formats[0]
        else:
            date_format = None

        return date_format

    def _split_date_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Split date string into multiple parts, each of which refers to a single date.
        E.g. "1988 to 1999" -> ["1988", "1999"]. Keeps track of the start index of each parsed date in the
        original text
        :param text: the text to (potentially) split into multiple subsections, each of which should represent one date
        """
        date_texts = re.split(self.split_regex, text)

        if len(date_texts) == 1:
            return [(text, 0)]
        else:
            new = []
            offset = 0
            for dt in date_texts:
                start_ix_in_text = text[offset:].index(dt) + offset
                offset += len(dt)
                new.append((dt, start_ix_in_text))

        return new

    def _get_date_from_text(self,
                            date_text: str,
                            offset: Optional[int] = None,
                            date_format: Optional[str] = None) -> Optional[Date]:
        """
        Given a segment of text representing a date, try and resolve it to a Date instance
        e.g. "20/01/2020" -> Date(day=20, month=1, year=2020)
        :param date_text: the segment of text to attempt to resolve to a date
        :param offset: in some cases we have split a larger segment of text into smaller subsections. In these cases
                        "offset" keeps track of the start character index of this subsection in the original text
        :param date_format: if known, one of "month_first" (denoting numerical dates are "US" style with month listed
                            before the day - e.g. "01/30/2020") or "day_first" (denoting day is listed before month -
                            e.g. "30/01/2020")
        :return: Date instance if we are able to resolve to a set date, else None
        """
        # Remove pre-text
        date_text_cleaned = re.sub(self.start_regex, "", date_text)

        res, _ = self.parser._parse(date_text_cleaned)

        if res is not None:
            try:
                date = Date(text=date_text, year=res.year, month=res.month, day=res.day, offset=offset)
            # Catch invalid dates which will throw error in __post_init__ method of Date class
            except AssertionError:
                date = None
        else:
            date = None

        if date is not None and not date.can_identify_format():

            # If we can't identify the format properly (i.e. can't tell if day is first or month is first),
            # then set the day, month, year to None for now
            if date_format is None:
                date.day, date.month, date.year = None, None, None
                date.known_format = False

            # If we know that dates are day first, then switch the day and the month, as the dateutil
            # parser defaults to month
            else:
                if date_format == "day_first":
                    day = copy.copy(date.month)
                    month = copy.copy(date.day)
                    date.day = day
                    date.month = month

        return date

    @staticmethod
    def set_non_test_attrs(date: Optional[Date]) -> Optional[Date]:
        """
        Used for testing - set attributes not defined in the test instances to None
        """
        if date is None:
            return date

        for attr in date.__dict__.keys():

            if attr not in {"day", "month", "year", "known_format"}:
                setattr(date, attr, None)

        return date

    def test_date_conversion(self, test_date_mentions: Dict):
        """
        Used for testing the date resolution code -> test cases are in TEST_DATE_MENTIONS in
        utilities/date_test_examples.py.
        """
        for date, target in test_date_mentions.items():

            parsed_dates = self.resolve_dates(date)

            # Remove unnecessary attribrutes from test instances
            parsed_dates = [self.set_non_test_attrs(date) for date in parsed_dates]

            if len(parsed_dates) == 1:
                parsed_dates = parsed_dates[0]
            else:
                parsed_dates = tuple(parsed_dates)

                if any(i is None for i in parsed_dates):
                    parsed_dates = None

            assert parsed_dates == target, f"Failed - expected {date} to convert to {target} but got {parsed_dates}"

    def test_multi_conversion(self, test_multi_mentions: Dict):
        """
        Used for testing the date resolution code which attempts to identify the format of ambiguous date mentions from
         other dates in the same doc -> test cases are in TEST_MULTI_MENTIONS in utilities/date_test_examples.py.
        """
        for date_texts, target in test_multi_mentions.items():

            parsed_dates = []
            for date_text in date_texts:
                parsed_dates += self.resolve_dates(date_text)

            parsed_dates = self.resolve_multiple(parsed_dates)

            # Remove unnecessary attribrutes from test instances
            parsed_dates = tuple([self.set_non_test_attrs(date) for date in parsed_dates])

            assert parsed_dates == target, f"Failed - expected {date_texts} to convert to {target} but got " \
                                           f"{parsed_dates}"


class SpanDateHandler:
    """
    For Span instances which have been identified by the MD layer as having type "DATE", resolve the dates from their
    text format (e.g. "12th June 1996") to an instance of the Date class
    """
    def __init__(self):

        self.date_handler = DateHandler()

    def resolve_spans(self, spans: List[Span]) -> List[Span]:
        """
        Given a list of spans with MD type "DATE", attempt to resolve them to a set date format. The list of spans
        should all be from the same document, as this method will attempt to detect the general format of dates in
        this document (i.e. day-first or month-first) from other spans in the list
        :param spans: list of spans identified as having ner type "DATE" by the mention detection layer
        """
        resolved_spans = []
        for span in spans:
            resolved_spans += self.resolve_date_span(span=span)

        span_indices = [ix for ix, span in enumerate(resolved_spans) if span.date is not None]
        span_dates = [span.date for ix, span in enumerate(resolved_spans) if span.date is not None]

        # If possible, resolve additional dates by identifying the common date format for this list of spans
        span_dates = self.date_handler.resolve_multiple(dates=span_dates)

        # Filter out dates which are incorrect - e.g. dates with less than two numbers in the year (as a year of e.g.
        # "86" will often be wrong and refer to someone's age instead of an actual year)
        span_dates = [self._check_for_incorrect_resolution(date) for date in span_dates]

        for ix, date in enumerate(span_dates):
            span_index = span_indices[ix]
            resolved_spans[span_index].date = date

        # Add predicted entity to span in same format as standard entities
        for span in resolved_spans:
            if span.date is None:
                continue
            date_string = span.date.to_string()
            if date_string is not None:
                span.predicted_entity = Entity(parsed_string=date_string)
                span.entity_linking_model_confidence_score = 1.0

        return resolved_spans

    def resolve_date_span(self, span: Span) -> List[Span]:
        """
        Resolve a single date -> i.e. convert raw text form to an instance of the Date class
        :param span: the span to resolve to a date
        """
        dates = self.date_handler.resolve_dates(span.text)

        # No resolved Date instance
        if len(dates) == 0:
            return [span]

        # Span has been resolved to a single DATE instance
        if len(dates) == 1:
            span.date = dates[0]
            return [span]

        # Span has been split into multiple Date instances - generate a new Span for each
        if len(dates) > 1:
            new_spans = []
            for date in dates:
                new_span = copy.copy(span)
                new_span.text = date.text
                new_span.start = span.start + date.offset
                new_span.ln = len(date.text)
                new_span.date = date
                new_spans.append(new_span)
            return new_spans

    def _check_for_incorrect_resolution(self, date: Date) -> Optional[Date]:
        """
        Rule based techniques for filtering out any dates which are potentially incorrect
        """
        # Filter out dates with less than two numbers in the year (as a year of e.g. "86" will often be wrong and refer
        # to someone's age instead of an actual year)
        if date.year is not None and len(date.text) == 2:
            return None

        return date
