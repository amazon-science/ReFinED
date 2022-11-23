
from refined.utilities.numeric_handling.date_utils import Date


TEST_DATE_MENTIONS = {
    # Just years
    "2012": Date(year=2012),
    "the start of 2012": Date(year=2012),
    "the end of 2012": Date(year=2012),
    "the year 1988": Date(year=1988),
    "1945 to 1988": (Date(year=1945), Date(year=1988)),

    # Just days
    "23rd": Date(day=23),

    # Just months
    "February": Date(month=2),
    "early June": Date(month=6),
    "early-June": Date(month=6),
    "mid-June": Date(month=6),
    "late-June": Date(month=6),

    # Day + Month
    "February 21st": Date(day=21, month=2),
    "Feb 21st": Date(day=21, month=2),
    "Feb. 21st": Date(day=21, month=2),

    # Month + Year
    "March 2021": Date(month=3, year=2021),

    # Day + Month + Year
    "February 21st 2009": Date(day=21, month=2, year=2009),
    "Feb. 21st 2009": Date(day=21, month=2, year=2009),
    "30/01/2021": Date(day=30, month=1, year=2021),
    "01/30/2021": Date(day=30, month=1, year=2021),

    "01/01/2021": Date(known_format=False),  # Can't tell whether US format date or not so resolve
    # to None for now
    "30-01-2021": Date(day=30, month=1, year=2021),
    "01-30-2021": Date(day=30, month=1, year=2021),
    "30.01.2021": Date(day=30, month=1, year=2021),

    # TODO: Decades
    "80s": None,
    "the 1980s": None,
    "the 1960s to the 1980s": None,
    "the early forties": None,

    # TODO: Centuries
    "the nineteenth century": None,
    "nineteenth century": None,
    "the 19th century": None,
    "19th century": None,
    "the fifth century A.D.": None,

    # TODO: BC and AC

    # Miscellaneous that shouldn't be parsed
    "1-day": None,
    "bimonthly": None,
    "this Christmas": None,
    "three to four months": None,
    "Two Days": None,
    "16 years": None,
    "the months": None,
    "years past": None,
    "many months": None,
    "Earlier this week": None,
    "Twenty-five years later": None,
    "century-old": None,
    "This July": None
}

TEST_MULTI_MENTIONS = {
    ("01/30/2021", "01/02/2021"): (Date(day=30, month=1, year=2021), Date(day=2, month=1, year=2021)),
    ("01/02/2021", "01/03/2021"): (Date(known_format=False), Date(known_format=False))
}
