import os

from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import *


latest_ekg_location = sc.textFile("s3://alexa-knowledge-evi-knowledge-graph-prod/last-consistency-output-location.txt").first()

print(f"The latest location of the EKG Dump is {latest_ekg_location}")

facts = spark.read.parquet(latest_ekg_location)


facts.printSchema()

filteredFacts = facts.filter(
    (facts["believed"] == True) &
    (facts["supported"] == True) &
    (facts["suppressed"] == False) &
    (facts["environment"] == "batch-prod") &
    (facts["visible"] == True) &
    (facts["eviTriple.isPositive"] == True) &
    (facts["eviTriple.isMetaFact"] == False)
)
filteredFacts.show()

relation_to_sample = "[is the wikidata id of]"

sampled_facts = filteredFacts.filter(filteredFacts["eviTriple.relation"] == relation_to_sample)

sampled_facts.show()
sampled_facts.count()


outputS3Location = "s3://fount.resources.dev/2022/entity_mappings/wikidata_to_id_march_2022_100_fixed/"

triples = sampled_facts.select("eviTriple")

triples.coalesce(100).write.json(outputS3Location)
