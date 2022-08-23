import boto3
import time

from typing import Tuple, Dict, List

from utilities.general_utils import get_logger


LOGGER = get_logger(__name__)


class AthenaQueryReader:

    def __init__(self, client: boto3.client, query_execution_id: str, max_results: int = 10, loud: bool = True):
        self.client: boto3.client = client
        self.query_execution_id: str = query_execution_id
        self.max_results = max_results
        self.loud = loud

    def get_query_status(self) -> Tuple[bool, bool]:
        queryExecution = self.client.get_query_execution(QueryExecutionId=self.query_execution_id)
        status = queryExecution["QueryExecution"]["Status"]["State"]

        completed = status in {"SUCCEEDED", "FAILED", "CANCELLED"}
        successful = status == "SUCCEEDED"

        if status in {"FAILED", "CANCELLED"}:
            raise Exception(f"query not completed with status {status}", queryExecution)

        return completed, successful

    def wait_for_query(self, sleep_time: int = 2, log_every: int = 5) -> bool:

        if self.loud:
            LOGGER.info(f"Waiting for athena query {self.query_execution_id}")
        completed = False
        attempt = 0
        while not completed:
            completed, success = self.get_query_status()
            time.sleep(sleep_time)
            attempt += 1

            if attempt % log_every == 0 and self.loud:
                LOGGER.info(f"completed={completed}, success={success}")

        return success

    def stream_query_results(self):

        success = self.wait_for_query()

        # Iterate query results
        response, next_token = self._get_results_and_next_token()
        yield from response
        while next_token is not None:
            response, next_token = self._get_results_and_next_token(next_token=next_token)
            yield from response

    def get_columns(self, results: Dict) -> List[str]:
        columns = results["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        columns = [c["Label"] for c in columns]
        return columns

    def _get_results_and_next_token(self, next_token: str = None):
        if next_token:
            res = self.client.get_query_results(
                QueryExecutionId=self.query_execution_id,
                MaxResults=self.max_results,
                NextToken=next_token,
            )
            response = res["ResultSet"]["Rows"]


        else:
            res = self.client.get_query_results(
                QueryExecutionId=self.query_execution_id, MaxResults=self.max_results
            )
            response = res["ResultSet"]["Rows"][1:]

        columns = self.get_columns(results=res)

        results = []
        for row in response:
            results.append({columns[ix]: r["VarCharValue"] for ix, r in enumerate(row["Data"])})

        next_token = res.get("NextToken", None)

        return results, next_token


class AthenaQueries:
    """
    Materials set:
    https://odin.amazon.com/#view/materialSet/com.amazon.access.fount-ml-prediction-pipeline-dev-test-user-1
    """

    def __init__(self, table_name: str, output_location: str = "s3://fshjos-test", loud: bool = True):

        self.table_name: str = table_name
        self.output_location: str = output_location
        self.client = boto3.client("athena", region_name="us-east-1")
        self.loud = loud

    def get_document_uri_from_sentence(self, sentences: List[str]) -> Dict[str, str]:
        """
        NB if this matches multiple documents for a single sentence, we only return one here
        """
        sentences = [s.replace("'", "''") for s in sentences]

        query_string = f"""SELECT document_uri, sentence FROM "default"."{self.table_name}" 
                           WHERE sentence IN {self._form_query_tuple(sentences)}
                        """

        results = self.get_query_results(query_string=query_string)

        sentence_to_uri = {}

        for r in results:
            sentence_to_uri[r["sentence"]] = r["document_uri"]

        return sentence_to_uri

    def get_article_from_document_uri(self, document_uris: List[str]) -> Dict[str, str]:

        query_string = f"""SELECT document_uri,  FROM "default"."{self.table_name}" 
                                  WHERE sentence IN {self._form_query_tuple(sentences)}
                               """

        results = self.get_query_results(query_string=query_string)

        uri_to_article = {}

        for r in results:
            uri_to_article[r["sentence"]] = r["document_uri"]

        return sentence_to_uri


    def get_query_results(self, query_string: str):

        response = self.client.start_query_execution(
            QueryString=query_string,
            QueryExecutionContext={"Database": "default"},
            ResultConfiguration={"OutputLocation": self.output_location},
        )

        query = AthenaQueryReader(query_execution_id=response["QueryExecutionId"], client=self.client, loud=self.loud)

        for row in query.stream_query_results():
            yield row

    def _form_query_tuple(self, lst: List[str]) -> str:
        return "('" + "','".join(lst) + "')"

    def _add_brackets_if_needed(self, relation: str) -> str:
        if relation[0] != "[":
            relation = f"[{relation}]"
        return relation


if __name__ == "__main__":
    athena = AthenaQueries(table_name="fact_postprocessing_output")

    results = athena.get_document_uri_from_sentence(sentences=['From the November 24, 1938 Mar-Ken Journal: Harris Berger and David Gorcey are featured in this picture [Universal’s “Newsboys’ Home”].'])

    print(results)