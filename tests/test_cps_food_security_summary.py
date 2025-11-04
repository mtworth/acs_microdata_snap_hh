import io
import json
import unittest
from unittest import mock

import cps_food_security_summary as summary


class BuildRequestUrlTests(unittest.TestCase):
    def test_builds_expected_url(self):
        url = summary.build_request_url(
            "abc123", year=2023, state="06", fields=("FSSTATUS", "FSWGT")
        )
        self.assertIn("https://api.census.gov/data/2023/cps/fss?", url)
        self.assertIn("get=FSSTATUS,FSWGT", url)
        self.assertIn("for=state:06", url)
        self.assertTrue(url.endswith("key=abc123"))

    def test_requires_api_key(self):
        with self.assertRaises(ValueError):
            summary.build_request_url("", state="06")

    def test_requires_state(self):
        with self.assertRaises(ValueError):
            summary.build_request_url("abc", state="  ")


class FetchRecordsTests(unittest.TestCase):
    def make_response(self, payload, status=200):
        buffer = io.BytesIO(json.dumps(payload).encode("utf-8"))

        class MockResponse(io.BytesIO):
            def __init__(self, buf, status_code):
                super().__init__(buf.getvalue())
                self.status = status_code

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()

        return MockResponse(buffer, status)

    def test_fetch_records_success(self):
        payload = [["FSSTATUS", "FSWGT", "state"], ["1", "100.0", "06"]]
        response = self.make_response(payload)
        with mock.patch("urllib.request.urlopen", return_value=response):
            records = summary.fetch_cps_fss_records("abc", state="06")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["FSSTATUS"], "1")

    def test_fetch_records_http_error(self):
        response = self.make_response([], status=500)
        with mock.patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(summary.CensusApiError):
                summary.fetch_cps_fss_records("abc", state="06")

    def test_fetch_records_url_error(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=summary.error.URLError("boom"),
        ):
            with self.assertRaises(summary.CensusApiError):
                summary.fetch_cps_fss_records("abc", state="06")


class SummarizeFoodSecurityTests(unittest.TestCase):
    def test_summary_with_valid_records(self):
        records = [
            {"FSSTATUS": "1", "FSWGT": "101.4"},
            {"FSSTATUS": "3", "FSWGT": "50"},
            {"FSSTATUS": "4", "FSWGT": "25.6"},
        ]
        result = summary.summarize_food_security(records)
        totals = {row.status_code: row.weighted_households for row in result.rows}
        self.assertEqual(totals["1"], 101)
        self.assertEqual(totals["3"], 50)
        self.assertEqual(totals["4"], 26)
        self.assertEqual(result.total_households, 177)

    def test_summary_ignores_invalid_weights(self):
        records = [
            {"FSSTATUS": "1", "FSWGT": ""},
            {"FSSTATUS": "2", "FSWGT": None},
            {"FSSTATUS": "3", "FSWGT": "bad"},
            {"FSSTATUS": "4", "FSWGT": "10"},
        ]
        result = summary.summarize_food_security(records)
        totals = {row.status_code: row.weighted_households for row in result.rows}
        self.assertEqual(totals["4"], 10)
        self.assertEqual(result.total_households, 10)

    def test_summary_tracks_suppressed_codes(self):
        records = [
            {"FSSTATUS": "1", "FSWGT": "100"},
            {"FSSTATUS": "9", "FSWGT": "25"},
        ]
        result = summary.summarize_food_security(records)
        self.assertEqual(result.total_households, 100)
        self.assertEqual(result.suppressed_weight, 25)


if __name__ == "__main__":
    unittest.main()
