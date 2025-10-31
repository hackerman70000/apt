import pytest
from apt.ingest.metadata import extract_apt_mentions, extract_technique_mentions

class TestExtractAPTMentions:
    def test_extract_apt_numbers(self):
        text = "Report on APT28 and APT29 activities"
        result = extract_apt_mentions(text)
        assert "APT28" in result
        assert "APT29" in result

    def test_extract_apt_with_hyphen(self):
        text = "APT-28 conducted operation"
        result = extract_apt_mentions(text)
        assert "APT-28" in result

    def test_extract_known_groups(self):
        text = "Lazarus Group and Kimsuky were observed"
        result = extract_apt_mentions(text)
        assert "LAZARUS" in result
        assert "KIMSUKY" in result

    def test_case_insensitive(self):
        text = "apt28 and Fancy Bear activity"
        result = extract_apt_mentions(text)
        assert "APT28" in result
        assert "FANCYBEAR" in result

    def test_no_mentions(self):
        text = "Generic malware analysis without APT attribution"
        result = extract_apt_mentions(text)
        assert len(result) == 0

class TestExtractTechniqueMentions:
    def test_extract_technique_basic(self):
        text = "Used T1566 for initial access"
        result = extract_technique_mentions(text)
        assert "T1566" in result

    def test_extract_technique_with_subtechnique(self):
        text = "Spearphishing via T1566.001 observed"
        result = extract_technique_mentions(text)
        assert "T1566.001" in result

    def test_extract_multiple_techniques(self):
        text = "Techniques T1566.001, T1059.001, and T1204 were used"
        result = extract_technique_mentions(text)
        assert "T1566.001" in result
        assert "T1059.001" in result
        assert "T1204" in result

    def test_no_techniques(self):
        text = "Generic threat report without technique IDs"
        result = extract_technique_mentions(text)
        assert len(result) == 0
