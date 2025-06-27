import io
import json
import zipfile

import pytest

import scripts.sgd_crawler as crawler


class MockResponse:
    def __init__(self, content):
        self.content = content.encode("utf-8") if isinstance(content, str) else content

    def raise_for_status(self):
        pass


def test_get_year_links(monkeypatch):
    html = "<a href='1999/'>1999</a><a href='foo'>foo</a><a href='2000/'>2000</a>"

    def mock_get(url, headers=None):
        return MockResponse(html)

    monkeypatch.setattr(crawler.requests, "get", mock_get)
    years = crawler.get_year_links()
    assert years == [f"{crawler.BASE_URL}/1999/", f"{crawler.BASE_URL}/2000/"]


def test_get_work_links(monkeypatch):
    html = "<a href='123/'>123</a><a href='bar'>bar</a><a href='456/'>456</a>"

    def mock_get(url, headers=None):
        return MockResponse(html)

    monkeypatch.setattr(crawler.requests, "get", mock_get)
    works = crawler.get_work_links("https://example.com/1999/")
    assert works == ["https://example.com/1999/123/", "https://example.com/1999/456/"]


def test_process_work(monkeypatch):
    xml_content = "<root><p>Hello</p></root>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ocr/test.xml", xml_content)

    def mock_get(url, headers=None, stream=True):
        return MockResponse(buf.getvalue())

    monkeypatch.setattr(crawler.requests, "get", mock_get)
    docs = crawler.process_work("https://example.com/work/1/")
    assert docs[0]["content"] == "Hello"


def test_push_batches_to_hub(monkeypatch, tmp_path):
    data = {"URL": "u", "content": "c", "source": "s"}
    batch = tmp_path / "batch.jsonl"
    with open(batch, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.write("\n")

    pushed = {}

    class DummyDS:
        def __init__(self, *args, **kwargs):
            pass
        
        def push_to_hub(self, *args, **kwargs):
            pushed["called"] = True

        def __len__(self):
            return 1

    monkeypatch.setattr(crawler, "Dataset", DummyDS)
    monkeypatch.setattr(crawler, "concatenate_datasets", lambda x: DummyDS())
    monkeypatch.setattr(crawler, "Features", lambda *a, **k: None)
    monkeypatch.setattr(crawler, "Value", lambda *a, **k: None)

    assert crawler.push_batches_to_hub([str(batch)])
    assert pushed.get("called")

