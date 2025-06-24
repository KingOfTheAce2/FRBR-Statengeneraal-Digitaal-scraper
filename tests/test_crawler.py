import builtins
import types
from bs4 import BeautifulSoup
import crawler_for_sgd as crawler


def test_strip_xml():
    xml = b"<root><a>hello</a>&amp;<b>world</b></root>"
    assert crawler.BaseCrawler.strip_xml(xml) == "hello & world"


def test_pagination(monkeypatch):
    pages = [
        BeautifulSoup('<a href="/frbr/sgd/area1"></a>', 'lxml'),
        BeautifulSoup('', 'lxml'),
    ]
    calls = []

    def fake_fetch(path):
        calls.append(path)
        return pages.pop(0)

    sg = crawler.SGDCrawler()
    monkeypatch.setattr(sg, 'fetch_soup', fake_fetch)
    result = list(sg.iter_subarea_paths())
    assert result == ['/frbr/sgd/area1']
    assert calls == ['/frbr/sgd', '/frbr/sgd?start=11']

