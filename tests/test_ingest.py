import re

from agent.ingest import MAX_CHUNK_CHARS, clean_markdown, parse_front_matter


def by_doc(chunks, doc_id):
    return [c for c in chunks if c.metadata["doc_id"] == doc_id]


def test_all_ten_pdfs_are_ingested(chunks):
    doc_ids = {c.metadata["doc_id"] for c in chunks}
    assert len(doc_ids) == 10
    assert "promo-rentree-2024" in doc_ids


def test_every_chunk_has_required_metadata(chunks):
    for c in chunks:
        for key in ("doc_id", "title", "status", "audience", "updated", "source_file", "chunk_id"):
            assert c.metadata[key], f"{key} manquant sur {c.metadata.get('chunk_id')}"
        assert c.metadata["status"] in {"current", "deprecated"}
        assert c.metadata["audience"] in {"public", "internal"}


def test_chunk_ids_are_unique(chunks):
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_deprecated_and_internal_flags_come_from_front_matter(chunks):
    assert {c.metadata["status"] for c in by_doc(chunks, "promo-rentree-2024")} == {"deprecated"}
    assert {c.metadata["audience"] for c in by_doc(chunks, "politique-geste-commercial")} == {"internal"}
    assert {c.metadata["audience"] for c in by_doc(chunks, "procedure-escalade-n2")} == {"internal"}
    assert {c.metadata["audience"] for c in by_doc(chunks, "grille-tarifaire-2026")} == {"public"}


def test_extraction_noise_is_removed(chunks):
    for c in chunks:
        text = c.page_content
        assert "CO R P U S" not in text
        assert not re.search(r"corpus/[\w-]+\.md", text)
        assert not re.search(r"^\d+ / \d+$", text, re.MULTILINE)
        assert "cookies" not in text
        assert "Mentions légales" not in text


def test_tariff_table_stays_in_one_chunk_with_its_prices(chunks):
    fibre = [c for c in by_doc(chunks, "grille-tarifaire-2026") if "Offres fibre" in c.metadata["section"]]
    assert len(fibre) == 1
    text = fibre[0].page_content
    assert "29,99" in text and "39,99" in text
    assert "|---|" in text


def test_chunks_are_bounded_and_carry_their_context(chunks):
    for c in chunks:
        assert len(c.page_content) <= MAX_CHUNK_CHARS + 300
        assert c.page_content.startswith(f"[{c.metadata['title']}]")


def test_section_spanning_a_page_break_is_not_cut(chunks):
    voyants = [c for c in by_doc(chunks, "faq-box-internet") if "Vérifications complémentaires" in c.metadata["section"]]
    text = " ".join(c.page_content for c in voyants)
    assert "Jarretière optique" in text and "Prise optique (PTO)" in text


def test_parse_front_matter_reads_all_fields():
    md = "**id** doc-x **version** 2026.2 **statut** current **public** internal **maj** 2026-06-01\n\nCorps"
    fields, rest = parse_front_matter(md)
    assert fields == {"id": "doc-x", "version": "2026.2", "statut": "current", "public": "internal", "maj": "2026-06-01"}
    assert rest.strip() == "Corps"


def test_clean_markdown_drops_repeated_running_title():
    md = "# Mon titre\n\nTexte\n\n1 / 2\n\nMon titre\n\nSuite"
    cleaned = clean_markdown(md, "Mon titre")
    assert cleaned.count("Mon titre") == 1
    assert "1 / 2" not in cleaned
