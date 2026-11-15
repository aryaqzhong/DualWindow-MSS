"""Source identity, license, archive and cache checks for the public demo."""
import io
import json
import zipfile

import pytest

from scripts import fetch_demo_audio as sources


def metadata():
    def item(file_id, nickname, extension="flac"):
        return {"file_id": file_id, "file_name": f"source_{file_id}.{extension}", "file_nicname": nickname,
                "download_url": f"https://ccmixter.org/content/artist/source_{file_id}.{extension}",
                "file_format_info": {"default-ext": extension}}
    return {"upload_id": 7, "upload_name": "Example", "user_name": "artist", "license_name": "Attribution (3.0)",
            "license_url": "http://creativecommons.org/licenses/by/3.0/", "file_page_url": "https://ccmixter.org/files/artist/7",
            "files": [item(10, "Full mix"), item(11, "Vocal"), item(12, "Synth"), item(13, "Synth"),
                      item(14, "Stems", "zip")]}


def roles():
    return {"mix": ["Full mix"], "vocals": ["Vocal"], "accompaniment": [{"file_id": 12}, {"file_id": 13}]}


@pytest.mark.parametrize("url", ["https://creativecommons.org/licenses/by-nc/3.0/",
                                "https://creativecommons.org/licenses/by-sa/4.0/",
                                "https://creativecommons.org/licenses/by-nd/3.0/",
                                "https://not-creativecommons.org/licenses/by/3.0/"])
def test_reject_non_by_or_wrong_license_host(url):
    meta = metadata()
    meta["license_url"] = url
    with pytest.raises(ValueError, match="Expected CC BY"):
        sources.resolve_files(meta, roles())


def test_duplicate_nicknames_require_distinct_file_identity():
    selected = sources.resolve_files(metadata(), roles())
    assert [s["file"]["file_id"] for s in selected if s["role"] == "accompaniment"] == [12, 13]
    ambiguous = roles()
    ambiguous["accompaniment"] = ["Synth"]
    with pytest.raises(ValueError, match="exactly once, got 2"):
        sources.resolve_files(metadata(), ambiguous)


def test_accompaniment_cannot_reuse_mix_or_be_missing():
    wrong = roles()
    wrong["accompaniment"] = ["Full mix"]
    with pytest.raises(ValueError, match="same source"):
        sources.resolve_files(metadata(), wrong)
    wrong.pop("accompaniment")
    with pytest.raises(ValueError, match="roles"):
        sources.resolve_files(metadata(), wrong)


def test_zip_requires_exact_member_and_direct_audio_rejects_member():
    selected = roles()
    selected["accompaniment"] = [{"file_id": 14}]
    with pytest.raises(ValueError, match="ZIP requires"):
        sources.resolve_files(metadata(), selected)
    selected["accompaniment"] = [{"file_id": 12, "member": "synth.flac"}]
    with pytest.raises(ValueError, match="direct audio"):
        sources.resolve_files(metadata(), selected)


def test_fetch_selected_archive_members_keeps_packaging_and_roles_separate(tmp_path, monkeypatch):
    meta = metadata()
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("project/vocal.flac", b"vocal bytes")
        archive.writestr("project/synth.flac", b"synth bytes")
    selected = {"mix": ["Full mix"], "vocals": [{"file_id": 14, "member": "project/vocal.flac"}],
                "accompaniment": [{"file_id": 14, "member": "project/synth.flac"}]}
    requests = []
    def get(url, referer):
        requests.append(url)
        if "/api/" in url:
            return json.dumps([meta]).encode()
        return archive_bytes.getvalue() if url.endswith(".zip") else b"mix bytes"
    monkeypatch.setattr(sources.time, "sleep", lambda _: None)
    provenance = sources.fetch_song(7, "example", selected, tmp_path, get)
    assert (tmp_path / "example/vocals_0.flac").read_bytes() == b"vocal bytes"
    assert (tmp_path / "example/accompaniment_0.flac").read_bytes() == b"synth bytes"
    assert sum(url.endswith(".zip") for url in requests) == 1
    assert provenance["files"]["vocals_0.flac"]["archive_member"] == "project/vocal.flac"
    assert provenance["files"]["vocals_0.flac"]["archive_sha256"] == provenance["files"]["accompaniment_0.flac"]["archive_sha256"]
    # A second fetch verifies role hashes and needs only current upload metadata.
    requests.clear()
    assert sources.fetch_song(7, "example", selected, tmp_path, get) == provenance
    assert len(requests) == 1
    # An API ID change at an unchanged URL/nickname must not relabel cached audio.
    meta["files"][0]["file_id"] = 999
    with pytest.raises(ValueError, match="recorded selection"):
        sources.fetch_song(7, "example", selected, tmp_path, get)
    meta["files"][0]["file_id"] = 10
    (tmp_path / "example/vocals_0.flac").write_bytes(b"changed")
    with pytest.raises(ValueError, match="recorded selection"):
        sources.fetch_song(7, "example", selected, tmp_path, get)


def test_wrong_upload_identity_fails_before_audio_download(tmp_path):
    meta = metadata()
    meta["upload_id"] = 8
    with pytest.raises(ValueError, match="requested upload"):
        sources.fetch_song(7, "example", roles(), tmp_path, lambda *args: json.dumps([meta]).encode())


def test_cached_original_url_and_hash_are_verified(tmp_path, monkeypatch):
    f = metadata()["files"][0]
    monkeypatch.setattr(sources.time, "sleep", lambda _: None)
    path = sources.source_file(tmp_path, f, "https://ccmixter.org/", lambda *args: b"source bytes")
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        sources.source_file(tmp_path, f, "https://ccmixter.org/", lambda *args: b"unreached")
