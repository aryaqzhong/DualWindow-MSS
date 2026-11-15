"""Download the CC BY songs used by the listening demo (ccMixter uploads with aligned vocal stems).

Each catalog song supplies the full mix, vocal stems and original instrumental/accompaniment stems.
Licenses are Creative Commons Attribution (CC BY 2.5 / 3.0), which allows remixing and redistribution with
attribution; attribution data is written next to the audio (provenance.json) and shown on the demo page.
ccMixter files are hotlink-protected, so each request carries the upload page as referer.

Usage: python -I scripts/fetch_demo_audio.py [--out data/demo_ccmixter]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

# Selectors are exact nicknames, or {file_id/file_name, member?} for ambiguous names/ZIP stems.
# Archive members are written to generated role filenames, never extracted to source-supplied paths.
SONGS = {
    22364: ("persephone", {"mix": ["Secret Mixer Remix"], "vocals": ["Lead Vox", "Back Vox", "BVox Last Chorus"],
                           "accompaniment": ["Piano", "Loop"]}),
    55541: ("falling", {"mix": [{"file_id": 89704}],
                        "vocals": [{"file_id": 89706}, {"file_id": 89707}],
                        "accompaniment": [{"file_id": 89710}, {"file_id": 89709}, {"file_id": 89708},
                                          {"file_id": 89711}, {"file_id": 89712}]}),
    68969: ("chill_relax", {"mix": ["Main Mix"], "vocals": ["Dry main acapella", "dry backing"],
                            "accompaniment": ["Instrumental"]}),
    # Select each FX performance once; the ZIP also has DRY alternatives, which are not summed in.
    60536: ("that_is_that", {"mix": [{"file_id": 100654}], "vocals": [
        {"file_id": 100655, "member": "Kara Square - That is That (130 BPM)/MainVOX with FX.flac"},
        {"file_id": 100655, "member": "Kara Square - That is That (130 BPM)/BackupVOX with FX.flac"},
    ], "accompaniment": [
        {"file_id": 100655, "member": "Kara Square - That is That (130 BPM)/Uke with FX.flac"},
        {"file_id": 100655, "member": "Kara Square - That is That (130 BPM)/Claps.flac"},
    ]}),
}
UA = "Mozilla/5.0 (DualWindow-MSS listening demo; one-time download)"
DISPLAY_ARTISTS = {60536: "Kara Square"}  # Public credit on the upload; account handle is mindmapthat.


def get(url: str, referer: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_cc_by(meta: dict):
    license_url = urlsplit(meta.get("license_url", ""))
    if (license_url.scheme not in {"http", "https"} or license_url.hostname != "creativecommons.org"
            or not re.fullmatch(r"/licenses/by/\d+(?:\.\d+)?/", license_url.path)
            or license_url.query or license_url.fragment):
        raise ValueError(f"Expected CC BY, got {meta.get('license_url')!r}")


def resolve_files(meta: dict, roles: dict, required_roles=("mix", "vocals", "accompaniment")) -> list[dict]:
    require_cc_by(meta)
    if set(roles) - {"mix", "vocals", "accompaniment"} or any(not roles.get(role) for role in required_roles):
        raise ValueError("Required original mix, vocal and accompaniment roles must all be populated")
    if len(roles.get("mix", [])) != 1:
        raise ValueError("Exactly one original full mix is required")
    result, seen = [], set()
    for role, selectors in roles.items():
        for index, selector in enumerate(selectors):
            selector = {"nickname": selector} if isinstance(selector, str) else dict(selector)
            if not set(selector) <= {"nickname", "file_id", "file_name", "member"}:
                raise ValueError(f"Unknown file selector: {selector}")
            keys = set(selector) - {"member"}
            if not keys:
                raise ValueError("A selector must identify an upload file")
            matches = [f for f in meta["files"] if all(
                f.get("file_nicname" if key == "nickname" else key) == selector[key] for key in keys)]
            if len(matches) != 1:
                raise ValueError(f"File selector must match exactly once, got {len(matches)}: {selector}")
            f = matches[0]
            member = selector.get("member")
            if member is not None and (not isinstance(member, str) or not member):
                raise ValueError("ZIP member must be an exact nonempty path")
            is_zip = f["file_format_info"]["default-ext"].lower() == "zip"
            if is_zip != (member is not None):
                raise ValueError("A ZIP requires an explicit member, and direct audio cannot select a member")
            identity = (f["download_url"], member)
            if identity in seen:
                raise ValueError("The same source file/member cannot fill two roles")
            seen.add(identity)
            ext = Path(member).suffix.lstrip(".").lower() if member else f["file_format_info"]["default-ext"].lower()
            if ext not in {"flac", "wav", "mp3", "ogg", "aif", "aiff"}:
                raise ValueError(f"Expected an audio source, got .{ext}")
            result.append({"role": role, "index": index, "file": f, "member": member,
                           "name": f"{role}_{index}.{ext}"})
    return result


def source_file(directory: Path, file: dict, page: str, getter=None) -> Path:
    """Cache a complete original download with a URL-bound checksum sidecar."""
    getter = getter or get
    directory.mkdir(parents=True, exist_ok=True)
    url = file["download_url"]
    if urlsplit(url).hostname not in {"ccmixter.org", "www.ccmixter.org"}:
        raise ValueError(f"Unexpected download origin: {url}")
    path = directory / (str(file["file_id"]) + "_" + Path(urlsplit(url).path).name)
    sidecar = path.with_suffix(path.suffix + ".json")
    if path.exists():
        if not sidecar.is_file():
            raise ValueError(f"Cached download has no provenance sidecar: {path}")
        record = json.loads(sidecar.read_text())
        if record != {"url": url, "sha256": sha256(path)}:
            raise ValueError(f"Cached original download changed: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(getter(url, page))
        temporary.replace(path)
        sidecar.write_text(json.dumps({"url": url, "sha256": sha256(path)}, indent=2) + "\n")
        time.sleep(1)
    return path


def fetch_song(uid: int, slug: str, roles: dict, out: Path, getter=None):
    getter = getter or get
    rows = json.loads(getter(f"https://ccmixter.org/api/query?f=json&ids={uid}", "https://ccmixter.org/"))
    if len(rows) != 1 or int(rows[0].get("upload_id", -1)) != uid:
        raise ValueError("The API did not return the requested upload identity")
    meta = rows[0]
    selected = resolve_files(meta, roles)
    display_artist = DISPLAY_ARTISTS.get(uid, meta["user_name"])
    if uid in DISPLAY_ARTISTS and display_artist != meta.get("user_real_name"):
        raise ValueError("The verified public artist credit differs from current upload metadata")
    d = out / slug
    d.mkdir(parents=True, exist_ok=True)
    prior = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").exists() else None
    if prior is not None and int(prior["upload_id"]) != uid:
        raise ValueError("Existing song directory identifies a different upload")
    prov = {"upload_id": uid, "title": meta["upload_name"], "artist": display_artist,
            "artist_account": meta["user_name"],
            "artist_real_name": meta.get("user_real_name", ""),
            "featuring": (meta.get("upload_extra") or {}).get("featuring", ""), "license": meta["license_name"],
            "license_url": meta["license_url"], "page": meta["file_page_url"],
            "downloaded": prior.get("downloaded", date.today().isoformat()) if prior else date.today().isoformat(),
            "metadata_verified": date.today().isoformat(),
            "files": {}}
    archives_checked = set()
    for selection in selected:
        f, name, member = selection["file"], selection["name"], selection["member"]
        path = d / name
        record = {"role": selection["role"], "nickname": f["file_nicname"], "source": f["download_url"],
                  "source_file_id": f["file_id"], "source_file_name": f.get("file_name"), "format": f["file_format_info"]}
        if member:
            record["archive_member"] = member
        old = (prior or {}).get("files", {}).get(name)
        if path.exists() and old is not None:
            keys = ("role", "nickname", "source", "archive_member")
            identity_changed = any(key in old and old[key] != record.get(key)
                                   for key in ("source_file_id", "source_file_name"))
            if identity_changed or any(old.get(key) != record.get(key) for key in keys) or old.get("sha256") != sha256(path):
                raise ValueError(f"Existing role asset differs from its recorded selection: {path}")
            if member:
                record["archive_sha256"] = old["archive_sha256"]
        else:
            original = source_file(d / "source_downloads", f, meta["file_page_url"], getter)
            if member:
                with zipfile.ZipFile(original) as archive:
                    if original not in archives_checked:
                        corrupt = archive.testzip()
                        if corrupt is not None:
                            raise ValueError(f"Corrupt ZIP member: {corrupt}")
                        archives_checked.add(original)
                    if sum(info.filename == member for info in archive.infolist()) != 1:
                        raise ValueError(f"Expected exactly one ZIP member: {member}")
                    data = archive.read(member)
                record["archive_sha256"] = sha256(original)
            else:
                data = original.read_bytes()
            if path.exists() and sha256(path) != hashlib.sha256(data).hexdigest():
                raise ValueError(f"Unrecorded existing role asset differs from the selected original: {path}")
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(path)
        record["sha256"] = sha256(path)
        prov["files"][name] = record
        print(f"{slug}: {name} <- {f['file_nicname']}" + (f" :: {member}" if member else ""), flush=True)
    temporary = d / "provenance.json.tmp"
    temporary.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(d / "provenance.json")
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_ccmixter")
    ap.add_argument("--songs", nargs="+", choices=[song[0] for song in SONGS.values()],
                    help="download only these catalog songs")
    args = ap.parse_args()
    out = Path(args.out)
    for uid, (slug, roles) in SONGS.items():
        if args.songs is None or slug in args.songs:
            fetch_song(uid, slug, roles, out)


if __name__ == "__main__":
    main()
