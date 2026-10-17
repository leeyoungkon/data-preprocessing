"""TXT and text-based PDF input. PDF page numbers are physical, 1-based."""
import hashlib
from pathlib import Path


def source_digest(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        # Preserve hashes from the original TXT-only lab, including BOM removal.
        text = path.read_text(encoding="utf-8-sig")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_source(path: Path) -> list[tuple[str, str]]:
    """Return (source label, extracted text) units; validate the whole file first."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            raise RuntimeError(f"빈 TXT 문서입니다: {path.name}")
        return [(path.name, text)]
    if suffix != ".pdf":
        raise RuntimeError(f"지원하지 않는 확장자: {path.name}")

    from pypdf import PdfReader

    units = []
    try:
        with path.open("rb") as stream:
            reader = PdfReader(stream)
            if reader.is_encrypted:
                raise RuntimeError(f"암호화된 PDF입니다: {path.name}. 암호를 해제한 사본을 사용하세요.")
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    # A truly empty page without a content stream can be skipped.
                    # Do not silently skip scans, illustrations or unreadable pages.
                    if page.get_contents() is None:
                        print(f"빈 페이지 건너뜀: {path.name} / {number}페이지", flush=True)
                        continue
                    raise RuntimeError(
                        f"텍스트를 추출할 수 없습니다: {path.name} / {number}페이지. "
                        "스캔·그림 페이지는 OCR이 필요합니다. 불필요한 빈 페이지라면 제거하세요."
                    )
                source = f"{path.name}#page={number}"
                # Keep otherwise identical pages distinct and retain provenance.
                content = f"[출처: {source}]\n\n{text}"
                units.append((source, content))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"PDF 읽기 실패: {path.name}: {exc}") from exc
    if not units:
        raise RuntimeError(f"추출된 텍스트가 없는 PDF입니다: {path.name}. OCR 처리를 확인하세요.")
    return units
