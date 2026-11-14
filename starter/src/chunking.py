import re
from typing import Any
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    if not page.text:
        return []

    text = page.text
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    chunk_idx = 0

    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()

        if len(chunk_text) < 30 and chunks:
            continue

        chunk_id = f"{page.document_id}_A_p{page.page:03d}_c{chunk_idx:03d}"

        chunk_meta: dict[str, Any] = dict(page.metadata)
        chunk_meta.update({
            "strategy": "A",
            "chunk_index": chunk_idx,
            "char_start": start,
            "char_end": end,
            "char_length": len(chunk_text),
            "window_size": size,
            "window_overlap": overlap,
        })

        chunk_obj = Chunk(
            chunk_id=chunk_id,
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=chunk_text,
            metadata=chunk_meta,
        )
        chunks.append(chunk_obj)
        chunk_idx += 1

        if end >= len(text):
            break

    return chunks


def _is_heading(text: str) -> bool:
    t = text.strip()
    return (
        len(t) < 65
        and not t.endswith((".", "!", "?", ':"', '."', "!'", "?'"))
        and (t[:1].isupper() or bool(re.match(r"^\d+\.", t)))
    )


def _split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])", text)
    return [s.strip() for s in sentences if s.strip()]


def by_paragraph(
    page: DocumentPage,
    max_size: int = 800,
    strategy: str = "B",
) -> list[Chunk]:
    if not page.text:
        return []

    raw_paras = [p.strip() for p in page.text.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    chunk_idx = 0
    current_unit_buffer: list[str] = []
    current_length = 0

    def flush_buffer() -> None:
        nonlocal current_unit_buffer, current_length, chunk_idx
        if not current_unit_buffer:
            return
        chunk_text = "\n\n".join(current_unit_buffer).strip()
        if not chunk_text:
            return

        chunk_id = f"{page.document_id}_{strategy}_p{page.page:03d}_c{chunk_idx:03d}"
        chunk_meta: dict[str, Any] = dict(page.metadata)
        chunk_meta.update({
            "strategy": strategy,
            "chunk_index": chunk_idx,
            "char_length": len(chunk_text),
            "max_size_limit": max_size,
            "paragraphs_merged": len(current_unit_buffer),
        })

        chunk_obj = Chunk(
            chunk_id=chunk_id,
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=chunk_text,
            metadata=chunk_meta,
        )
        chunks.append(chunk_obj)
        chunk_idx += 1
        current_unit_buffer = []
        current_length = 0

    for u in raw_paras:
        u_len = len(u)
        is_hd = _is_heading(u)

        if is_hd and current_length > max_size * 0.4 and current_unit_buffer:
            flush_buffer()

        if u_len > max_size:
            heading_prefix = ""
            if current_unit_buffer and all(_is_heading(b) for b in current_unit_buffer):
                heading_prefix = "\n\n".join(current_unit_buffer) + "\n\n"
                current_unit_buffer = []
                current_length = 0
            else:
                flush_buffer()

            sentences = _split_into_sentences(u)
            s_buf: list[str] = []
            s_len = 0
            is_first = True

            for s in sentences:
                cur_sentence = (heading_prefix + s) if is_first and heading_prefix else s
                if s_len + len(cur_sentence) + 1 > max_size and s_buf:
                    sub_text = " ".join(s_buf).strip()
                    sub_id = f"{page.document_id}_{strategy}_p{page.page:03d}_c{chunk_idx:03d}"
                    sub_meta = dict(page.metadata)
                    sub_meta.update({
                        "strategy": strategy,
                        "chunk_index": chunk_idx,
                        "char_length": len(sub_text),
                        "max_size_limit": max_size,
                        "split_type": "sentence_boundary_oversized",
                    })
                    chunks.append(Chunk(
                        chunk_id=sub_id,
                        document_id=page.document_id,
                        source=page.source,
                        page=page.page,
                        text=sub_text,
                        metadata=sub_meta,
                    ))
                    chunk_idx += 1
                    s_buf = [s]
                    s_len = len(s)
                else:
                    s_buf.append(cur_sentence)
                    s_len += len(cur_sentence) + 1

                if is_first:
                    is_first = False
                    heading_prefix = ""

            if s_buf:
                sub_text = " ".join(s_buf).strip()
                sub_id = f"{page.document_id}_{strategy}_p{page.page:03d}_c{chunk_idx:03d}"
                sub_meta = dict(page.metadata)
                sub_meta.update({
                    "strategy": strategy,
                    "chunk_index": chunk_idx,
                    "char_length": len(sub_text),
                    "max_size_limit": max_size,
                    "split_type": "sentence_boundary_oversized",
                })
                chunks.append(Chunk(
                    chunk_id=sub_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=sub_text,
                    metadata=sub_meta,
                ))
                chunk_idx += 1

        elif current_length + u_len + 2 > max_size and current_unit_buffer:
            flush_buffer()
            current_unit_buffer.append(u)
            current_length = u_len
        else:
            current_unit_buffer.append(u)
            current_length += u_len + 2

    flush_buffer()
    return chunks
