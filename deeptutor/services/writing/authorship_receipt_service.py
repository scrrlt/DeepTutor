"""Hash-chained authorship receipt persistence helpers."""

from __future__ import annotations

import hashlib
import time
from typing import Literal, cast

from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from deeptutor.models.text_mutation_receipt import TextMutationReceipt
from deeptutor.services.writing.types import EditSource

_MAX_APPEND_RETRIES = 3
_ALLOWED_EDIT_SOURCES: set[str] = {"manual_typing", "api_scrub", "context_expansion"}


class MutationReceiptAppendError(RuntimeError):
    """Raised when append_mutation_receipt cannot persist a valid next event."""


class MutationReceiptRecord(BaseModel):
    """Strict Pydantic contract for persisted mutation receipts."""

    id: int
    user_id: str
    document_id: str
    sequence_index: int
    source_type: EditSource
    character_delta_count: int
    timestamp_epoch: float
    previous_block_hash: str
    signature_hash: str


class MutationChainVerificationResult(BaseModel):
    """Strict Pydantic contract for chain verification outcomes."""

    user_id: str
    document_id: str
    checked_count: int
    is_valid: bool
    first_invalid_sequence_index: int | None = None
    reason: str | None = None



def _signature_hash(
    *,
    sequence_index: int,
    source_type: EditSource,
    character_delta_count: int,
    timestamp_epoch: float,
    previous_block_hash: str,
) -> str:
    payload = (
        f"{sequence_index}:{source_type}:{character_delta_count}:"
        f"{timestamp_epoch}:{previous_block_hash}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _to_record(row: TextMutationReceipt) -> MutationReceiptRecord:
    source_value = row.source_type
    if source_value not in _ALLOWED_EDIT_SOURCES:
        source_value = "manual_typing"
    return MutationReceiptRecord(
        id=row.id,
        user_id=row.user_id,
        document_id=row.document_id,
        sequence_index=row.sequence_index,
        source_type=cast(EditSource, source_value),
        character_delta_count=row.character_delta_count,
        timestamp_epoch=row.timestamp_epoch,
        previous_block_hash=row.previous_block_hash,
        signature_hash=row.signature_hash,
    )


async def append_mutation_receipt(
    db: AsyncSession,
    *,
    user_id: str,
    document_id: str,
    source_type: EditSource,
    character_delta_count: int,
) -> MutationReceiptRecord:
    """Append one mutation receipt to the per-document hash chain."""
    last_error: IntegrityError | None = None
    for _ in range(_MAX_APPEND_RETRIES):
        prior = (
            await db.execute(
                select(TextMutationReceipt)
                .where(TextMutationReceipt.user_id == user_id)
                .where(TextMutationReceipt.document_id == document_id)
                .order_by(desc(TextMutationReceipt.sequence_index))
                .limit(1)
            )
        ).scalar_one_or_none()

        sequence_index = 0 if prior is None else (prior.sequence_index + 1)
        previous_hash = "0" * 64 if prior is None else prior.signature_hash
        timestamp_epoch = time.time()
        signature = _signature_hash(
            sequence_index=sequence_index,
            source_type=source_type,
            character_delta_count=character_delta_count,
            timestamp_epoch=timestamp_epoch,
            previous_block_hash=previous_hash,
        )

        row = TextMutationReceipt(
            user_id=user_id,
            document_id=document_id,
            sequence_index=sequence_index,
            source_type=source_type,
            character_delta_count=character_delta_count,
            timestamp_epoch=timestamp_epoch,
            previous_block_hash=previous_hash,
            signature_hash=signature,
        )
        try:
            async with db.begin_nested():
                db.add(row)
                await db.flush()
                await db.refresh(row)
            return _to_record(row)
        except IntegrityError as exc:
            last_error = exc

    if last_error is not None:
        raise MutationReceiptAppendError(
            "Failed to append mutation receipt after retries"
        ) from last_error

    raise MutationReceiptAppendError("Failed to append mutation receipt")


async def list_mutation_receipts(
    db: AsyncSession,
    *,
    user_id: str,
    document_id: str,
    limit: int = 200,
) -> list[MutationReceiptRecord]:
    """Return ordered mutation receipts for one user document."""
    rows = (
        (
            await db.execute(
                select(TextMutationReceipt)
                .where(TextMutationReceipt.user_id == user_id)
                .where(TextMutationReceipt.document_id == document_id)
                .order_by(TextMutationReceipt.sequence_index.asc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [_to_record(row) for row in rows]


def _verify_row_chain(
    row: MutationReceiptRecord,
    expected_previous_hash: str,
    expected_sequence_index: int,
) -> tuple[bool, str | None]:
    if row.sequence_index != expected_sequence_index:
        return False, "non_contiguous_sequence"

    if row.previous_block_hash != expected_previous_hash:
        return False, "broken_previous_hash_link"

    recomputed = _signature_hash(
        sequence_index=row.sequence_index,
        source_type=row.source_type,
        character_delta_count=row.character_delta_count,
        timestamp_epoch=row.timestamp_epoch,
        previous_block_hash=row.previous_block_hash,
    )
    if recomputed != row.signature_hash:
        return False, "signature_mismatch"

    return True, None


async def verify_mutation_chain(
    db: AsyncSession,
    *,
    user_id: str,
    document_id: str,
    limit: int = 2_000,
) -> MutationChainVerificationResult:
    """Verify sequence continuity and hash-link integrity for one document ledger."""
    rows = await list_mutation_receipts(
        db,
        user_id=user_id,
        document_id=document_id,
        limit=limit,
    )
    if not rows:
        return MutationChainVerificationResult(
            user_id=user_id,
            document_id=document_id,
            checked_count=0,
            is_valid=True,
            first_invalid_sequence_index=None,
            reason=None,
        )

    expected_previous_hash = "0" * 64

    for expected_sequence_index, row in enumerate(rows):
        is_valid, reason = _verify_row_chain(
            row,
            expected_previous_hash=expected_previous_hash,
            expected_sequence_index=expected_sequence_index,
        )
        if not is_valid:
            return MutationChainVerificationResult(
                user_id=user_id,
                document_id=document_id,
                checked_count=expected_sequence_index,
                is_valid=False,
                first_invalid_sequence_index=row.sequence_index,
                reason=reason,
            )

        expected_previous_hash = row.signature_hash

    return MutationChainVerificationResult(
        user_id=user_id,
        document_id=document_id,
        checked_count=len(rows),
        is_valid=True,
        first_invalid_sequence_index=None,
        reason=None,
    )
