"""Unit tests for authorship receipt append and verification helpers."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deeptutor.models.user import User
from deeptutor.services.writing.authorship_receipt_service import (
    append_mutation_receipt,
    verify_mutation_chain,
)


@pytest.fixture
def service_user(db_session: AsyncSession) -> User:
    """Create an in-memory user fixture for service-level provenance tests."""
    user = User(
        id="test-user-authorship-service",
        username="authorship_service",
        email="authorship_service@test.local",
        password_hash="hashed",
        role="student",
        is_active=True,
    )
    db_session.add(user)
    return user


@pytest.mark.asyncio
async def test_append_mutation_receipt_creates_genesis_record(
    db_session: AsyncSession,
    service_user: User,
) -> None:
    """First append creates sequence 0 with zeroed previous hash."""
    receipt = await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-1",
        source_type="manual_typing",
        character_delta_count=11,
    )
    await db_session.commit()

    assert receipt.sequence_index == 0
    assert receipt.previous_block_hash == "0" * 64
    assert receipt.signature_hash != ""


@pytest.mark.asyncio
async def test_append_mutation_receipt_advances_hash_chain(
    db_session: AsyncSession,
    service_user: User,
) -> None:
    """Subsequent append references prior signature and increments sequence."""
    first = await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-2",
        source_type="manual_typing",
        character_delta_count=5,
    )
    second = await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-2",
        source_type="api_scrub",
        character_delta_count=2,
    )
    await db_session.commit()

    assert first.sequence_index == 0
    assert second.sequence_index == 1
    assert second.previous_block_hash == first.signature_hash


@pytest.mark.asyncio
async def test_verify_mutation_chain_accepts_empty_chain(
    db_session: AsyncSession,
    service_user: User,
) -> None:
    """Verification should be valid for documents with no receipts."""
    result = await verify_mutation_chain(
        db_session,
        user_id=service_user.id,
        document_id="missing-ledger",
    )

    assert result.is_valid is True
    assert result.checked_count == 0
    assert result.reason is None


@pytest.mark.asyncio
async def test_verify_mutation_chain_accepts_valid_chain(
    db_session: AsyncSession,
    service_user: User,
) -> None:
    """Verification should pass for untampered contiguous ledgers."""
    await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-3",
        source_type="manual_typing",
        character_delta_count=9,
    )
    await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-3",
        source_type="context_expansion",
        character_delta_count=3,
    )
    await db_session.commit()

    result = await verify_mutation_chain(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-3",
    )

    assert result.is_valid is True
    assert result.checked_count == 2
    assert result.first_invalid_sequence_index is None
    assert result.reason is None


@pytest.mark.asyncio
async def test_verify_mutation_chain_detects_signature_tamper(
    db_session: AsyncSession,
    service_user: User,
) -> None:
    """Verification flags chain when signature hash no longer matches payload."""
    from deeptutor.models.text_mutation_receipt import TextMutationReceipt

    await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-4",
        source_type="manual_typing",
        character_delta_count=6,
    )
    await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-4",
        source_type="api_scrub",
        character_delta_count=1,
    )
    await db_session.commit()

    rows = (
        (
            await db_session.execute(
                select(TextMutationReceipt)
                .where(TextMutationReceipt.user_id == service_user.id)
                .where(TextMutationReceipt.document_id == "service-doc-4")
                .order_by(TextMutationReceipt.sequence_index.asc())
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2

    rows[1].signature_hash = "tampered-signature-hash"
    await db_session.commit()

    result = await verify_mutation_chain(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-4",
    )

    assert result.is_valid is False
    assert result.first_invalid_sequence_index == 1
    assert result.reason == "signature_mismatch"


@pytest.mark.asyncio
async def test_verify_mutation_chain_detects_sequence_gap(
    db_session: AsyncSession,
    service_user: User,
) -> None:
    """Verification flags non-contiguous sequence indexes as invalid."""
    from deeptutor.models.text_mutation_receipt import TextMutationReceipt

    await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-5",
        source_type="manual_typing",
        character_delta_count=4,
    )
    await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-5",
        source_type="api_scrub",
        character_delta_count=2,
    )
    await db_session.commit()

    rows = (
        (
            await db_session.execute(
                select(TextMutationReceipt)
                .where(TextMutationReceipt.user_id == service_user.id)
                .where(TextMutationReceipt.document_id == "service-doc-5")
                .order_by(TextMutationReceipt.sequence_index.asc())
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2

    rows[1].sequence_index = 4
    await db_session.commit()

    result = await verify_mutation_chain(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-5",
    )

    assert result.is_valid is False
    assert result.first_invalid_sequence_index == 4
    assert result.reason == "non_contiguous_sequence"


@pytest.mark.asyncio
async def test_verify_mutation_chain_detects_previous_hash_break(
    db_session: AsyncSession,
    service_user: User,
) -> None:
    """Verification flags records whose previous hash link is broken."""
    from deeptutor.models.text_mutation_receipt import TextMutationReceipt

    await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-6",
        source_type="manual_typing",
        character_delta_count=10,
    )
    await append_mutation_receipt(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-6",
        source_type="context_expansion",
        character_delta_count=3,
    )
    await db_session.commit()

    rows = (
        (
            await db_session.execute(
                select(TextMutationReceipt)
                .where(TextMutationReceipt.user_id == service_user.id)
                .where(TextMutationReceipt.document_id == "service-doc-6")
                .order_by(TextMutationReceipt.sequence_index.asc())
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2

    rows[1].previous_block_hash = "0" * 64
    await db_session.commit()

    result = await verify_mutation_chain(
        db_session,
        user_id=service_user.id,
        document_id="service-doc-6",
    )

    assert result.is_valid is False
    assert result.first_invalid_sequence_index == 1
    assert result.reason == "broken_previous_hash_link"
