"""Tests for the RFC 6962 Merkle tree (contract section 7).

The v1 implementation was forgeable in four independent ways. Every one of them has a
test here that fails against v1's behaviour and passes against this one:

1. no domain separation  -> a 64-byte data item collided with an interior node;
2. duplicated unpaired nodes -> ``root([A,B,C]) == root([A,B,C,C])`` (CVE-2012-2459);
3. ``verify_proof(proof)`` compared against ``proof.root_hash`` -> every forgery verified;
4. ``generate_consistency_proof`` returned ``[]`` unconditionally -> "proven" by default.

Expectations are hand-derived from RFC 6962 section 2.1, never pasted from this module's
own output.
"""

import dataclasses
import hashlib
import math
import subprocess
import sys

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from signledger.core.exceptions import ConfigurationError, ValidationError
from signledger.crypto.merkle import (
    ROOT_PREFIX,
    ConsistencyProof,
    InclusionProof,
    MerkleTree,
    ProofStep,
    expected_path_length,
    leaf_hash,
    node_hash,
    verify_consistency,
    verify_inclusion,
)

A, B, C, D, E = b"a", b"b", b"c", b"d", b"e"

# --------------------------------------------------------------------------------------
# Break 1: domain separation
# --------------------------------------------------------------------------------------


def test_leaf_hash_prefixes_the_data_item_with_the_zero_domain_tag():
    assert leaf_hash(b"payload") == hashlib.sha256(b"\x00payload").hexdigest()


def test_node_hash_prefixes_the_raw_child_digests_with_the_one_domain_tag():
    left, right = leaf_hash(b"a"), leaf_hash(b"b")
    expected = hashlib.sha256(
        b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)
    ).hexdigest()
    assert node_hash(left, right) == expected


def test_a_data_item_shaped_like_an_interior_preimage_does_not_hash_to_that_node():
    """The classic second-preimage attack. v1 had no domain tags and fell to it.

    ``forged`` is the exact byte string an interior node is built from. Without the
    0x00/0x01 tags ``H(forged)`` *is* the interior digest, so an attacker can present two
    real leaves as a single leaf and shift every index in the tree.
    """
    left, right = leaf_hash(b"a"), leaf_hash(b"b")
    forged = bytes.fromhex(left) + bytes.fromhex(right)
    assert leaf_hash(forged) != node_hash(left, right)


# --------------------------------------------------------------------------------------
# RFC 6962 section 2.1 -- the Merkle Tree Hash, hand-derived
# --------------------------------------------------------------------------------------


def test_tree_hash_of_one_leaf_is_that_leaf_hash():
    assert MerkleTree([A]).merkle_tree_hash == leaf_hash(A)


def test_tree_hash_of_two_leaves_is_the_interior_node_over_both():
    expected = node_hash(leaf_hash(A), leaf_hash(B))
    assert MerkleTree([A, B]).merkle_tree_hash == expected


def test_tree_hash_of_four_leaves_is_a_balanced_pair_of_pairs():
    expected = node_hash(
        node_hash(leaf_hash(A), leaf_hash(B)),
        node_hash(leaf_hash(C), leaf_hash(D)),
    )
    assert MerkleTree([A, B, C, D]).merkle_tree_hash == expected


def test_tree_hash_splits_at_the_largest_power_of_two_below_the_leaf_count():
    """RFC 6962: ``k`` is the largest power of 2 *strictly less than* n, not n//2.

    For n=5 that puts four leaves in the left subtree and one in the right.
    """
    left = node_hash(
        node_hash(leaf_hash(A), leaf_hash(B)),
        node_hash(leaf_hash(C), leaf_hash(D)),
    )
    assert MerkleTree([A, B, C, D, E]).merkle_tree_hash == node_hash(left, leaf_hash(E))


def test_leaf_count_reports_the_number_of_data_items():
    assert MerkleTree([A, B, C]).leaf_count == 3


# --------------------------------------------------------------------------------------
# Break 2: unpaired nodes are promoted, never duplicated (CVE-2012-2459)
# --------------------------------------------------------------------------------------


def test_unpaired_leaf_is_promoted_unchanged():
    """n=3: the odd leaf is carried up as-is, not paired with a copy of itself."""
    expected = node_hash(node_hash(leaf_hash(A), leaf_hash(B)), leaf_hash(C))
    assert MerkleTree([A, B, C]).merkle_tree_hash == expected


def test_three_leaves_hash_differently_from_three_with_the_tail_duplicated():
    """CVE-2012-2459.

    v1 duplicated the unpaired node, which made these two *different* logs share one
    root: an attacker could append a copy of the last entry and the root never moved.
    Asserted on the bare MTH on purpose - comparing the count-bound root would pass even
    if the duplication bug were still there, because the leaf counts differ.
    """
    assert (
        MerkleTree([A, B, C]).merkle_tree_hash
        != MerkleTree([A, B, C, C]).merkle_tree_hash
    )


def test_duplicated_tail_tree_hashes_the_duplicate_as_a_real_pair():
    """The n=4 shape is reached honestly: [A,B,C,C] pairs C with C at the leaf level."""
    expected = node_hash(
        node_hash(leaf_hash(A), leaf_hash(B)),
        node_hash(leaf_hash(C), leaf_hash(C)),
    )
    assert MerkleTree([A, B, C, C]).merkle_tree_hash == expected


# --------------------------------------------------------------------------------------
# Break 6: the leaf count is bound into the root
# --------------------------------------------------------------------------------------


def test_root_binds_the_leaf_count_under_its_own_domain_tag():
    tree = MerkleTree([A, B, C])
    expected = hashlib.sha256(
        b"\x02" + (3).to_bytes(8, "big") + bytes.fromhex(tree.merkle_tree_hash)
    ).hexdigest()
    assert tree.root == expected


def test_root_is_not_the_bare_tree_hash():
    tree = MerkleTree([A, B, C])
    assert tree.root != tree.merkle_tree_hash


def test_single_leaf_root_is_distinct_from_the_leaf_hash_it_wraps():
    """MTH([X]) is *literally* leaf_hash(X). Without the count binding a leaf digest
    could be presented as a whole tree's root."""
    assert MerkleTree([A]).root != leaf_hash(A)


def test_root_prefix_is_the_two_byte_tag():
    assert ROOT_PREFIX == b"\x02"


# --------------------------------------------------------------------------------------
# Algorithm agility
# --------------------------------------------------------------------------------------


def test_tree_defaults_to_sha256():
    assert MerkleTree([A]).algorithm == "sha256"


def test_tree_rejects_an_unsupported_hash_algorithm():
    with pytest.raises(ConfigurationError):
        MerkleTree([A], algorithm="md5")


def test_unsupported_algorithm_message_names_the_algorithm_not_the_payload():
    with pytest.raises(ConfigurationError) as excinfo:
        MerkleTree([b"top-secret-payload"], algorithm="md5")
    assert "top-secret-payload" not in str(excinfo.value)


def test_sha512_tree_hash_is_hand_derived_from_sha512_primitives():
    expected = hashlib.sha512(
        b"\x01"
        + hashlib.sha512(b"\x00a").digest()
        + hashlib.sha512(b"\x00b").digest()
    ).hexdigest()
    assert MerkleTree([A, B], algorithm="sha512").merkle_tree_hash == expected


def test_sha512_root_is_128_hex_characters():
    assert len(MerkleTree([A, B, C], algorithm="sha512").root) == 128


def test_leaf_hash_honours_the_algorithm_argument():
    assert leaf_hash(A, algorithm="sha3_256") == hashlib.sha3_256(b"\x00a").hexdigest()


def test_node_hash_honours_the_algorithm_argument():
    left = right = leaf_hash(A, algorithm="sha3_256")
    expected = hashlib.sha3_256(
        b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)
    ).hexdigest()
    assert node_hash(left, right, algorithm="sha3_256") == expected


# --------------------------------------------------------------------------------------
# Inclusion proof generation (RFC 6962 section 2.1.1)
# --------------------------------------------------------------------------------------


def test_inclusion_proof_of_a_left_leaf_names_its_right_sibling():
    proof = MerkleTree([A, B]).generate_inclusion_proof(0)
    assert proof.path == (ProofStep(leaf_hash(B), sibling_is_left=False),)


def test_inclusion_proof_of_a_right_leaf_marks_the_sibling_as_left():
    proof = MerkleTree([A, B]).generate_inclusion_proof(1)
    assert proof.path == (ProofStep(leaf_hash(A), sibling_is_left=True),)


def test_inclusion_proof_records_the_leaf_index_and_the_tree_size():
    proof = MerkleTree([A, B, C]).generate_inclusion_proof(2)
    assert (proof.leaf_index, proof.tree_size) == (2, 3)


def test_inclusion_proof_path_runs_from_the_leaf_upwards():
    proof = MerkleTree([A, B, C, D]).generate_inclusion_proof(0)
    assert proof.path == (
        ProofStep(leaf_hash(B), sibling_is_left=False),
        ProofStep(node_hash(leaf_hash(C), leaf_hash(D)), sibling_is_left=False),
    )


def test_inclusion_proof_for_a_promoted_leaf_climbs_only_the_levels_it_has_siblings_on():
    """n=5, leaf 4: the leaf is promoted through two levels untouched, so its audit path
    holds a single step - the whole left subtree - not ceil(log2(5)) = 3 steps."""
    proof = MerkleTree([A, B, C, D, E]).generate_inclusion_proof(4)
    left_subtree = node_hash(
        node_hash(leaf_hash(A), leaf_hash(B)),
        node_hash(leaf_hash(C), leaf_hash(D)),
    )
    assert proof.path == (ProofStep(left_subtree, sibling_is_left=True),)


def test_inclusion_proof_has_exactly_the_documented_fields():
    proof = MerkleTree([A, B]).generate_inclusion_proof(0)
    names = {f.name for f in dataclasses.fields(proof)}
    assert names == {"leaf_index", "tree_size", "path", "algorithm"}


def test_inclusion_proof_carries_no_root_for_a_verifier_to_read():
    """Break 3. v1's proof carried ``root_hash`` and ``verify_proof`` compared against
    it, so a forged proof authenticated itself."""
    proof = MerkleTree([A, B, C]).generate_inclusion_proof(1)
    assert [n for n in dir(proof) if "root" in n.lower()] == []


def test_inclusion_proof_carries_no_leaf_hash_for_a_verifier_to_trust():
    """Break 4. The leaf digest must be re-derived from the data item at verify time."""
    proof = MerkleTree([A, B, C]).generate_inclusion_proof(1)
    assert [n for n in dir(proof) if "leaf" in n.lower()] == ["leaf_index"]


def test_inclusion_proof_is_immutable():
    proof = MerkleTree([A, B]).generate_inclusion_proof(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        proof.leaf_index = 1


def test_proof_step_is_immutable():
    step = MerkleTree([A, B]).generate_inclusion_proof(0).path[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.sibling_hash = "00" * 32


def test_inclusion_proof_records_the_algorithm_of_the_tree_that_made_it():
    proof = MerkleTree([A, B], algorithm="sha512").generate_inclusion_proof(0)
    assert proof.algorithm == "sha512"


def test_inclusion_proof_type_is_the_exported_one():
    assert isinstance(MerkleTree([A]).generate_inclusion_proof(0), InclusionProof)


# --------------------------------------------------------------------------------------
# verify_inclusion: round trip
# --------------------------------------------------------------------------------------

ALL_SIZES = list(range(1, 18))


def items(n):
    """n distinct data items. Distinct so a wrong-index proof is actually detectable."""
    return [b"item-%d" % i for i in range(n)]


@pytest.mark.parametrize("size", ALL_SIZES)
def test_every_leaf_of_every_tree_size_verifies_against_the_real_root(size):
    leaves = items(size)
    tree = MerkleTree(leaves)
    for index in range(size):
        proof = tree.generate_inclusion_proof(index)
        assert verify_inclusion(proof, leaves[index], tree.root) is True


@pytest.mark.parametrize("size", ALL_SIZES)
def test_no_leaf_verifies_against_the_root_of_a_tree_one_leaf_longer(size):
    leaves = items(size)
    tree = MerkleTree(leaves)
    other_root = MerkleTree([*leaves, b"appended"]).root
    for index in range(size):
        proof = tree.generate_inclusion_proof(index)
        assert verify_inclusion(proof, leaves[index], other_root) is False


@pytest.mark.parametrize("size", ALL_SIZES)
def test_a_proof_never_verifies_for_a_data_item_it_was_not_made_for(size):
    """Break 4: the leaf digest is re-derived from ``data_item`` every time."""
    leaves = items(size)
    tree = MerkleTree(leaves)
    for index in range(size):
        proof = tree.generate_inclusion_proof(index)
        assert verify_inclusion(proof, b"not the item", tree.root) is False


def test_sha512_proof_round_trips():
    leaves = items(6)
    tree = MerkleTree(leaves, algorithm="sha512")
    proof = tree.generate_inclusion_proof(3)
    assert verify_inclusion(proof, leaves[3], tree.root) is True


# --------------------------------------------------------------------------------------
# Break 3: the root is a caller argument, never read off the proof
# --------------------------------------------------------------------------------------


def test_a_fully_attacker_controlled_proof_does_not_verify_against_an_honest_root():
    """The whole of v1's Merkle security failure in one test.

    The attacker picks the data item, builds their own tree around it and generates a
    structurally perfect proof. Nothing about the proof is malformed - it is simply a
    proof about a different tree.
    """
    honest = MerkleTree(items(8))
    forged_item = b"attacker was here"
    attacker_tree = MerkleTree([b"x", forged_item, b"y", b"z"])
    forged_proof = attacker_tree.generate_inclusion_proof(1)
    assert verify_inclusion(forged_proof, forged_item, honest.root) is False


def test_the_same_forged_proof_verifies_only_against_the_attackers_own_root():
    """Confirms the forged proof above is internally consistent, so the previous test
    fails for the right reason: v1 read ``proof.root_hash`` and returned True here."""
    forged_item = b"attacker was here"
    attacker_tree = MerkleTree([b"x", forged_item, b"y", b"z"])
    forged_proof = attacker_tree.generate_inclusion_proof(1)
    assert verify_inclusion(forged_proof, forged_item, attacker_tree.root) is True


def test_a_second_preimage_proof_does_not_verify_against_the_honest_root():
    """Break 1 end to end: the attacker claims an interior preimage is a leaf."""
    leaves = items(4)
    honest = MerkleTree(leaves)
    interior_preimage = bytes.fromhex(leaf_hash(leaves[0])) + bytes.fromhex(
        leaf_hash(leaves[1])
    )
    forged = InclusionProof(
        leaf_index=0,
        tree_size=2,
        path=(
            ProofStep(
                node_hash(leaf_hash(leaves[2]), leaf_hash(leaves[3])),
                sibling_is_left=False,
            ),
        ),
        algorithm="sha256",
    )
    assert verify_inclusion(forged, interior_preimage, honest.root) is False


# --------------------------------------------------------------------------------------
# Break 5: the path length and its directions are pinned by (tree_size, leaf_index)
# --------------------------------------------------------------------------------------

# Hand-derived from RFC 6962 section 2.1.1 by expanding PATH(m, D[n]) for each n.
HAND_DERIVED_PATH_LENGTHS = {
    1: [0],
    2: [1, 1],
    3: [2, 2, 1],
    4: [2, 2, 2, 2],
    5: [3, 3, 3, 3, 1],
    6: [3, 3, 3, 3, 2, 2],
    7: [3, 3, 3, 3, 3, 3, 2],
    8: [3, 3, 3, 3, 3, 3, 3, 3],
}


@pytest.mark.parametrize("size", sorted(HAND_DERIVED_PATH_LENGTHS))
def test_expected_path_length_matches_the_hand_derived_table(size):
    lengths = [expected_path_length(size, i) for i in range(size)]
    assert lengths == HAND_DERIVED_PATH_LENGTHS[size]


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
def test_full_tree_path_length_is_ceil_log2_of_the_leaf_count(size):
    """The contract's ``len(proof.path) == ceil(log2(n))``, which holds exactly when the
    tree is full."""
    for index in range(size):
        assert expected_path_length(size, index) == math.ceil(math.log2(size))


@pytest.mark.parametrize("size", ALL_SIZES)
def test_the_longest_path_in_any_tree_is_ceil_log2_of_the_leaf_count(size):
    """The generalisation for ragged trees: promoted leaves sit *above* the deepest
    level, so their paths are shorter, and no path is ever longer."""
    longest = max(expected_path_length(size, i) for i in range(size))
    assert longest == math.ceil(math.log2(size))


@pytest.mark.parametrize("size", ALL_SIZES)
def test_generated_path_length_always_equals_the_expected_length(size):
    tree = MerkleTree(items(size))
    for index in range(size):
        proof = tree.generate_inclusion_proof(index)
        assert len(proof.path) == expected_path_length(size, index)


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_path_directions_are_the_bits_of_the_leaf_index_in_a_full_tree(size):
    tree = MerkleTree(items(size))
    for index in range(size):
        proof = tree.generate_inclusion_proof(index)
        directions = [step.sibling_is_left for step in proof.path]
        assert directions == [bool(index >> level & 1) for level in range(len(proof.path))]


def test_expected_path_length_rejects_an_empty_tree():
    with pytest.raises(ValidationError):
        expected_path_length(0, 0)


def test_expected_path_length_rejects_a_negative_index():
    with pytest.raises(ValidationError):
        expected_path_length(4, -1)


def test_expected_path_length_rejects_an_index_at_the_tree_size():
    with pytest.raises(ValidationError):
        expected_path_length(4, 4)


# --------------------------------------------------------------------------------------
# Tampered proofs
# --------------------------------------------------------------------------------------


def honest(size=8, index=3):
    leaves = items(size)
    tree = MerkleTree(leaves)
    return tree, tree.generate_inclusion_proof(index), leaves[index]


def test_a_proof_whose_leaf_index_was_edited_does_not_verify():
    """Without a structural check ``leaf_index`` is unauthenticated decoration: the
    digests still chain up to the real root, so the proof would claim the item sits at
    an index it does not occupy."""
    tree, proof, item = honest(size=4, index=0)
    moved = dataclasses.replace(proof, leaf_index=3)
    assert verify_inclusion(moved, item, tree.root) is False


def test_a_truncated_path_does_not_verify():
    tree, proof, item = honest()
    truncated = dataclasses.replace(proof, path=proof.path[:-1])
    assert verify_inclusion(truncated, item, tree.root) is False


def test_an_extended_path_does_not_verify():
    tree, proof, item = honest()
    extended = dataclasses.replace(
        proof, path=(*proof.path, ProofStep("11" * 32, sibling_is_left=False))
    )
    assert verify_inclusion(extended, item, tree.root) is False


def test_a_path_with_a_flipped_direction_does_not_verify():
    tree, proof, item = honest()
    flipped = dataclasses.replace(
        proof,
        path=(
            dataclasses.replace(proof.path[0], sibling_is_left=not proof.path[0].sibling_is_left),
            *proof.path[1:],
        ),
    )
    assert verify_inclusion(flipped, item, tree.root) is False


def test_a_tampered_sibling_digest_does_not_verify():
    tree, proof, item = honest()
    zeroed = ProofStep("00" * 32, sibling_is_left=proof.path[0].sibling_is_left)
    tampered = dataclasses.replace(proof, path=(zeroed, *proof.path[1:]))
    assert verify_inclusion(tampered, item, tree.root) is False


def test_a_proof_whose_tree_size_was_edited_does_not_verify():
    tree, proof, item = honest()
    resized = dataclasses.replace(proof, tree_size=9)
    assert verify_inclusion(resized, item, tree.root) is False


# --------------------------------------------------------------------------------------
# A verifier may never crash on attacker-supplied input
# --------------------------------------------------------------------------------------


def test_a_non_hex_sibling_returns_false_rather_than_raising():
    tree, proof, item = honest()
    junk = dataclasses.replace(
        proof, path=(ProofStep("zz" * 32, sibling_is_left=False), *proof.path[1:])
    )
    assert verify_inclusion(junk, item, tree.root) is False


def test_an_odd_length_hex_sibling_returns_false_rather_than_raising():
    tree, proof, item = honest()
    junk = dataclasses.replace(
        proof, path=(ProofStep("abc", sibling_is_left=False), *proof.path[1:])
    )
    assert verify_inclusion(junk, item, tree.root) is False


def test_a_sibling_of_the_wrong_digest_width_returns_false():
    tree, proof, item = honest()
    short = dataclasses.replace(
        proof, path=(ProofStep("ab" * 16, sibling_is_left=False), *proof.path[1:])
    )
    assert verify_inclusion(short, item, tree.root) is False


def test_a_proof_naming_an_unsupported_algorithm_returns_false():
    tree, proof, item = honest()
    downgraded = dataclasses.replace(proof, algorithm="md5")
    assert verify_inclusion(downgraded, item, tree.root) is False


def test_a_proof_naming_a_different_supported_algorithm_returns_false():
    tree, proof, item = honest()
    swapped = dataclasses.replace(proof, algorithm="sha512")
    assert verify_inclusion(swapped, item, tree.root) is False


def test_a_proof_with_a_negative_leaf_index_returns_false():
    tree, proof, item = honest()
    assert verify_inclusion(dataclasses.replace(proof, leaf_index=-1), item, tree.root) is False


def test_a_proof_with_a_leaf_index_at_the_tree_size_returns_false():
    tree, proof, item = honest()
    assert verify_inclusion(dataclasses.replace(proof, leaf_index=8), item, tree.root) is False


def test_a_proof_claiming_an_empty_tree_returns_false():
    tree, proof, item = honest()
    empty = dataclasses.replace(proof, tree_size=0, leaf_index=0, path=())
    assert verify_inclusion(empty, item, tree.root) is False


def test_a_proof_claiming_a_negative_tree_size_returns_false():
    tree, proof, item = honest()
    assert verify_inclusion(dataclasses.replace(proof, tree_size=-4), item, tree.root) is False


# --------------------------------------------------------------------------------------
# The empty tree
# --------------------------------------------------------------------------------------


def test_empty_tree_hash_is_the_digest_of_the_empty_string():
    """RFC 6962 section 2.1: MTH({}) = HASH()."""
    assert MerkleTree([]).merkle_tree_hash == hashlib.sha256(b"").hexdigest()


def test_empty_tree_root_binds_a_zero_leaf_count():
    expected = hashlib.sha256(
        b"\x02" + (0).to_bytes(8, "big") + hashlib.sha256(b"").digest()
    ).hexdigest()
    assert MerkleTree([]).root == expected


def test_empty_tree_reports_zero_leaves():
    assert MerkleTree([]).leaf_count == 0


def test_empty_tree_cannot_produce_an_inclusion_proof():
    with pytest.raises(ValidationError):
        MerkleTree([]).generate_inclusion_proof(0)


def test_no_item_verifies_against_the_empty_tree_root():
    empty_root = MerkleTree([]).root
    tree = MerkleTree(items(4))
    proof = tree.generate_inclusion_proof(0)
    assert verify_inclusion(proof, b"item-0", empty_root) is False


# --------------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------------


def test_tree_rejects_a_str_leaf():
    """Leaves are raw bytes. Accepting str would make the digest depend on an implicit
    encoding choice that a second implementation cannot guess."""
    with pytest.raises(ValidationError):
        MerkleTree(["not bytes"])


def test_tree_rejects_a_bytearray_leaf():
    with pytest.raises(ValidationError):
        MerkleTree([bytearray(b"ab")])


def test_leaf_rejection_message_names_the_type_not_the_payload():
    with pytest.raises(ValidationError) as excinfo:
        MerkleTree(["patient-record-4471"])
    message = str(excinfo.value)
    assert "patient-record-4471" not in message
    assert "str" in message


def test_leaf_hash_rejects_a_str_data_item():
    with pytest.raises(ValidationError):
        leaf_hash("not bytes")


def test_generate_inclusion_proof_rejects_a_negative_index():
    with pytest.raises(ValidationError):
        MerkleTree(items(4)).generate_inclusion_proof(-1)


def test_generate_inclusion_proof_rejects_an_index_at_the_leaf_count():
    with pytest.raises(ValidationError):
        MerkleTree(items(4)).generate_inclusion_proof(4)


def test_verify_inclusion_rejects_a_proof_of_the_wrong_type():
    tree, _proof, item = honest()
    with pytest.raises(ValidationError):
        verify_inclusion("not a proof", item, tree.root)


def test_verify_inclusion_rejects_a_str_data_item():
    tree, proof, _item = honest()
    with pytest.raises(ValidationError):
        verify_inclusion(proof, "item-3", tree.root)


def test_verify_inclusion_rejects_a_non_str_expected_root():
    tree, proof, item = honest()
    with pytest.raises(ValidationError):
        verify_inclusion(proof, item, bytes.fromhex(tree.root))


def test_verify_inclusion_rejects_an_empty_expected_root():
    """An empty root usually means a caller read a missing field. Answering "not
    included" would report that configuration bug as tampering."""
    _tree, proof, item = honest()
    with pytest.raises(ValidationError):
        verify_inclusion(proof, item, "")


def test_verify_inclusion_argument_errors_carry_no_payload():
    _tree, proof, _item = honest()
    with pytest.raises(ValidationError) as excinfo:
        verify_inclusion(proof, "sensitive-account-number", "00" * 32)
    assert "sensitive-account-number" not in str(excinfo.value)


def test_a_path_step_of_the_wrong_type_returns_false():
    tree, proof, item = honest()
    junk = dataclasses.replace(proof, path=("not a step", *proof.path[1:]))
    assert verify_inclusion(junk, item, tree.root) is False


def test_a_non_str_sibling_hash_returns_false():
    tree, proof, item = honest()
    junk = dataclasses.replace(
        proof, path=(ProofStep(b"\x00" * 32, sibling_is_left=False), *proof.path[1:])
    )
    assert verify_inclusion(junk, item, tree.root) is False


# --------------------------------------------------------------------------------------
# Break 7: consistency proofs (RFC 6962 section 2.1.2) with a real verifier
# --------------------------------------------------------------------------------------


def test_consistency_proof_has_exactly_the_documented_fields():
    proof = MerkleTree(items(4)).generate_consistency_proof(2)
    names = {f.name for f in dataclasses.fields(proof)}
    assert names == {"old_size", "new_size", "path", "algorithm"}


def test_consistency_proof_carries_no_roots_for_a_verifier_to_read():
    proof = MerkleTree(items(4)).generate_consistency_proof(2)
    assert [n for n in dir(proof) if "root" in n.lower()] == []


def test_consistency_proof_records_both_sizes():
    proof = MerkleTree(items(7)).generate_consistency_proof(3)
    assert (proof.old_size, proof.new_size) == (3, 7)


def test_consistency_proof_between_equal_sizes_is_empty():
    """RFC 6962: SUBPROOF(m, D[m], true) = {}. There is nothing to prove."""
    assert MerkleTree(items(4)).generate_consistency_proof(4).path == ()


@pytest.mark.parametrize("new_size", ALL_SIZES)
def test_consistency_proof_is_never_empty_when_the_tree_grew(new_size):
    """v1 returned ``[]`` unconditionally, which a naive caller reads as "proven"."""
    tree = MerkleTree(items(new_size))
    for old_size in range(1, new_size):
        assert tree.generate_consistency_proof(old_size).path != ()


def test_consistency_proof_from_three_to_four_is_hand_derived():
    """SUBPROOF(3, D[4], true) expanded by hand from RFC 6962 section 2.1.2."""
    leaves = items(4)
    proof = MerkleTree(leaves).generate_consistency_proof(3)
    assert proof.path == (
        leaf_hash(leaves[2]),
        leaf_hash(leaves[3]),
        node_hash(leaf_hash(leaves[0]), leaf_hash(leaves[1])),
    )


def test_consistency_proof_materialises_the_old_tree_hash_when_the_old_size_is_a_power_of_two():
    """RFC 9162 section 2.1.4.2 step 2 prepends ``hash1`` when the old size is a power of
    two, because SUBPROOF omitted it. SignLedger performs that step at generation time so
    the proof stays verifiable from the two *count-bound* roots alone - a bound root
    cannot be unwrapped back into the tree hash the RFC algorithm would need."""
    leaves = items(2)
    proof = MerkleTree(leaves).generate_consistency_proof(1)
    assert proof.path == (leaf_hash(leaves[0]), leaf_hash(leaves[1]))


@pytest.mark.parametrize("new_size", ALL_SIZES)
def test_every_prefix_of_every_tree_size_proves_consistent(new_size):
    leaves = items(new_size)
    tree = MerkleTree(leaves)
    for old_size in range(1, new_size + 1):
        old_root = MerkleTree(leaves[:old_size]).root
        proof = tree.generate_consistency_proof(old_size)
        assert verify_consistency(proof, old_root, tree.root) is True


def test_consistency_round_trips_under_sha512():
    leaves = items(11)
    tree = MerkleTree(leaves, algorithm="sha512")
    old_root = MerkleTree(leaves[:6], algorithm="sha512").root
    proof = tree.generate_consistency_proof(6)
    assert verify_consistency(proof, old_root, tree.root) is True


def test_an_empty_consistency_path_does_not_prove_a_tree_that_grew():
    """The exact v1 failure: an empty list must never read as proof."""
    leaves = items(6)
    tree = MerkleTree(leaves)
    old_root = MerkleTree(leaves[:3]).root
    empty = ConsistencyProof(old_size=3, new_size=6, path=(), algorithm="sha256")
    assert verify_consistency(empty, old_root, tree.root) is False


def test_consistency_fails_when_the_old_tree_is_not_a_prefix_of_the_new_one():
    """The tree grew, but history was rewritten at index 2 on the way."""
    old_root = MerkleTree([b"a", b"b", b"c"]).root
    rewritten = MerkleTree([b"a", b"b", b"X", b"d"])
    proof = rewritten.generate_consistency_proof(3)
    assert verify_consistency(proof, old_root, rewritten.root) is False


def test_consistency_fails_against_a_wrong_old_root():
    leaves = items(8)
    tree = MerkleTree(leaves)
    proof = tree.generate_consistency_proof(5)
    assert verify_consistency(proof, MerkleTree(leaves[:4]).root, tree.root) is False


def test_consistency_fails_against_a_wrong_new_root():
    leaves = items(8)
    tree = MerkleTree(leaves)
    proof = tree.generate_consistency_proof(5)
    old_root = MerkleTree(leaves[:5]).root
    assert verify_consistency(proof, old_root, MerkleTree(items(9)).root) is False


def test_equal_sizes_require_the_two_roots_to_match():
    tree = MerkleTree(items(4))
    proof = tree.generate_consistency_proof(4)
    assert verify_consistency(proof, tree.root, tree.root) is True


def test_equal_sizes_with_differing_roots_do_not_verify():
    tree = MerkleTree(items(4))
    proof = tree.generate_consistency_proof(4)
    assert verify_consistency(proof, MerkleTree(items(4, )).root, "aa" * 32) is False


def test_equal_sizes_with_a_non_empty_path_do_not_verify():
    tree = MerkleTree(items(4))
    forged = ConsistencyProof(
        old_size=4, new_size=4, path=("00" * 32,), algorithm="sha256"
    )
    assert verify_consistency(forged, tree.root, tree.root) is False


def test_a_tampered_consistency_path_element_does_not_verify():
    leaves = items(8)
    tree = MerkleTree(leaves)
    old_root = MerkleTree(leaves[:5]).root
    proof = tree.generate_consistency_proof(5)
    tampered = dataclasses.replace(proof, path=("00" * 32, *proof.path[1:]))
    assert verify_consistency(tampered, old_root, tree.root) is False


def test_a_truncated_consistency_path_does_not_verify():
    leaves = items(8)
    tree = MerkleTree(leaves)
    old_root = MerkleTree(leaves[:5]).root
    proof = tree.generate_consistency_proof(5)
    assert verify_consistency(
        dataclasses.replace(proof, path=proof.path[:-1]), old_root, tree.root
    ) is False


def test_an_extended_consistency_path_does_not_verify():
    leaves = items(8)
    tree = MerkleTree(leaves)
    old_root = MerkleTree(leaves[:5]).root
    proof = tree.generate_consistency_proof(5)
    assert verify_consistency(
        dataclasses.replace(proof, path=(*proof.path, "11" * 32)), old_root, tree.root
    ) is False


def test_consistency_generation_rejects_an_old_size_above_the_tree_size():
    with pytest.raises(ValidationError):
        MerkleTree(items(4)).generate_consistency_proof(5)


def test_consistency_generation_rejects_a_zero_old_size():
    """Consistency with an empty tree carries no information; refusing is the only
    answer that cannot be mistaken for proof."""
    with pytest.raises(ValidationError):
        MerkleTree(items(4)).generate_consistency_proof(0)


def test_consistency_generation_rejects_a_negative_old_size():
    with pytest.raises(ValidationError):
        MerkleTree(items(4)).generate_consistency_proof(-1)


def test_verify_consistency_rejects_a_zero_old_size():
    tree = MerkleTree(items(4))
    forged = ConsistencyProof(old_size=0, new_size=4, path=(), algorithm="sha256")
    assert verify_consistency(forged, MerkleTree([]).root, tree.root) is False


def test_verify_consistency_rejects_an_old_size_above_the_new_size():
    tree = MerkleTree(items(4))
    forged = ConsistencyProof(old_size=5, new_size=4, path=("aa" * 32,), algorithm="sha256")
    assert verify_consistency(forged, tree.root, tree.root) is False


def test_verify_consistency_rejects_a_non_hex_path_element():
    leaves = items(8)
    tree = MerkleTree(leaves)
    old_root = MerkleTree(leaves[:5]).root
    proof = tree.generate_consistency_proof(5)
    junk = dataclasses.replace(proof, path=("zz" * 32, *proof.path[1:]))
    assert verify_consistency(junk, old_root, tree.root) is False


def test_verify_consistency_rejects_an_unsupported_algorithm():
    leaves = items(8)
    tree = MerkleTree(leaves)
    old_root = MerkleTree(leaves[:5]).root
    proof = tree.generate_consistency_proof(5)
    assert verify_consistency(
        dataclasses.replace(proof, algorithm="md5"), old_root, tree.root
    ) is False


def test_verify_consistency_rejects_a_proof_of_the_wrong_type():
    tree = MerkleTree(items(4))
    with pytest.raises(ValidationError):
        verify_consistency("not a proof", tree.root, tree.root)


def test_verify_consistency_rejects_a_non_str_old_root():
    proof = MerkleTree(items(4)).generate_consistency_proof(2)
    with pytest.raises(ValidationError):
        verify_consistency(proof, b"\x00" * 32, MerkleTree(items(4)).root)


def test_verify_consistency_rejects_an_empty_new_root():
    proof = MerkleTree(items(4)).generate_consistency_proof(2)
    with pytest.raises(ValidationError):
        verify_consistency(proof, MerkleTree(items(2)).root, "")


def test_consistency_proof_is_immutable():
    proof = MerkleTree(items(4)).generate_consistency_proof(2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        proof.old_size = 3


# --------------------------------------------------------------------------------------
# Properties (hypothesis)
# --------------------------------------------------------------------------------------

LEAF_LISTS = st.lists(st.binary(max_size=48), min_size=1, max_size=24)
PROPERTY = settings(deadline=None, max_examples=200)


@PROPERTY
@given(LEAF_LISTS, st.data())
def test_property_every_leaf_verifies_against_its_own_trees_root(leaves, data):
    index = data.draw(st.integers(min_value=0, max_value=len(leaves) - 1))
    tree = MerkleTree(leaves)
    proof = tree.generate_inclusion_proof(index)
    assert verify_inclusion(proof, leaves[index], tree.root) is True


@PROPERTY
@given(LEAF_LISTS, st.binary(min_size=32, max_size=32), st.data())
def test_property_no_proof_verifies_against_any_other_root(leaves, other, data):
    index = data.draw(st.integers(min_value=0, max_value=len(leaves) - 1))
    tree = MerkleTree(leaves)
    other_root = other.hex()
    assume(other_root != tree.root)
    proof = tree.generate_inclusion_proof(index)
    assert verify_inclusion(proof, leaves[index], other_root) is False


@PROPERTY
@given(LEAF_LISTS, st.binary(max_size=48), st.data())
def test_property_no_proof_verifies_for_another_data_item(leaves, other_item, data):
    index = data.draw(st.integers(min_value=0, max_value=len(leaves) - 1))
    assume(other_item != leaves[index])
    tree = MerkleTree(leaves)
    proof = tree.generate_inclusion_proof(index)
    assert verify_inclusion(proof, other_item, tree.root) is False


@PROPERTY
@given(LEAF_LISTS, st.data())
def test_property_a_proof_never_verifies_at_another_index(leaves, data):
    assume(len(leaves) > 1)
    index = data.draw(st.integers(min_value=0, max_value=len(leaves) - 1))
    moved = data.draw(
        st.integers(min_value=0, max_value=len(leaves) - 1).filter(lambda i: i != index)
    )
    tree = MerkleTree(leaves)
    proof = dataclasses.replace(
        tree.generate_inclusion_proof(index), leaf_index=moved
    )
    assert verify_inclusion(proof, leaves[index], tree.root) is False


@PROPERTY
@given(LEAF_LISTS, st.binary(max_size=48))
def test_property_appending_a_leaf_always_changes_the_root(leaves, extra):
    assert MerkleTree(leaves).root != MerkleTree([*leaves, extra]).root


@PROPERTY
@given(LEAF_LISTS, st.data())
def test_property_a_truncated_log_never_shares_a_root_with_the_full_log(leaves, data):
    kept = data.draw(st.integers(min_value=0, max_value=len(leaves) - 1))
    assert MerkleTree(leaves[:kept]).root != MerkleTree(leaves).root


@PROPERTY
@given(LEAF_LISTS)
def test_property_leaf_order_changes_the_root(leaves):
    reversed_leaves = list(reversed(leaves))
    assume(reversed_leaves != list(leaves))
    assert MerkleTree(leaves).root != MerkleTree(reversed_leaves).root


@PROPERTY
@given(LEAF_LISTS, st.data())
def test_property_every_prefix_proves_consistent_with_the_whole(leaves, data):
    old_size = data.draw(st.integers(min_value=1, max_value=len(leaves)))
    tree = MerkleTree(leaves)
    old_root = MerkleTree(leaves[:old_size]).root
    proof = tree.generate_consistency_proof(old_size)
    assert verify_consistency(proof, old_root, tree.root) is True


@PROPERTY
@given(LEAF_LISTS, st.binary(min_size=32, max_size=32), st.data())
def test_property_consistency_never_holds_against_another_old_root(leaves, other, data):
    old_size = data.draw(st.integers(min_value=1, max_value=len(leaves)))
    tree = MerkleTree(leaves)
    other_root = other.hex()
    assume(other_root != MerkleTree(leaves[:old_size]).root)
    proof = tree.generate_consistency_proof(old_size)
    assert verify_consistency(proof, other_root, tree.root) is False


@PROPERTY
@given(st.integers(min_value=1, max_value=200), st.data())
def test_property_no_audit_path_is_longer_than_ceil_log2(size, data):
    index = data.draw(st.integers(min_value=0, max_value=size - 1))
    assert expected_path_length(size, index) <= math.ceil(math.log2(size))


@PROPERTY
@given(LEAF_LISTS, st.data())
def test_property_generated_paths_are_always_the_expected_length(leaves, data):
    index = data.draw(st.integers(min_value=0, max_value=len(leaves) - 1))
    proof = MerkleTree(leaves).generate_inclusion_proof(index)
    assert len(proof.path) == expected_path_length(len(leaves), index)


def test_an_empty_consistency_path_is_rejected_whatever_roots_it_is_given():
    """The empty path must be refused *structurally*, not merely fail a root comparison.

    These two roots are crafted to be exactly what a replay of an empty path would
    produce, so a verifier that skipped the empty-path guard would answer True. That is
    the shape of the v1 bug: no proof at all, read as proof.
    """
    old_root = hashlib.sha256(b"\x02" + (3).to_bytes(8, "big")).hexdigest()
    new_root = hashlib.sha256(b"\x02" + (6).to_bytes(8, "big")).hexdigest()
    empty = ConsistencyProof(old_size=3, new_size=6, path=(), algorithm="sha256")
    assert verify_consistency(empty, old_root, new_root) is False


HOSTILE_HEX = st.one_of(
    st.text(max_size=70),
    st.binary(max_size=40).map(bytes.hex),
    st.just("00" * 32),
)
HOSTILE_ALGORITHM = st.sampled_from(["sha256", "sha512", "sha3_256", "md5", "", "SHA256"])


@PROPERTY
@given(
    st.integers(min_value=-5, max_value=40),
    st.integers(min_value=-5, max_value=40),
    st.lists(st.tuples(HOSTILE_HEX, st.booleans()), max_size=8),
    HOSTILE_ALGORITHM,
    st.binary(max_size=24),
)
def test_property_verify_inclusion_never_raises_on_a_hostile_proof(
    leaf_index, tree_size, steps, algorithm, data_item
):
    """A verifier that can be crashed by the input it is verifying is a denial of
    service. Every malformed proof must produce an answer, and never True by accident."""
    proof = InclusionProof(
        leaf_index=leaf_index,
        tree_size=tree_size,
        path=tuple(ProofStep(h, d) for h, d in steps),
        algorithm=algorithm,
    )
    assert verify_inclusion(proof, data_item, MerkleTree(items(6)).root) is False


@PROPERTY
@given(
    st.integers(min_value=-5, max_value=40),
    st.integers(min_value=-5, max_value=40),
    st.lists(HOSTILE_HEX, max_size=8),
    HOSTILE_ALGORITHM,
)
def test_property_verify_consistency_never_raises_on_a_hostile_proof(
    old_size, new_size, path, algorithm
):
    leaves = items(9)
    proof = ConsistencyProof(
        old_size=old_size, new_size=new_size, path=tuple(path), algorithm=algorithm
    )
    verdict = verify_consistency(
        proof, MerkleTree(leaves[:4]).root, MerkleTree(leaves).root
    )
    assert verdict is False


def test_a_tree_can_be_built_from_a_generator():
    generated = MerkleTree(bytes([i]) for i in range(5))
    assert generated.root == MerkleTree([bytes([i]) for i in range(5)]).root


# --------------------------------------------------------------------------------------
# Independent oracle
# --------------------------------------------------------------------------------------


def reference_tree_hash(leaves):
    """An independently written Merkle tree hash, for cross-checking.

    Deliberately a *different algorithm* from the one under test: this walks the tree
    level by level carrying an unpaired node up unchanged, where the implementation
    recurses on RFC 6962's "largest power of two below n" split. Agreeing on every size
    is real evidence about the tree shape; a copy of the implementation would be none.
    """
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(b"\x00" + item).digest() for item in leaves]
    while len(level) > 1:
        parents = [
            hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest()
            for i in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            parents.append(level[-1])
        level = parents
    return level[0].hex()


@pytest.mark.parametrize("size", range(0, 34))
def test_tree_hash_agrees_with_an_independently_written_implementation(size):
    leaves = items(size)
    assert MerkleTree(leaves).merkle_tree_hash == reference_tree_hash(leaves)


# --------------------------------------------------------------------------------------
# Mutation-sweep hardening
#
# Everything below kills a specific mutant that survived the sweep required by
# docs/ARCHITECTURE.md section 11. Each test names the production edit it catches.
# --------------------------------------------------------------------------------------


def test_a_leaf_index_one_past_the_tree_is_refused_even_when_the_path_shape_fits():
    """Kills ``0 <= leaf_index < tree_size`` -> ``0 <= leaf_index <= tree_size``.

    ``test_a_proof_with_a_leaf_index_at_the_tree_size_returns_false`` uses an 8-leaf tree,
    where ``_audit_directions(8, 8)`` demands a different direction tuple from any real
    proof and the forgery dies on the shape check instead of the range check. At size 2
    the two agree: ``_audit_directions(2, 2) == _audit_directions(2, 1) == (True,)``, so a
    relaxed bound lets an out-of-range index chain all the way up to the honest root and
    verify. The index is part of what an inclusion proof *asserts*; a verifier that
    accepts ``leaf 2 of a 2-leaf tree`` has authenticated a statement that is false.
    """
    leaves = items(2)
    tree = MerkleTree(leaves)
    proof = tree.generate_inclusion_proof(1)

    # Guard the premise: this forgery reaches the digest replay, it is not shape-rejected.
    assert expected_path_length(2, 1) == len(proof.path)

    out_of_range = dataclasses.replace(proof, leaf_index=2)
    assert verify_inclusion(out_of_range, leaves[1], tree.root) is False


def test_consistency_between_two_empty_trees_is_refused_rather_than_proven():
    """Kills ``old_size < 1 or new_size < old_size`` -> ``... and ...``.

    Both existing size-guard tests survive the ``and`` because each trips only one of the
    two clauses and then fails later for an unrelated reason. A proof claiming 0 -> 0 with
    one root repeated trips only the first clause *and* lands in the equal-sizes branch,
    which answers True on an empty path. "Nothing is consistent with nothing" is exactly
    the v1 shape: a statement about no data, read as proof.
    """
    empty_root = MerkleTree([]).root
    forged = ConsistencyProof(old_size=0, new_size=0, path=(), algorithm="sha256")
    assert verify_consistency(forged, empty_root, empty_root) is False


def test_consistency_between_two_negative_sizes_is_refused():
    """The same guard, below zero: no arithmetic on a negative size means anything."""
    empty_root = MerkleTree([]).root
    forged = ConsistencyProof(old_size=-3, new_size=-3, path=(), algorithm="sha256")
    assert verify_consistency(forged, empty_root, empty_root) is False


# --------------------------------------------------------------------------------------
# Diagnostics: the operator-facing half of every guard above
#
# A `pytest.raises(ValidationError)` with no message assertion proves the guard fired and
# nothing about what it reported. Every message below survived the sweep as
# `raise ValidationError(None)` - a guard that refuses correctly and then says nothing.
# --------------------------------------------------------------------------------------


def test_empty_tree_inclusion_error_reports_the_leaf_count():
    with pytest.raises(ValidationError) as excinfo:
        MerkleTree([]).generate_inclusion_proof(0)
    assert str(excinfo.value) == "a tree of 0 leaves has no leaf to prove"


def test_out_of_range_leaf_index_error_reports_the_index_and_the_size():
    with pytest.raises(ValidationError) as excinfo:
        MerkleTree(items(4)).generate_inclusion_proof(7)
    assert str(excinfo.value) == "leaf_index 7 is outside a tree of 4 leaves"


def test_unsupported_algorithm_error_lists_the_supported_algorithms():
    with pytest.raises(ConfigurationError) as excinfo:
        MerkleTree([A], algorithm="md5")
    assert str(excinfo.value) == (
        "unsupported hash algorithm 'md5'; "
        "choose one of ['sha256', 'sha3_256', 'sha3_512', 'sha512']"
    )


def test_zero_old_size_error_reports_the_size_it_was_given():
    with pytest.raises(ValidationError) as excinfo:
        MerkleTree(items(4)).generate_consistency_proof(0)
    assert str(excinfo.value) == "old_size must be at least 1 to prove anything, got 0"


def test_oversized_old_size_error_reports_both_sizes():
    with pytest.raises(ValidationError) as excinfo:
        MerkleTree(items(4)).generate_consistency_proof(5)
    assert str(excinfo.value) == "old_size 5 exceeds the tree's 4 leaves"


#: Stands in for "the honest proof" in the argument tables below, so a row can vary the
#: proof or hold it fixed without every row having to build one.
REAL_PROOF = object()


@pytest.mark.parametrize(
    ("proof_arg", "data_item", "expected_root", "expected"),
    [
        ("not a proof", b"item-3", "00" * 32, "proof must be an InclusionProof, got str"),
        (None, b"item-3", "00" * 32, "proof must be an InclusionProof, got NoneType"),
        (REAL_PROOF, "item-3", "00" * 32, "data_item must be bytes, got str"),
        (
            REAL_PROOF,
            bytearray(b"item-3"),
            "00" * 32,
            "data_item must be bytes, got bytearray",
        ),
        (REAL_PROOF, b"item-3", b"\x00" * 32, "expected_root must be a hex str, got bytes"),
        (REAL_PROOF, b"item-3", None, "expected_root must be a hex str, got NoneType"),
        (REAL_PROOF, b"item-3", "", "expected_root must not be empty"),
    ],
)
def test_verify_inclusion_argument_errors_say_what_was_wrong(
    proof_arg, data_item, expected_root, expected
):
    _tree, proof, _item = honest()
    with pytest.raises(ValidationError) as excinfo:
        verify_inclusion(
            proof if proof_arg is REAL_PROOF else proof_arg, data_item, expected_root
        )
    assert str(excinfo.value) == expected


@pytest.mark.parametrize(
    ("proof_arg", "old_root", "new_root", "expected"),
    [
        ("not a proof", "00" * 32, "00" * 32, "proof must be a ConsistencyProof, got str"),
        (None, "00" * 32, "00" * 32, "proof must be a ConsistencyProof, got NoneType"),
        (REAL_PROOF, b"\x00" * 32, "00" * 32, "old_root must be a hex str, got bytes"),
        (REAL_PROOF, "", "00" * 32, "old_root must not be empty"),
        (REAL_PROOF, "00" * 32, b"\x00" * 32, "new_root must be a hex str, got bytes"),
        (REAL_PROOF, "00" * 32, "", "new_root must not be empty"),
    ],
)
def test_verify_consistency_argument_errors_name_which_root_was_wrong(
    proof_arg, old_root, new_root, expected
):
    """The two roots are separate arguments; an error that does not say which one is bad
    sends the caller looking in the wrong place."""
    proof = MerkleTree(items(4)).generate_consistency_proof(2)
    with pytest.raises(ValidationError) as excinfo:
        verify_consistency(proof if proof_arg is REAL_PROOF else proof_arg, old_root, new_root)
    assert str(excinfo.value) == expected


class TestTheReplayLoopCannotBeMadeToHang:
    """``_replay_consistency`` must terminate on any input, not only on reachable input.

    ``fn = old_size - 1``, and the loop ``while fn & 1: fn >>= 1`` does not terminate for a
    negative ``fn``: ``-1 & 1`` is 1 and ``-1 >> 1`` is ``-1``, forever. Today that is
    unreachable because ``verify_consistency`` refuses ``old_size < 1`` first, so exactly
    one guard stands between an attacker-supplied ``ConsistencyProof`` and a wedged
    verifier thread. A mutation sweep found two single-edit changes to that guard which
    hang the process instead of returning False.

    Depth is cheap here and the loop is the last thing that should trust its caller, so it
    refuses non-positive sizes itself. Run in a subprocess: if this regresses, the test
    must fail in seconds rather than wedging the suite until the global timeout.
    """

    def run_replay(self, old_size: int, new_size: int) -> subprocess.CompletedProcess:
        program = (
            "from signledger.crypto.merkle import _replay_consistency\n"
            "path = [b'x' * 32, b'y' * 32]\n"
            f"print(_replay_consistency('sha256', path, {old_size}, {new_size}))\n"
        )
        return subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=15
        )

    def test_an_old_size_of_zero_returns_rather_than_spinning(self) -> None:
        """``old_size=0`` is the one value that wedges the loop, and the only one.

        ``fn`` becomes ``-1``, which is the fixed point of arithmetic right shift: ``-1``
        is odd, so the loop continues, and ``-1 >> 1`` is ``-1`` again. Other negatives
        terminate on their own — ``-(2**40)`` shifts down to ``-2`` in 40 steps and exits
        because ``-2`` is even. Those are covered below as boundaries, but this is the
        case that discriminates: it is the only one that hangs without the guard.
        """
        result = self.run_replay(0, 8)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "None"

    @pytest.mark.parametrize("old_size", [-1, -(2**40)])
    def test_other_negative_sizes_are_refused_rather_than_answered(self, old_size: int) -> None:
        """These already terminated, but by accident of the arithmetic, not by decision.

        Before the guard they fell through the replay and returned ``None`` only because a
        later ``sn != 0`` check happened to catch them. Refusing at the top makes that
        deliberate instead of incidental.
        """
        result = self.run_replay(old_size, 8)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "None"

    def test_verify_consistency_still_answers_false_for_a_zero_sized_proof(self) -> None:
        """The public guard stays; this only removes its sole responsibility for liveness."""
        leaves = [b"a", b"b", b"c", b"d"]
        tree = MerkleTree(leaves)

        proof = ConsistencyProof(
            old_size=0, new_size=0, path=(bytes.fromhex(tree.root),), algorithm="sha256"
        )

        assert verify_consistency(proof, tree.root, tree.root) is False
