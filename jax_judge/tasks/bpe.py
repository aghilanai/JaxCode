"""Byte-Pair Encoding (BPE) task."""

TASK = {
    "title": "Byte-Pair Encoding (BPE)",
    "difficulty": "Hard",
    "function_name": "SimpleBPE",
    "hint": "train: split words into chars + </w>. Iteratively find most frequent adjacent pair, merge it. encode: apply learned merges in order to split text into subwords.",
    "tests": [
        {
            "name": "Correct number of merges",
            "code": """
bpe = {fn}()
bpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=5)
assert len(bpe.merges) == 5, f'Expected 5 merges, got {len(bpe.merges)}'
""",
        },
        {
            "name": "Most frequent pair merged first",
            "code": """
bpe = {fn}()
bpe.train(['aaa', 'aaa', 'aaa', 'bbb'], num_merges=1)
assert bpe.merges[0] == ('a', 'a'), f'First merge: {bpe.merges[0]}'
""",
        },
        {
            "name": "Encode returns list of strings",
            "code": """
bpe = {fn}()
bpe.train(['low', 'lower', 'lowest'] * 3, num_merges=10)
tokens = bpe.encode('low')
assert isinstance(tokens, list), 'encode must return a list'
assert all(isinstance(t, str) for t in tokens), 'tokens must be strings'
reconstructed = ''.join(t.replace('</w>', '') for t in tokens)
assert reconstructed == 'low', f'Reconstruction: {reconstructed}'
""",
        },
        {
            "name": "More merges -> fewer tokens",
            "code": """
bpe1 = {fn}()
bpe1.train(['hello'] * 10, num_merges=2)
bpe2 = {fn}()
bpe2.train(['hello'] * 10, num_merges=10)
assert len(bpe2.encode('hello')) <= len(bpe1.encode('hello')), 'More merges should reduce tokens'
""",
        },
    ],
}
