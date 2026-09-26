#!/usr/bin/env python3
import sys

with open("docs/MATHEMATICS.md", "r") as f:
    text = f.read()

# Let's verify line count and see where we can expand
lines = text.splitlines()
print(f"Current lines: {len(lines)}")

