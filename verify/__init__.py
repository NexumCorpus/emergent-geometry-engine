"""The verifier — a SEPARATE code path from the kernel.

Hard rule: nothing in this package may import from ``geometry``. The verifier
certifies physics/math by independent re-derivation; if it shared code with the
kernel a single bug could hide in both. Cross-checks that compare kernel output to
the verifier live in ``tests/``, where importing both is the explicit purpose.
"""
