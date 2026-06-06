from __future__ import annotations

from argparse import Namespace

from dinoml import cli


def test_compile_context_kwargs_injects_target_into_accepting_build_specs():
    def build_spec(*, snapshot, target=None, arch=None):
        return snapshot, target, arch

    args = Namespace(target="rocm", arch="gfx1201", no_tf32=False, use_fp16_acc=False)

    enriched = cli._with_compile_context_kwargs(build_spec, args, {"snapshot": "x"})

    assert enriched == {"snapshot": "x", "target": "rocm", "arch": "gfx1201"}


def test_compile_context_kwargs_preserves_explicit_user_values():
    def build_spec(*, snapshot, target=None):
        return snapshot, target

    args = Namespace(target="rocm", arch=None, no_tf32=False, use_fp16_acc=False)

    enriched = cli._with_compile_context_kwargs(build_spec, args, {"snapshot": "x", "target": "cuda"})

    assert enriched["target"] == "cuda"
