from types import SimpleNamespace


def test_dgl_compat_aliases_old_message_api():
    from scripts.external_scalable_compat_runner import patch_dgl_message_api

    calls = []

    def copy_u(*args, **kwargs):
        calls.append(("copy_u", args, kwargs))
        return "u-result"

    def copy_e(*args, **kwargs):
        calls.append(("copy_e", args, kwargs))
        return "e-result"

    fn = SimpleNamespace(
        copy_u=copy_u,
        copy_e=copy_e,
    )

    patch_dgl_message_api(fn)

    assert fn.copy_src(src="f", out="msg") == "u-result"
    assert fn.copy_edge(edge="w", out="msg") == "e-result"
    assert calls == [
        ("copy_u", ("f", "msg"), {}),
        ("copy_e", ("w", "msg"), {}),
    ]


def test_graphsaint_np_bool_alias():
    from scripts.external_scalable_compat_runner import patch_numpy_bool

    np_like = SimpleNamespace(bool_=object())

    patch_numpy_bool(np_like)

    assert np_like.bool is np_like.bool_


def test_external_entry_argv_uses_remaining_args():
    from scripts.external_scalable_compat_runner import build_entry_argv

    assert build_entry_argv("src/sagn.py", ["--dataset", "ogbn-products"]) == [
        "src/sagn.py",
        "--dataset",
        "ogbn-products",
    ]


def test_entry_parent_is_inserted_once(monkeypatch):
    from scripts.external_scalable_compat_runner import ensure_entry_parent_on_path
    import sys

    monkeypatch.setattr(sys, "path", ["existing"])

    ensure_entry_parent_on_path("src/sagn.py")
    ensure_entry_parent_on_path("src/sagn.py")

    assert sys.path == ["src", "existing"]


def test_cwd_is_inserted_once(monkeypatch, tmp_path):
    from scripts.external_scalable_compat_runner import ensure_cwd_on_path
    import sys

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", ["existing"])

    ensure_cwd_on_path()
    ensure_cwd_on_path()

    assert sys.path == [str(tmp_path), "existing"]


def test_safe_cuda_device_uses_nullcontext_for_cpu():
    from scripts.external_scalable_compat_runner import make_safe_cuda_device

    calls = []

    def original(device=None):
        calls.append(device)
        return "cuda-context"

    safe_device = make_safe_cuda_device(original)

    with safe_device("cpu") as value:
        assert value is None
    assert safe_device("cuda:0") == "cuda-context"
    assert calls == ["cuda:0"]
