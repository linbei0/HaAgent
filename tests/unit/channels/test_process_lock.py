"""
tests/unit/channels/test_process_lock.py - gateway 单实例锁测试

验证锁竞争、异常退出释放与重复释放行为。
"""

from pathlib import Path

from haagent.channels.process_lock import GatewayInstanceLock


def test_second_gateway_lock_is_rejected_until_first_releases(tmp_path: Path) -> None:
    path = tmp_path / "gateway.lock"
    first = GatewayInstanceLock(path)
    second = GatewayInstanceLock(path)

    assert first.acquire() is True
    assert second.acquire() is False

    first.release()
    assert second.acquire() is True
    second.release()


def test_gateway_lock_context_releases_after_exception(tmp_path: Path) -> None:
    path = tmp_path / "gateway.lock"
    try:
        with GatewayInstanceLock(path):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    retry = GatewayInstanceLock(path)
    assert retry.acquire() is True
    retry.release()


def test_gateway_lock_release_is_idempotent(tmp_path: Path) -> None:
    lock = GatewayInstanceLock(tmp_path / "gateway.lock")
    assert lock.acquire() is True

    lock.release()
    lock.release()
    assert lock.acquire() is True
    lock.release()
