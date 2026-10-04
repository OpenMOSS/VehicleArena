"""Browser regressions for the cockpit shown to driving agents."""

from pathlib import Path

import pytest

from visualization.web3d_camera import Web3DCameraRenderer


@pytest.mark.parametrize("capture", [True, False])
def test_cockpit_has_no_speed_readout(capture):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Playwright Chromium is not installed")

    renderer = Web3DCameraRenderer()
    try:
        renderer._start()
        page = renderer._browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        port = renderer._server.server_address[1]
        page.goto(
            f"http://127.0.0.1:{port}/?view=cockpit&capture={int(capture)}",
            wait_until="networkidle",
        )
        page.wait_for_function(
            "typeof window.vehicleArenaCaptureFrame === 'function'")
        # Exercise ordinary animation too: deleting the DOM readout must not
        # leave a stale JavaScript reference that breaks the live viewer.
        page.evaluate("() => new Promise(resolve => requestAnimationFrame("
                      "() => requestAnimationFrame(resolve)))")
        assert page.locator("#cockpit").is_visible()
        assert page.locator("#cluster-speed, .cluster").count() == 0
        assert "km/h" not in page.locator("#cockpit").inner_text()
        if capture:
            # The separate human-facing debug panel must not leak speed into
            # agent screenshots either.
            assert not page.locator("#ego-speed").is_visible()
            assert "km/h" not in page.locator("body").inner_text()
        assert not errors
    finally:
        renderer.close()
