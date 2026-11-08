"""Vérifie les trois onglets, les gains partagés et l'export agenda dans Chrome.

Lancer FlowPilot sur localhost:8501, puis : python tests/browser_smoke.py
"""
from pathlib import Path
import json

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def check_no_error(page):
    errors = page.locator('[data-testid="stException"]').all_text_contents()
    assert not errors, errors


def panel(page, name):
    return page.get_by_role("tabpanel", name=name, exact=True)


def screenshot(page, filename):
    page.locator('[data-testid="stMain"]').evaluate('(el) => el.scrollTop = 0')
    page.screenshot(path=str(ARTIFACTS / filename), full_page=True)


def metric(scope, label):
    return scope.locator(".metric").filter(has_text=label).locator(".metric-value")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 1100}, device_scale_factor=1)
        page_errors = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        page.get_by_role("tab", name="Vue d’ensemble", exact=True).wait_for(timeout=45000)
        overview = panel(page, "Vue d’ensemble")
        overview.get_by_text("Paris, heure par heure", exact=True).wait_for(timeout=45000)
        overview.locator(".js-plotly-plot").first.wait_for(timeout=45000)
        expect(page.get_by_role("tab")).to_have_count(3)
        expect(page.get_by_role("tab", name="Données", exact=True)).to_have_count(0)
        expect(page.locator('[data-testid="stSidebar"]')).to_have_count(0)
        for name in ["Économies estimées / jour", "Temps gagné / jour", "Coût en moins", "Départ recommandé"]:
            expect(overview.get_by_text(name, exact=True)).to_be_visible()
        check_no_error(page)
        initial_gain = metric(overview, "Économies estimées / jour").inner_text()
        screenshot(page, "flowpilot_overview.png")
        print(json.dumps({"overview": "OK", "tabs": 3, "gain": initial_gain}, ensure_ascii=False), flush=True)

        page.get_by_role("tab", name="Impact", exact=True).click()
        impact = panel(page, "Impact")
        impact.get_by_text("Avant / après", exact=True).wait_for(timeout=20000)
        volume = impact.get_by_role("spinbutton", name="Livraisons par jour", exact=True)
        volume.fill("80")
        volume.press("Enter")
        expect(volume).to_have_value("80")
        expect(metric(impact, "Économies estimées / jour")).not_to_have_text(initial_gain, timeout=20000)
        check_no_error(page)
        screenshot(page, "flowpilot_impact.png")
        page.get_by_role("tab", name="Vue d’ensemble", exact=True).click()
        expect(metric(overview, "Économies estimées / jour")).not_to_have_text(initial_gain, timeout=20000)
        updated_gain = metric(overview, "Économies estimées / jour").inner_text()
        print(json.dumps({"impact": "OK", "shared_gain": updated_gain}, ensure_ascii=False), flush=True)

        page.get_by_role("tab", name="Livraisons", exact=True).click()
        deliveries = panel(page, "Livraisons")
        deliveries.get_by_role("button", name="Construire mon planning").click()
        deliveries.get_by_text("Votre feuille de route", exact=True).wait_for(timeout=20000)
        check_no_error(page)
        with page.expect_download() as download:
            deliveries.get_by_role("button", name="Ajouter à mon agenda", exact=True).click()
        download.value.save_as(str(ARTIFACTS / "planning_verification.ics"))
        assert b"BEGIN:VEVENT" in (ARTIFACTS / "planning_verification.ics").read_bytes()
        screenshot(page, "flowpilot_planning.png")
        print(json.dumps({"planning": "OK", "calendar": "OK", "page_errors": page_errors}, ensure_ascii=False), flush=True)
        assert not page_errors, page_errors
        browser.close()


if __name__ == "__main__":
    main()
