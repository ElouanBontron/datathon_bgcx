"""Vérifie les filtres de date/heure et l'import CSV de la page simplifiée."""
from pathlib import Path
import csv
import io
import re

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 1100})
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        overview = page.get_by_role("tabpanel", name="Vue d’ensemble", exact=True)
        overview.locator(".js-plotly-plot").first.wait_for(timeout=45000)
        slider = overview.get_by_role("slider").first
        slider.focus()
        slider.press("ArrowRight")
        overview.get_by_text("11:00 — 12:00", exact=True).wait_for(timeout=20000)
        assert not page.locator('[data-testid="stException"]').count()
        print("HOUR_FILTER_OK", flush=True)

        date_box = page.locator('[data-testid="stDateInput"]').filter(has=page.locator('[data-testid="stWidgetLabel"]').get_by_text("Journée à explorer", exact=True))
        date_box.get_by_role("button", name="Choose date").click()
        page.get_by_role("dialog").get_by_role("gridcell").filter(has_text=re.compile("^8$")).click()
        expect(date_box.locator('[data-type="day"]')).to_have_text("08", timeout=20000)
        date_box.get_by_role("button", name="Choose date").click()
        page.get_by_role("dialog").get_by_role("gridcell").filter(has_text=re.compile("^7$")).click()
        expect(date_box.locator('[data-type="day"]')).to_have_text("07", timeout=20000)
        print("DAY_FILTER_OK", flush=True)

        # Valeurs synthétiques constantes pour vérifier le branchement de la carte.
        content = io.StringIO()
        writer = csv.writer(content)
        writer.writerow(["arc", "datetime", "debit_horaire", "taux_occupation"])
        for axis in ["Champs-Elysées", "Convention", "Saint-Pères"]:
            for hour in range(24):
                writer.writerow([axis, f"2026-11-07 {hour:02d}:00", 250, 25])
        page.get_by_role("button", name=re.compile("Réglages")).click()
        page.get_by_text("CSV de l’équipe", exact=True).click()
        page.locator('input[type="file"]').set_input_files({
            "name": "previsions_test.csv", "mimeType": "text/csv", "buffer": content.getvalue().encode("utf-8")
        })
        page.get_by_text("Prévisions de l’équipe · Gains estimés", exact=True).wait_for(timeout=20000)
        page.keyboard.press("Escape")
        page.wait_for_function("document.querySelector('.js-plotly-plot')?.layout?.annotations?.some(a => a.text.includes('25.0 %'))")
        assert not page.locator('[data-testid="stException"]').count()
        assert not page.locator('[data-testid="stAlert"]').filter(has_text="Impossible").count()
        print("CSV_IMPORT_UI_OK", flush=True)

        page.get_by_role("button", name=re.compile("Réglages")).click()
        page.get_by_text("Profil historique", exact=True).click()
        page.get_by_text("Profil historique saisonnier · Gains estimés", exact=True).wait_for(timeout=20000)
        page.keyboard.press("Escape")
        page.wait_for_function("document.querySelector('.js-plotly-plot')?.layout?.annotations?.some(a => a.text.includes('%') && !a.text.includes('25.0 %'))")
        page.locator('[data-testid="stMain"]').evaluate('(el) => el.scrollTop = 0')
        page.screenshot(path=str(OUT / "flowpilot_overview.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
