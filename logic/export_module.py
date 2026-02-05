import csv
import json
from PyQt6.QtWidgets import QFileDialog


def export_image(parent_layout, image, text):
    if image:
        try:
            path, _ = QFileDialog.getSaveFileName(
                parent_layout, text, "", "PNG Files (*.png)")
            image.save(path)

        except Exception as error:
            print(f"Error saving image: {error}")


def _export_provinces_csv_to(metadata, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["region_id", "R", "G", "B",
                    "region_type", "x", "y"])
        for d in metadata:
            w.writerow([d["region_id"], d["R"], d["G"], d["B"],
                        d["region_type"], round(d["x"], 2), round(d["y"], 2)])


def export_provinces_csv(main_layout):
    metadata = getattr(main_layout, "province_data", None)
    if not metadata:
        print("No province data to export.")
        return

    path, _ = QFileDialog.getSaveFileName(
        main_layout, "Export Province CSV", "", "CSV Files (*.csv)")
    if not path:
        return

    try:
        _export_provinces_csv_to(metadata, path)
    except Exception as e:
        print("Error saving province data:", e)


def _export_territories_csv_to(metadata, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["region_id", "R", "G", "B",
                    "region_type", "x", "y"])
        for d in metadata:
            if "x" not in d:
                breakpoint()
            w.writerow([d["region_id"], d["R"], d["G"], d["B"],
                        d["region_type"], round(d["x"], 2), round(d["y"], 2)])

def export_territories_csv(main_layout):
    metadata = getattr(main_layout, "territory_data", None)
    if not metadata:
        print("No territory data to export.")
        return

    path, _ = QFileDialog.getSaveFileName(
        main_layout, "Export territory CSV", "", "CSV Files (*.csv)")
    if not path:
        return

    try:
        _export_territories_csv_to(metadata, path)
    except Exception as e:
        print("Error saving territory data:", e)

def _export_territories_json_to(territories: list[dict], export_path: str) -> None:
    territories = [{
        "region_id": t["region_id"],
        "provinces": t.get("region_ids", [])
    } for t in territories]

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(territories, f, indent=4)

def export_territories_json(main_layout):

    # Ask user for export directory
    path, _ = QFileDialog.getSaveFileName(
        main_layout, "Export Territories JSON", "", "JSON Files (*.json)")

    if not path:
        print("Territory export cancelled.")
        return

    _export_territories_json_to(main_layout.territory_data, path)

    print(f"Exported territories to: {path}")
