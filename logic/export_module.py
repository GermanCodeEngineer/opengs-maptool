import os
import json
from PyQt6.QtWidgets import QFileDialog
import csv


def export_image(parent_layout, image, text):
    if image:
        try:
            path, _ = QFileDialog.getSaveFileName(
                parent_layout, text, "", "PNG Files (*.png)")
            image.save(path)

        except Exception as error:
            print(f"Error saving image: {error}")


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
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(["province_id", "R", "G", "B",
                       "province_type", "x", "y"])
            for d in metadata:
                w.writerow([d["province_id"], d["R"], d["G"], d["B"],
                            d["province_type"], round(d["x"], 2), round(d["y"], 2)])
    except Exception as e:
        print("Error saving province data:", e)


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
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(["territory_id", "R", "G", "B",
                       "territory_type", "x", "y"])
            for d in metadata:
                w.writerow([d["territory_id"], d["R"], d["G"], d["B"],
                            d["territory_type"], round(d["x"], 2), round(d["y"], 2)])
    except Exception as e:
        print("Error saving territory data:", e)


def export_territories_json(main_layout):

    # Ask user for export directory
    export_dir = QFileDialog.getExistingDirectory(
        main_layout,
        "Select Territory Export Folder",
        "",
        QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
    )

    if not export_dir:
        print("Territory export cancelled.")
        return

    territories = main_layout.territory_data

    for terr in territories:
        tid = terr["territory_id"]
        provinces = terr.get("province_ids", [])

        data = {
            "territory_id": tid,
            "provinces": provinces
        }

        filename = os.path.join(export_dir, f"{tid}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    print(f"Exported {len(territories)} territories to: {export_dir}")
