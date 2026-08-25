from PIL import Image
import json
import csv
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Any
from opengs_maptool.models.project import Project
import opengs_maptool.logic.datastructure as ds


def export_image(path: str, image: Image.Image) -> None:
    """Export an image to the specified file path."""
    if not image or not path:
        return

    try:
        # Remove the alpha channel for JPEG image export
        ext = path.lower().rsplit('.', 1)[-1]
        if ext in ("jpg", "jpeg"):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        image.save(path)
    except Exception as error:
        print(f"Error saving image: {error}")


def export_territory_definitions(project: Project, path: str, fmt: str) -> None:
    """Export territory definitions to the specified path in the given format."""
    territory_data = project.territory_data
    if not territory_data:
        print("No territory data to export.")
        return

    data: dict[str, dict[str, Any]] = {}
    for d in territory_data:
        data[d.territory_id] = d.serialize_territory_json()
    
    if fmt in ("json", "yaml", "xml"):
        if fmt == "json":
            _write_json(path, data)
        elif fmt == "yaml":
            _write_yaml(path, data)
        elif fmt == "xml":
            root = ET.Element("territories")

            for territory_id, info in data.items():
                territory_element = ET.SubElement(root, "territory")
                territory_element.set("id", str(territory_id))

                for key, value in info.items():
                    element = ET.SubElement(territory_element, key)
                    element.text = str(value)

            rough_xml = ET.tostring(root, encoding="unicode")
            pretty_xml = minidom.parseString(rough_xml).toprettyxml(indent="    ")

            with open(path, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=';')
            keys = ds.RegionMetadata.TERRITORY_JSON_KEYS
            keys = ("id", *keys)

            w.writerow(keys)
            for territory_id, info in data.items():
                w.writerow(_get_csv_row_values(keys, info, id_value=territory_id, id_key="id"))


def export_territory_history(project: Project, path: str, fmt: str) -> None:
    """Export territory history to the specified path in the given format."""
    territory_data = project.territory_data
    if not territory_data:
        print("No territory data to export.")
        return

    HISTORY_FORMAT_KEYS = ("provinces",)
    data: dict[str, dict[str, list[str]]] = {}
    for d in territory_data:
        data[d.territory_id] = {
            "provinces": d.province_ids or [],
        }

    if fmt in ("json", "yaml", "xml"):
        if fmt == "json":
            _write_json(path, data)
        elif fmt == "yaml":
            _write_yaml(path, data)
        elif fmt == "xml":
            root = ET.Element("territories")

            for territory_id, info in data.items():
                territory_element = ET.SubElement(root, "territory")
                territory_element.set("id", str(territory_id))

                provinces_element = ET.SubElement(territory_element, "provinces")

                for province_id in info.get("provinces", []):
                    province_element = ET.SubElement(provinces_element, "province")
                    province_element.text = str(province_id)

            rough_xml = ET.tostring(root, encoding="unicode")
            pretty_xml = minidom.parseString(rough_xml).toprettyxml(indent="    ")

            with open(path, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=';')
            keys = ("id", *HISTORY_FORMAT_KEYS)
            w.writerow(keys)

            for province_id, info in data.items():
                w.writerow(_get_csv_row_values(keys, info, id_value=province_id, id_key="id"))


def export_province_definitions(project: Project, path: str, fmt: str) -> None:
    """Export province definitions to the specified path in the given format."""
    province_data = project.province_data
    if not province_data:
        print("No province data to export.")
        return

    has_terrain = any((d.province_terrain is not None) for d in province_data)

    data: dict[str, dict[str, Any]] = {}
    for d in province_data:
        data[d.province_id] = d.serialize_province_json(include_terrain=has_terrain)

    if fmt in ("json", "yaml", "xml"):
        if fmt == "json":
            _write_json(path, data)
        elif fmt == "yaml":
            _write_yaml(path, data)
        elif fmt == "xml":
            root = ET.Element("provinces")

            for province_id, info in data.items():
                province_element = ET.SubElement(root, "province")
                province_element.set("id", str(province_id))

                for key, value in info.items():
                    element = ET.SubElement(province_element, key)
                    element.text = str(value)

            rough_xml = ET.tostring(root, encoding="unicode")
            pretty_xml = minidom.parseString(rough_xml).toprettyxml(indent="    ")

            with open(path, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=';')
            keys = ds.RegionMetadata.PROVINCE_JSON_KEYS_WITH_TERRAIN if has_terrain else ds.RegionMetadata.PROVINCE_JSON_KEYS_WITHOUT_TERRAIN
            keys = ("id", *keys)
            w.writerow(keys)

            for province_id, info in data.items():
                w.writerow(_get_csv_row_values(keys, info, id_value=province_id, id_key="id"))

def _get_csv_row_values(keys: tuple[str, ...], data: dict[str, Any], id_value: Any, id_key: str = "id") -> list[Any]:
    values = []
    for key in keys:
        if key == id_key:
            values.append(id_value)
        else:
            values.append(data.get(key))
    return values

def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _write_yaml(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
