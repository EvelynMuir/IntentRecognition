"""Minimal reader for the subset of Caffe Datum used by the LDL release."""

from __future__ import annotations

import io
import struct
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def _read_varint(data: bytes, offset: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift >= 70:
            raise ValueError("Malformed protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def parse_caffe_datum(payload: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {"float_data": []}
    offset = 0
    while offset < len(payload):
        tag, offset = _read_varint(payload, offset)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = _read_varint(payload, offset)
            names = {1: "channels", 2: "height", 3: "width", 5: "label"}
            if field in names:
                result[names[field]] = value
            elif field == 7:
                result["encoded"] = bool(value)
        elif wire == 2:
            size, offset = _read_varint(payload, offset)
            end = offset + size
            if end > len(payload):
                raise ValueError("Truncated protobuf length-delimited field")
            value = payload[offset:end]
            offset = end
            if field == 4:
                result["data"] = value
            elif field == 6:
                if size % 4:
                    raise ValueError("Packed Datum.float_data is not float32-aligned")
                result["float_data"].extend(struct.unpack(f"<{size // 4}f", value))
        elif wire == 5:
            if offset + 4 > len(payload):
                raise ValueError("Truncated protobuf fixed32 field")
            value = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            if field == 6:
                result["float_data"].append(value)
        elif wire == 1:
            offset += 8
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
    return result


def datum_to_rgb(datum: Dict[str, Any]) -> Image.Image:
    raw = datum.get("data")
    if raw is None:
        raise ValueError("Datum has no image data")
    if datum.get("encoded", False):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    channels = int(datum["channels"])
    height = int(datum["height"])
    width = int(datum["width"])
    array = np.frombuffer(raw, dtype=np.uint8)
    if array.size != channels * height * width:
        raise ValueError(f"Datum size mismatch: {array.size} != {channels}*{height}*{width}")
    array = array.reshape(channels, height, width).transpose(1, 2, 0)
    if channels == 3:
        array = array[:, :, ::-1]
    elif channels == 1:
        array = np.repeat(array, 3, axis=2)
    else:
        raise ValueError(f"Unsupported image channel count: {channels}")
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")
