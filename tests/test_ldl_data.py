import struct

import numpy as np

from src.data.components.caffe_datum import parse_caffe_datum


def test_parse_caffe_datum_unpacked_float_data() -> None:
    image = bytes(range(12))
    payload = b"\x08\x03\x10\x02\x18\x02\x22\x0c" + image
    payload += b"".join(b"\x35" + struct.pack("<f", value) for value in (0.25, 0.75))
    payload += b"\x38\x00"
    datum = parse_caffe_datum(payload)
    assert (datum["channels"], datum["height"], datum["width"]) == (3, 2, 2)
    assert datum["data"] == image
    assert np.allclose(datum["float_data"], [0.25, 0.75])
    assert datum["encoded"] is False
