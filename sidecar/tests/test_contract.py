import json

import numpy as np
import pytest

from contract import ContractError, build_result, parse_request
from processing.grouping import decode_label_png
from tests.helpers import metadata, png_bytes, rgb_image


def test_parses_valid_request():
    image = rgb_image()
    req = parse_request(png_bytes(image), metadata(match_against={"dinov2_vits14": "p-0"}, reset_history=True))
    assert req.probe_id == "p-1"
    assert req.session_id == "s-1"
    assert req.models == ["dinov2_vits14"]
    assert req.match_against == {"dinov2_vits14": "p-0"}
    assert req.reset_history is True
    assert req.image.shape == (224, 224, 3)
    assert req.image.dtype == np.uint8
    np.testing.assert_array_equal(req.image, image)


def test_rgba_drops_alpha():
    image = rgb_image()
    rgba = np.dstack([image, np.full((224, 224), 128, dtype=np.uint8)])
    req = parse_request(png_bytes(rgba), metadata())
    assert req.image.shape == (224, 224, 3)
    np.testing.assert_array_equal(req.image, image)


def test_defaults():
    body = json.loads(metadata())
    del body["match_against"], body["reset_history"]
    req = parse_request(png_bytes(rgb_image()), json.dumps(body))
    assert req.match_against == {}
    assert req.reset_history is False


def test_match_against_null_means_absent():
    req = parse_request(png_bytes(rgb_image()), metadata(match_against=None))
    assert req.match_against == {}


def test_match_against_ignores_unrequested_models():
    req = parse_request(png_bytes(rgb_image()), metadata(match_against={"dinov2_vits14": "p-0", "other": "p-9"}))
    assert req.match_against == {"dinov2_vits14": "p-0"}


GOOD_PNG = png_bytes(rgb_image())

BAD_CASES = {
    "not_png": (b"definitely not a png", metadata()),
    "truncated_png": (GOOD_PNG[: len(GOOD_PNG) // 2], metadata()),
    "jpeg": (None, metadata()),  # filled below
    "wrong_size": (png_bytes(rgb_image(256)), metadata()),
    "grayscale": (png_bytes(rgb_image(), "L"), metadata()),
    "palette": (png_bytes(rgb_image(), "P"), metadata()),
    "metadata_not_json": (GOOD_PNG, "{nope"),
    "metadata_list": (GOOD_PNG, "[1, 2]"),
    "schema_2": (GOOD_PNG, metadata(schema_version=2)),
    "empty_probe_id": (GOOD_PNG, metadata(probe_id="")),
    "no_models": (GOOD_PNG, metadata(models=[])),
    "duplicate_models": (GOOD_PNG, metadata(models=["dinov2_vits14", "dinov2_vits14"])),
    "match_against_int": (GOOD_PNG, metadata(match_against={"dinov2_vits14": 3})),
    "schema_true": (GOOD_PNG, metadata(schema_version=True)),  # JSON true == 1 in Python
    "schema_float": (GOOD_PNG, metadata(schema_version=1.0)),
    "schema_string": (GOOD_PNG, metadata(schema_version="1")),
    "match_against_list": (GOOD_PNG, metadata(match_against=[])),
    "match_against_false": (GOOD_PNG, metadata(match_against=False)),
    "match_against_zero": (GOOD_PNG, metadata(match_against=0)),
    "match_against_empty_string": (GOOD_PNG, metadata(match_against="")),
    "match_against_empty_value": (GOOD_PNG, metadata(match_against={"dinov2_vits14": ""})),
    "reset_not_bool": (GOOD_PNG, metadata(reset_history="yes")),
}


def _jpeg_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgb_image()).save(buffer, format="JPEG")
    return buffer.getvalue()


BAD_CASES["jpeg"] = (_jpeg_bytes(), metadata())


@pytest.mark.parametrize("case", sorted(BAD_CASES))
def test_rejects(case):
    image_bytes, meta = BAD_CASES[case]
    with pytest.raises(ContractError) as info:
        parse_request(image_bytes, meta)
    assert info.value.status == 400
    assert info.value.code == "invalid_input"
    assert info.value.message


def test_error_keeps_probe_id():
    with pytest.raises(ContractError) as info:
        parse_request(b"garbage", metadata())
    assert info.value.probe_id == "p-1"


def test_build_result_shape():
    labels = np.full((224, 224), 3, dtype=np.uint8)
    result = build_result(
        "dinov2_vits14",
        labels,
        [0, 0, 0, 2, 0, 0, 0, 0],
        "nearest",
        None,
        {"encode": 1.0, "cluster": 2.0, "featup": None, "assign": 0.5, "match": None},
    )
    assert set(result) == {
        "model_id", "width", "height", "cluster_map_png_base64", "attention",
        "attention_norm", "upsampler", "matched_to", "timings_ms",
    }
    assert (result["width"], result["height"]) == (224, 224)
    assert result["attention_norm"][3] == 1.0
    assert result["attention_norm"][0] == 0.0
    np.testing.assert_array_equal(decode_label_png(result["cluster_map_png_base64"]), labels)
    json.dumps(result)  # must be JSON-serializable


def _png_with_header_size(width: int, height: int) -> bytes:
    """A valid 1x1 PNG whose IHDR claims another size (CRC fixed up)."""
    import struct
    import zlib

    data = bytearray(png_bytes(rgb_image(1)))
    ihdr = 8  # after the PNG signature: length(4) type(4) data(13) crc(4)
    data[ihdr + 8 : ihdr + 16] = struct.pack(">II", width, height)
    crc = zlib.crc32(bytes(data[ihdr + 4 : ihdr + 21])) & 0xFFFFFFFF
    data[ihdr + 21 : ihdr + 25] = struct.pack(">I", crc)
    return bytes(data)


def test_decompression_bomb_is_400():
    with pytest.raises(ContractError) as info:
        parse_request(_png_with_header_size(20000, 20000), metadata())
    assert info.value.status == 400
    assert info.value.probe_id == "p-1"


def test_oversized_header_rejected_before_decoding():
    # Claims 4000x4000 but holds 1x1 of pixel data: must fail on size, not on decoding.
    with pytest.raises(ContractError) as info:
        parse_request(_png_with_header_size(4000, 4000), metadata())
    assert "224x224" in info.value.message
