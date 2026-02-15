from pathlib import Path

import soundfile as sf

from modules.core_components.tools.generated_output_save import (
    parse_modal_submission,
    sanitize_output_name,
    save_generated_output,
)


def test_parse_modal_submission_normal_and_cancel():
    matched, cancelled, name = parse_modal_submission(
        "save_demo_my_file_1739663000000",
        "save_demo_",
    )
    assert matched is True
    assert cancelled is False
    assert name == "my_file"

    matched, cancelled, name = parse_modal_submission("save_demo_cancel_1739", "save_demo_")
    assert matched is True
    assert cancelled is True
    assert name is None

    matched, cancelled, name = parse_modal_submission("save_other_name_1739", "save_demo_")
    assert matched is False
    assert cancelled is False
    assert name is None


def test_sanitize_output_name_matches_modal_rules():
    assert sanitize_output_name("  My file?.wav  ") == "My_filewav"
    assert sanitize_output_name("hello/world") == "helloworld"
    assert sanitize_output_name(None) == ""


def test_save_generated_output_from_tuple_and_filepath(tmp_path: Path):
    output_dir = tmp_path / "output"

    # Tuple/list payload save
    out1 = save_generated_output(
        audio_value=(16000, [0.0, 0.1, -0.1, 0.0]),
        output_dir=output_dir,
        raw_name="tuple_audio",
        metadata_text="Seed: 1",
    )
    assert out1.exists()
    data, sr = sf.read(out1)
    assert sr == 16000
    assert len(data) == 4
    assert out1.with_suffix(".txt").exists()

    # Filepath payload save
    src = tmp_path / "src.wav"
    sf.write(str(src), [0.0, 0.2, 0.0], 24000)
    out2 = save_generated_output(
        audio_value=str(src),
        output_dir=output_dir,
        raw_name="from_path",
    )
    assert out2.exists()
    data2, sr2 = sf.read(out2)
    assert sr2 == 24000
    assert len(data2) == 3
