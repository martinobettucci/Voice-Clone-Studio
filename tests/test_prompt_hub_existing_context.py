import modules.core_components.prompt_hub as prompt_hub


def test_build_target_instruction_replaces_existing_placeholder(monkeypatch):
    target_id = "test.prompt.inline_existing"
    monkeypatch.setitem(
        prompt_hub.PROMPT_TARGETS,
        target_id,
        {
            "label": "Test Inline Existing",
            "tool": "Test",
            "tab_id": "tab_test",
            "component_key": "test_field",
            "default_system_preset": "TTS / Voice",
            "template": "Existing:\n{existing}\n\nInstruction:\n{instruction}",
        },
    )

    built = prompt_hub._build_target_instruction(
        target_id=target_id,
        instruction="  write this  ",
        existing_text="  old text  ",
        apply_mode="append",
    )

    assert built == "Existing:\nold text\n\nInstruction:\nwrite this"
    assert "Previous existing content was:" not in built


def test_build_target_instruction_append_fallback_preamble(monkeypatch):
    target_id = "test.prompt.append_fallback"
    monkeypatch.setitem(
        prompt_hub.PROMPT_TARGETS,
        target_id,
        {
            "label": "Test Append Fallback",
            "tool": "Test",
            "tab_id": "tab_test",
            "component_key": "test_field",
            "default_system_preset": "TTS / Voice",
            "template": "User instruction:\n{instruction}",
        },
    )

    built = prompt_hub._build_target_instruction(
        target_id=target_id,
        instruction="build me",
        existing_text="already there",
        apply_mode="append",
    )

    assert built == (
        "Previous existing content was: already there\n\n"
        "To continue the content, follow these instructions: User instruction:\nbuild me"
    )


def test_build_target_instruction_replace_fallback_preamble(monkeypatch):
    target_id = "test.prompt.replace_fallback"
    monkeypatch.setitem(
        prompt_hub.PROMPT_TARGETS,
        target_id,
        {
            "label": "Test Replace Fallback",
            "tool": "Test",
            "tab_id": "tab_test",
            "component_key": "test_field",
            "default_system_preset": "TTS / Voice",
            "template": "User instruction:\n{instruction}",
        },
    )

    built = prompt_hub._build_target_instruction(
        target_id=target_id,
        instruction="build me",
        existing_text="already there",
        apply_mode="replace",
    )

    assert built == (
        "Previous existing content was: already there\n\n"
        "To replace the content, follow these instructions: User instruction:\nbuild me"
    )


def test_build_target_instruction_empty_existing_skips_preamble(monkeypatch):
    target_id = "test.prompt.empty_existing"
    monkeypatch.setitem(
        prompt_hub.PROMPT_TARGETS,
        target_id,
        {
            "label": "Test Empty Existing",
            "tool": "Test",
            "tab_id": "tab_test",
            "component_key": "test_field",
            "default_system_preset": "TTS / Voice",
            "template": "User instruction:\n{instruction}",
        },
    )

    built = prompt_hub._build_target_instruction(
        target_id=target_id,
        instruction="do this",
        existing_text="   ",
        apply_mode="append",
    )

    assert built == "User instruction:\ndo this"
    assert "Previous existing content was:" not in built


def test_generate_for_target_builds_user_message_with_existing_context(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        content = b'{"ok":true}'

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "generated"}}]}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("modules.core_components.prompt_hub.requests.post", fake_post)

    text, error = prompt_hub.generate_for_target(
        user_config={},
        target_id="voice_clone.text",
        instruction="make it dramatic",
        existing_text="Current draft line",
        apply_mode="append",
    )

    assert error is None
    assert text == "generated"
    assert captured["url"].endswith("/chat/completions")
    assert captured["json"]["messages"][1]["content"] == (
        "Previous existing content was: Current draft line\n\n"
        "To continue the content, follow these instructions: "
        "Write final spoken text for voice generation. "
        "Return only the exact text to speak, no extra labels or notes.\n\n"
        "User instruction:\nmake it dramatic"
    )


def test_generate_for_target_without_existing_text_keeps_previous_prompt_shape(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        content = b'{"ok":true}'

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "generated"}}]}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("modules.core_components.prompt_hub.requests.post", fake_post)

    text, error = prompt_hub.generate_for_target(
        user_config={},
        target_id="voice_clone.text",
        instruction="make it dramatic",
    )

    assert error is None
    assert text == "generated"
    assert captured["json"]["messages"][1]["content"] == (
        "Write final spoken text for voice generation. "
        "Return only the exact text to speak, no extra labels or notes.\n\n"
        "User instruction:\nmake it dramatic"
    )
