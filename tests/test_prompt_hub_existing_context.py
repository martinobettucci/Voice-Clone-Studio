import modules.core_components.prompt_hub as prompt_hub


def test_apply_prompt_placeholders_replaces_supported_tokens():
    result = prompt_hub._apply_prompt_placeholders(
        "Mode={mode}; Existing={existing}; Instruction={instruction}",
        existing_text="draft line",
        instruction_text="write line",
        apply_mode="append",
    )
    assert result == "Mode=append; Existing=draft line; Instruction=write line"


def test_apply_prompt_placeholders_replaces_available_emotions_token():
    result = prompt_hub._apply_prompt_placeholders(
        "Emotions={available_emotions}",
        available_emotions_text="calm, excited",
    )
    assert result == "Emotions=calm, excited"


def test_resolve_prompt_assistant_language_uses_settings_default_when_requested():
    assert (
        prompt_hub.resolve_prompt_assistant_language(
            {"prompt_assistant_default_language": "French"},
            prompt_hub.PROMPT_ASSISTANT_LANGUAGE_USE_SETTINGS,
        )
        == "French"
    )


def test_get_prompt_assistant_language_instruction_for_italian():
    assert prompt_hub.get_prompt_assistant_language_instruction("Italian") == "Genera in italiano."


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
        "User instruction:\nmake it dramatic\n\n"
        "Generate the output in English."
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
        "User instruction:\nmake it dramatic\n\n"
        "Generate the output in English."
    )


def test_generate_for_target_resolves_placeholders_in_instruction_and_custom_system(monkeypatch):
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
        instruction="Continue from {existing} in {mode} mode.",
        custom_system_override="System sees mode={mode}; instruction={instruction}; existing={existing}",
        existing_text="draft one",
        apply_mode="append",
    )

    assert error is None
    assert text == "generated"
    assert captured["json"]["messages"][0]["content"] == (
        "System sees mode=append; instruction=Continue from draft one in append mode.; existing=draft one"
    )
    assert captured["json"]["messages"][1]["content"] == (
        "Previous existing content was: draft one\n\n"
        "To continue the content, follow these instructions: "
        "Write final spoken text for voice generation. "
        "Return only the exact text to speak, no extra labels or notes.\n\n"
        "User instruction:\nContinue from draft one in append mode.\n\n"
        "Generate the output in English."
    )


def test_generate_for_target_conversation_uses_available_emotions_in_system_prompt(monkeypatch):
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
        user_config={"emotions": {"happy": {}, "sad": {}}},
        target_id="conversation.script",
        instruction="Two friends discuss weekend plans.",
    )

    assert error is None
    assert text == "generated"
    system_prompt = captured["json"]["messages"][0]["content"]
    assert "[n]: (emotion) text" in system_prompt
    assert "happy, sad" in system_prompt


def test_generate_for_target_uses_selected_language_override(monkeypatch):
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
        user_config={"prompt_assistant_default_language": "English"},
        target_id="voice_clone.text",
        instruction="Write a short line",
        output_language="French",
    )

    assert error is None
    assert text == "generated"
    assert captured["json"]["messages"][1]["content"].endswith("Genere la sortie en francais.")
