# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    batch_lines = []
    batch_audios = []
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for line in _iter_jsonl(args.input_jsonl):

            batch_lines.append(line)
            batch_audios.append(line['audio'])

            if len(batch_lines) >= BATCH_INFER_NUM:
                enc_res = tokenizer_12hz.encode(batch_audios)
                for code, sample in zip(enc_res.audio_codes, batch_lines):
                    sample['audio_codes'] = code.cpu().tolist()
                    out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                batch_lines.clear()
                batch_audios.clear()

        if batch_audios:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, sample in zip(enc_res.audio_codes, batch_lines):
                sample['audio_codes'] = code.cpu().tolist()
                out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
