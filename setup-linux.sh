#!/bin/bash
# Linux installation helper for Voice Clone Studio
# This script helps with common Linux installation issues

sudo apt update
sudo apt install -y ffmpeg sox libsox-fmt-all
sudo apt-get update && sudo apt-get install -y espeak-ng libespeak-ng-dev libespeak-ng1


if pip install "zipvoice @ git+https://github.com/ysharma3501/LuxTTS.git"; then
    echo "LuxTTS installed successfully!"
else
    echo "zipvoice installation failed. LuxTTS will not be available."
fi

echo ""
echo "Prompt Manager now uses OpenAI-compatible endpoints directly."
echo "Optional local mode is available with Ollama (no llama.cpp install required)."
