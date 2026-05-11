import os
import sys
import streamlit as st


def _show_env_help_and_stop():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")

    st.set_page_config(page_title="Medika-AI", layout="centered")
    st.error("Incompatible Python runtime for offline chat model.")
    st.write(
        "This app is running with a Python environment where `transformers` and "
        "`torch` are not compatible for offline generation."
    )
    st.code(
        "Active Python:\n"
        f"{sys.executable}\n\n"
        "Run this command from the project root:\n"
        f"\"{venv_python}\" -m streamlit run src/chat_launcher.py"
    )
    st.stop()


# Quick compatibility gate before importing the heavier chat module.
try:
    import transformers
    import torch

    def _ver_tuple(v):
        base = str(v).split("+", 1)[0]
        parts = []
        for piece in base.split("."):
            if piece.isdigit():
                parts.append(int(piece))
            else:
                break
        return tuple(parts)

    if _ver_tuple(transformers.__version__) >= (5,) and _ver_tuple(torch.__version__) < (2, 4):
        _show_env_help_and_stop()
except Exception:
    _show_env_help_and_stop()

from chat import run

st.set_page_config(page_title="Medika-AI", layout="centered")
# Simple launcher to run the chat UI as a separate Streamlit app
run()
