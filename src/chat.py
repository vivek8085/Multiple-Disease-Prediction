import os
import streamlit as st
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datetime import datetime

# st.set_page_config(page_title="Medika-AI", layout="centered")

SYSTEM_PROMPT = (
    "You are Medika, a medical education assistant. "
    "Answer ONLY the user's specific question. "
    "Do NOT give generic health lists. "
    "Do NOT repeat previous answers. "
    "Keep responses short and relevant. "
    "If information is insufficient, say what is missing. "
    "End every response with: 'Consult a qualified doctor for diagnosis.'"
)

HF_TOKEN = os.getenv("HF_TOKEN")

# path for the single CSV file storing chat query/response pairs
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "chat_log.csv"))


def save_chat_to_csv(messages, path=CSV_PATH):
    # collect user->assistant pairs
    rows = []
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            user_text = m.get("content", "")
            # find next assistant reply (if any)
            response_text = ""
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "assistant":
                    response_text = messages[j].get("content", "")
                    break
            rows.append({
                "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
                "user": "user",
                "query": user_text,
                "response": response_text,
            })

    if not rows:
        return

    try:
        df = pd.DataFrame(rows)
        file_exists = os.path.exists(path)
        df.to_csv(path, mode="a", index=False, header=not file_exists)
    except Exception:
        # if anything goes wrong with pandas IO, fail silently to avoid crashing the UI
        return


def append_row_to_csv(row, path=CSV_PATH):
    try:
        df = pd.DataFrame([row])
        file_exists = os.path.exists(path)
        df.to_csv(path, mode="a", index=False, header=not file_exists)
    except Exception:
        return


def append_last_pair(messages, path=CSV_PATH):
    # find last user message that has a following assistant reply
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_text = messages[i].get("content", "")
            # find next assistant after this index
            response_text = ""
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "assistant":
                    response_text = messages[j].get("content", "")
                    break
            if user_text:
                row = {
                    "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "user": "user",
                    "query": user_text,
                    "response": response_text,
                }
                append_row_to_csv(row, path)
            break


def run():
    # initialize chat storage for history
    if "chats" not in st.session_state:
        st.session_state.chats = []

    with st.sidebar:
        st.header("🗂️ Chats")

        if st.button("➕ New Chat"):
            # start a fresh conversation (do not auto-save here)
            st.session_state.messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            # clear uploaded image context
            st.session_state.uploaded_image_name = None
            st.session_state.uploaded_image_bytes = None
            st.rerun()

        st.subheader("History")
        # display entries from the CSV file (most recent first). Each row is a single query/response pair.
        if os.path.exists(CSV_PATH):
            try:
                # read CSV without assuming a header (some users supply CSV without header row)
                df = pd.read_csv(CSV_PATH, header=None, names=["timestamp", "user", "query", "response"]) 
                if df.empty:
                    st.write("No previous chats")
                else:
                    # normalize and filter out empty queries
                    rows = df.to_dict(orient="records")
                    valid_rows = []
                    for r in rows:
                        q = r.get("query", "")
                        if pd.isna(q):
                            continue
                        if isinstance(q, str) and q.strip() == "":
                            continue
                        valid_rows.append(r)

                    if not valid_rows:
                        st.write("No previous chats")
                    else:
                        # show most recent valid entries
                        for idx, row in enumerate(reversed(valid_rows[-200:])):
                            display_idx = len(valid_rows) - idx
                            ts = row.get("timestamp", "") or ""
                            q = row.get("query", "")
                            q_text = "" if pd.isna(q) else str(q)
                            snippet = (q_text[:80] + "...") if len(q_text) > 80 else q_text
                            label = f"{display_idx} — {ts} — {snippet}" if ts else f"{display_idx} — {snippet}"
                            if st.button(label, key=f"hist_btn_{display_idx}"):
                                # normalize row values to strings before storing in session
                                clean_row = {
                                    "timestamp": row.get("timestamp", "") if not pd.isna(row.get("timestamp", "")) else "",
                                    "user": row.get("user", "user") if not pd.isna(row.get("user", "user")) else "user",
                                    "query": q_text,
                                    "response": "" if pd.isna(row.get("response", "")) else str(row.get("response", "")),
                                }
                                st.session_state.selected_history = clean_row
                                st.rerun()
            except Exception:
                st.write("Unable to read history")
        else:
            st.write("No previous chats")
        # no separate saved-excel display; history above shows saved rows

    @st.cache_resource
    def load_offline_model():
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )
        model.eval()
        return tokenizer, model

    def run_online(prompt):
        if not HF_TOKEN:
            return "Online mode not configured. Hugging Face token missing."

        api_url = (
            "https://api-inference.huggingface.co/models/"
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.85,
                "repetition_penalty": 1.2
            }
        }

        response = requests.post(
            api_url, headers=headers, json=payload, timeout=60
        )

        if response.status_code != 200:
            return "Online model error. Check internet or token."

        return response.json()[0]["generated_text"]


    # Show UI
    st.title("🩺 Medika AI – Medical Assistant")
    # Modes/settings removed; default to Offline
    MODE = "Offline"
    # st.caption("Mode: 📴 Offline")
    st.caption("⚠️ Educational use only. Not a substitute for a doctor.")

    # If a history row was selected from the sidebar, show full details here
    if "selected_history" in st.session_state and st.session_state.get("selected_history"):
        row = st.session_state.get("selected_history")
        st.subheader("Selected History")
        st.write(f"**Timestamp:** {row.get('timestamp','')}")
        st.write(f"**User:** {row.get('user','user')}")
        st.write(f"**Query:** {row.get('query','')}")
        st.write(f"**Response:** {row.get('response','')}")
        c1, c2 = st.columns([1, 1])
        if c1.button("Load into chat", key="load_hist_main"):
            st.session_state.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row.get("query", "")},
                {"role": "assistant", "content": row.get("response", "")},
            ]
            st.session_state.selected_history = None
            st.rerun()
        if c2.button("Close", key="close_hist_main"):
            st.session_state.selected_history = None
            st.rerun()

    # Image upload for visual context
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png","pdf"])
    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        st.image(img_bytes, caption=uploaded_file.name, use_column_width=True)
        st.session_state.uploaded_image_name = uploaded_file.name
        st.session_state.uploaded_image_bytes = img_bytes

    if MODE.startswith("Offline"):
        tokenizer, model = load_offline_model()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a medical question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        history = st.session_state.messages[-4:]
        prompt = ""

        # If an image was uploaded, prepend a short image-description hint
        image_hint = ""
        if st.session_state.get("uploaded_image_name"):
            image_hint = (
                f"[Image uploaded: {st.session_state.get('uploaded_image_name')}. "
                "Refer to the image when answering the question; if more detail is needed, ask for clarification.]\n"
            )
        for m in history:
            if m["role"] == "system":
                prompt += f"<|system|>\n{m['content']}\n"
            elif m["role"] == "user":
                prompt += f"<|user|>\n{m['content']}\n"
            else:
                prompt += f"<|assistant|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        # prepend image hint to the full prompt
        if image_hint:
            prompt = image_hint + prompt

        with st.chat_message("assistant"):
            with st.spinner("Medika is thinking..."):
                if MODE.startswith("Offline"):
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        top_p=0.85,
                        repetition_penalty=1.2,
                        do_sample=True
                    )
                    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    reply = decoded.split("<|assistant|>")[-1].strip()
                else:
                    reply = run_online(prompt)

                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        # append the latest user->assistant pair to the Excel file
        try:
            append_last_pair(st.session_state.messages)
        except Exception:
            pass


if __name__ == "__main__":
    run()