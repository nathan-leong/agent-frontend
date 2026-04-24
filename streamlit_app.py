"""
Usage:
    streamlit run streamlit_app.py
"""

import json
import os
import time
import logging
import uuid
import io
import re
import tempfile
import httpx
import boto3
import streamlit as st
from urllib.parse import quote
from docx import Document
from markdown_pdf import MarkdownPdf, Section
import fitz  # PyMuPDF

# logging.basicConfig(level=logging.DEBUG)  # root handler; optional if you only set botocore
# boto3.set_stream_logger("botocore", logging.DEBUG)
# optional: see lower-level HTTP details
# logging.getLogger("urllib3").setLevel(logging.DEBUG)
# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NEXTGEN PDM Agent",
    page_icon="nextgen logo.png",
    layout="wide",
)

SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
# ---------------------------------------------------------------------------
# Helper functions (self-contained, mirrors invoke.py patterns)
# ---------------------------------------------------------------------------


def _load_cognito_config() -> dict | None:
    """Load cognito_config.json from the sample directory."""
    path = os.path.join(SAMPLE_DIR, "cognito_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _find_project_dir() -> str:
    """Find the agentcore project directory (subdirectory containing agentcore/)."""
    for entry in os.listdir(SAMPLE_DIR):
        candidate = os.path.join(SAMPLE_DIR, entry)
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "agentcore")):
            return candidate
    raise FileNotFoundError(
        "No agentcore project directory found. Run 'agentcore create' first."
    )


def _find_in_json(obj, key):
    """Recursively search for a key in nested JSON."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            result = _find_in_json(v, key)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_in_json(item, key)
            if result:
                return result
    return None


def _resolve_agent_arn() -> str:
    """Read the deployed agent ARN from deployed-state.json.

    Searches for runtimeArn recursively to work across CLI versions.
    """
    project_dir = _find_project_dir()
    state_file = os.path.join(project_dir, "agentcore", ".cli", "deployed-state.json")
    if not os.path.exists(state_file):
        raise FileNotFoundError(
            "No deployed-state.json found. Run 'agentcore deploy -y' first."
        )
    with open(state_file) as f:
        state = json.load(f)
    arn = _find_in_json(state, "runtimeArn")
    if arn:
        return arn
    raise ValueError("No deployed agent found. Run 'agentcore deploy -y' first.")


def _get_bearer_token(config: dict) -> str:
    """Authenticate with Cognito and return an access token."""
    cognito = boto3.client("cognito-idp", region_name=config["region"])
    auth = cognito.initiate_auth(
        ClientId=config["client_id"],
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={
            "USERNAME": config["username"],
            "PASSWORD": config["password"],
        },
    )
    return auth["AuthenticationResult"]["AccessToken"]


def _parse_event_stream(response: dict) -> str:
    """Extract text from the boto3 EventStream response."""
    parts: list[str] = []
    for event in response.get("response", []):
        raw = (
            event
            if isinstance(event, bytes)
            else event.get("chunk", {}).get("bytes", b"")
        )
        if raw:
            try:
                decoded = json.loads(raw.decode("utf-8"))
                if isinstance(decoded, str):
                    parts.append(decoded)
                elif isinstance(decoded, dict):
                    content = decoded.get("content", [])
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(c["text"])
                        elif isinstance(c, str):
                            parts.append(c)
                    if not content and "message" in decoded:
                        msg = decoded["message"]
                        if isinstance(msg, dict):
                            for c in msg.get("content", []):
                                if isinstance(c, dict) and c.get("type") == "text":
                                    parts.append(c["text"])
            except Exception:
                parts.append(raw.decode("utf-8"))
    raw = "\n".join(parts) if parts else "(no response)"
    return _sse_stream_to_plain_text(raw)


def _sse_stream_to_plain_text(s: str) -> str:
    """Turn SSE-style lines (`data: "chunk"\\n`) into plain concatenated text.

    Agent runtimes often return newline-delimited `data: "<json-string>"` chunks.
    If the payload does not look like SSE, return it unchanged.
    """
    lines = s.splitlines()
    if not any(line.strip().startswith("data:") for line in lines if line.strip()):
        return s
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
            if isinstance(chunk, str):
                out.append(chunk)
            elif chunk is not None:
                out.append(str(chunk))
        except json.JSONDecodeError:
            out.append(payload)
    return "".join(out) if out else s

def _invoke_agent_streaming(
    agent_arn: str,
    region: str,
    prompt: str,
    bearer_token: str | None = None,
    session_id: str | None = None,
    on_chunk=None,  # callback for streaming UI
) -> dict:
    url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{quote(agent_arn, safe='')}/invocations?qualifier=DEFAULT"

    headers = {
        "Content-Type": "application/json",
    }

    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    if session_id:
        headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] = session_id

    payload = {"prompt": prompt}

    t0 = time.time()
    full_text = []

    try:
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()

                # Stream line-by-line (SSE format)
                for line in resp.iter_lines():
                    if not line:
                        continue

                    line = line.strip()

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        if isinstance(chunk, str):
                            text = chunk
                        else:
                            text = str(chunk)
                    except json.JSONDecodeError:
                        text = data

                    full_text.append(text)

                    # 🔥 STREAM OUT
                    if on_chunk:
                        on_chunk(text)

        elapsed = time.time() - t0

        return {
            "success": True,
            "text": "".join(full_text),
            "elapsed": elapsed,
            "error": None,
            "status_code": resp.status_code,
        }

    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "success": False,
            "text": None,
            "elapsed": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
            "status_code": getattr(exc, "response", None).status_code
            if hasattr(exc, "response") and exc.response
            else None,
        }


def _format_response(text: str) -> str:
    """Replace literal \\n sequences with real line breaks for display."""
    return text.replace("\\n", "\n")


def _truncate_arn(arn: str, max_len: int = 45) -> str:
    """Truncate an ARN for sidebar display."""
    if len(arn) <= max_len:
        return arn
    return arn[: max_len - 3] + "..."


def _create_markdown_file(question: str, answer: str, username: str) -> str:
    """Create a markdown string with the agent's response."""
    return answer


def _create_word_document(question: str, answer: str, username: str) -> io.BytesIO:
    """Create a Word document with the agent's response."""
    doc = Document()

    # Add answer - split by lines and add as paragraphs
    for line in answer.split('\n'):
        if line.strip():
            doc.add_paragraph(line)
        else:
            doc.add_paragraph("")

    # Save to BytesIO
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def _markdown_to_pdf(markdown_content: str) -> bytes:
    """Convert markdown content to PDF using markdown-pdf library with logo."""
    temp_pdf_path = tempfile.mktemp(suffix='.pdf')
    temp_pdf_with_logo = tempfile.mktemp(suffix='.pdf')

    try:
        # Add HTML spacing at the top for logo

        # Convert markdown to PDF - pass content directly
        pdf = MarkdownPdf()
        pdf.add_section(Section(markdown_content, toc=False))
        pdf.meta["title"] = "Document"
        pdf.save(temp_pdf_path)

        # Add logo and increase top margin using PyMuPDF
        logo_path = os.path.join(SAMPLE_DIR, "nextgen logo.png")
        doc = fitz.open(temp_pdf_path)

        # Create new document with shifted content
        new_doc = fitz.open()
        shift_down = 30  # 80 points = ~1.1 inches extra top margin

        for page_num, page in enumerate(doc):
            # Create new page with same dimensions
            new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)

            # Copy page content with vertical offset (shift down)
            new_page.show_pdf_page(
                new_page.rect + (0, shift_down, 0, 0),  # Shift down by 80 points
                doc,
                page_num
            )

            # Add logo to every page
            if os.path.exists(logo_path):
                logo_size = 40
                margin = 15
                page_width = new_page.rect.width
                x0 = page_width - logo_size - margin
                y0 = margin
                x1 = page_width - margin
                y1 = margin + logo_size

                rect = fitz.Rect(x0, y0, x1, y1)
                new_page.insert_image(rect, filename=logo_path, keep_proportion=True)

        # Save the modified document
        new_doc.save(temp_pdf_with_logo)
        new_doc.close()
        doc.close()

        # Read the modified PDF
        with open(temp_pdf_with_logo, 'rb') as f:
            pdf_bytes = f.read()

        return pdf_bytes
    finally:
        # Clean up temp files
        try:
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
            if os.path.exists(temp_pdf_with_logo):
                os.unlink(temp_pdf_with_logo)
        except:
            pass


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for key, default in {
    "logged_in": False,
    "jwt_token": None,
    "username": "",
    "chat_history": [],
    "agent_arn": None,
    "arn_error": None,
    "region": "us-east-1",
    "last_request": None,
    "login_error": None,
    "session_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
config = _load_cognito_config()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide sidebar toggle when not logged in */
    .login-page [data-testid="collapsedControl"] { display: none; }

    /* Login card styling */
    .login-card {
        padding: 2rem 0;
    }
    .login-header {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .login-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .login-desc {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-bottom: 2rem;
        line-height: 1.5;
    }

    /* Sidebar signed-in badge */
    .signed-in-badge {
        background: #065f46;
        color: #d1fae5;
        padding: 0.4rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: 500;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sidebar-meta {
        color: #9ca3af;
        font-size: 0.75rem;
        word-break: break-all;
        margin-bottom: 0.25rem;
    }

    /* Preset buttons row */
    .stButton > button {
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# SCREEN 1: LOGIN (full page, centered)
# =========================================================================
if not st.session_state.logged_in:
    # Hide sidebar on login page
    st.markdown(
        "<style>[data-testid='stSidebar'] { display: none; }</style>",
        unsafe_allow_html=True,
    )

    # Vertical spacer
    st.markdown("")
    st.markdown("")

    # Center column
    _, center, _ = st.columns([1, 2, 1])

    with center:
        # Display logo
        logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])
        with logo_col2:
            st.image("nextgen logo.png", width=150)

        st.markdown(
            "<h1 class='login-header'>NEXTGEN Partner Development Manager Agent</h1>",
            unsafe_allow_html=True,
        )

        if not config:
            st.error(
                "**cognito_config.json not found.** "
                "Run `python setup_cognito.py` before using this app."
            )
            st.stop()

        # Show any previous login error
        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        with st.form("login_form"):
            username = st.text_input(
                "Username",
                value=config.get("username", ""),
            )
            password = st.text_input(
                "Password",
                value=config.get("password", ""),
                type="password",
            )
            sign_in = st.form_submit_button("Sign In", use_container_width=True)

        if sign_in:
            if not username or not password:
                st.session_state.login_error = "Username and password are required."
                st.rerun()
            else:
                login_config = {**config, "username": username, "password": password}
                try:
                    with st.spinner("Authenticating with Cognito..."):
                        token = _get_bearer_token(login_config)

                    # Resolve agent ARN right away
                    with st.spinner("Resolving agent ARN..."):
                        try:
                            agent_arn = _resolve_agent_arn()
                        except Exception as exc:
                            agent_arn = None
                            st.session_state.arn_error = str(exc)

                    # Commit to session state
                    st.session_state.jwt_token = token
                    st.session_state.bearer_input = token  # pre-fill the token field
                    st.session_state.username = username
                    st.session_state.logged_in = True
                    st.session_state.agent_arn = agent_arn
                    st.session_state.region = config.get("region", "us-east-1")
                    st.session_state.session_id = str(uuid.uuid4())  # Generate session ID
                    st.session_state.login_error = None
                    st.rerun()

                except Exception as exc:
                    st.session_state.login_error = f"Login failed: {exc}"
                    st.rerun()

    st.stop()


# =========================================================================
# SCREEN 2: DASHBOARD (after login)
# =========================================================================

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        f"<div class='signed-in-badge'>Signed in as {st.session_state.username}</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.agent_arn:
        st.markdown(
            f"<p class='sidebar-meta'><b>Agent:</b> {_truncate_arn(st.session_state.agent_arn)}</p>",
            unsafe_allow_html=True,
        )
    elif st.session_state.arn_error:
        st.error(st.session_state.arn_error, icon="\u26a0\ufe0f")
    else:
        st.warning("Agent ARN not resolved.")

    st.markdown(
        f"<p class='sidebar-meta'><b>Region:</b> {st.session_state.region}</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.session_id:
        st.markdown(
            f"<p class='sidebar-meta'><b>Session ID:</b> {st.session_state.session_id[:8]}...</p>",
            unsafe_allow_html=True,
        )

    if st.button("Sign Out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()

    # --- Tools Section ---
    st.markdown("### 🛠️ Tools")

    # Tool 1: Copy Bearer Token
    with st.expander("📋 Copy Bearer Token", expanded=False):
        if st.session_state.jwt_token:
            st.code(st.session_state.jwt_token, language=None)
            st.caption("Copy the token above to use in other applications.")
        else:
            st.info("No token available")

    # Tool 2: Markdown to PDF Converter
    with st.expander("📄 Markdown to PDF Converter", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload markdown file",
            type=['md', 'markdown'],
            key="md_uploader",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Read the markdown content
            markdown_content = uploaded_file.read().decode('utf-8')

            # Convert to PDF
            pdf_bytes = _markdown_to_pdf(markdown_content)

            # Download button
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.pdf",
                mime="application/pdf",
                key="md_to_pdf_download",
                use_container_width=True
            )

# --- Main area ---
# Display logo in the app
logo_left, logo_right = st.columns([1, 9])
with logo_left:
    st.image("nextgen logo.png", width=50)
with logo_right:
    st.markdown("#### NEXTGEN Partner Development Manager Agent")

# st.caption("Clear the Bearer Token in the sidebar to see a 403 rejection. The agent retrieves the API key from AgentCore Identity — never hardcoded. The agent is deployed in AWS using AgentCore.")

# Chat history
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(_format_response(msg["content"]))

    # Add download buttons for assistant responses (outside chat_message)
    if msg["role"] == "assistant" and msg["content"]:
        # Find the corresponding user question
        user_question = ""
        if i > 0 and st.session_state.chat_history[i-1]["role"] == "user":
            user_question = st.session_state.chat_history[i-1]["content"]

        # Create markdown content
        markdown_content = _create_markdown_file(
            user_question,
            msg["content"],
            st.session_state.username
        )

        # Create Word document
        doc_buffer = _create_word_document(
            user_question,
            msg["content"],
            st.session_state.username
        )

        # Add download buttons with small styling
        col1, col2, col3, col4 = st.columns([1, 2, 2, 5])
        with col2:
            st.download_button(
                label="📄 Markdown",
                data=markdown_content,
                file_name=f"agent_response_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key=f"download_md_{i}",
                use_container_width=True
            )
        with col3:
            st.download_button(
                label="📘 Word",
                data=doc_buffer,
                file_name=f"agent_response_{time.strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"download_docx_{i}",
                use_container_width=True
            )

# Preset buttons
prompt_to_send = None

if len(st.session_state.chat_history) == 0:
    presets = [
        "What can you do?",
        "Show me all opportunities created this month.",
        "How much discounts do software partners receive through the distribution program discounts?",
        "What are the requirements for a partner to become Advanced Tier?"
    ]

    preset_cols = st.columns(len(presets))

    for i, preset in enumerate(presets):
        with preset_cols[i]:
            if st.button(preset, key=f"preset_{i}", use_container_width=True):
                prompt_to_send = preset

# Chat input
user_input = st.chat_input("Ask the agent...")
if user_input:
    prompt_to_send = user_input

# Send prompt
if prompt_to_send:
    if not st.session_state.agent_arn:
        st.error("Agent ARN not resolved. Deploy the agent first.")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt_to_send,
        })
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            streamed_text = [""]
            first_chunk_received = [False]

            spinner_container = st.empty()
            with spinner_container:
                st.spinner("Invoking agent...")

            # 🔥 Initialize message in chat history BEFORE streaming
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "",
            })
            message_index = len(st.session_state.chat_history) - 1

            def handle_chunk(chunk):
                if not first_chunk_received[0]:
                    first_chunk_received[0] = True
                    spinner_container.empty()

                streamed_text[0] += chunk

                # 🔥 Persist LIVE into session state
                st.session_state.chat_history[message_index]["content"] = streamed_text[0]

                placeholder.markdown(_format_response(streamed_text[0]))

            result = _invoke_agent_streaming(
                st.session_state.agent_arn,
                st.session_state.region,
                prompt_to_send,
                bearer_token=st.session_state.get("bearer_input", "").strip() or None,
                session_id=st.session_state.get("session_id"),
                on_chunk=handle_chunk,
            )

            if not first_chunk_received[0]:
                spinner_container.empty()

            # Rerun to show download buttons
            st.rerun()
# Last request details (collapsed)
if st.session_state.last_request:
    with st.expander("Last request details", expanded=False):
        req = st.session_state.last_request
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "Success" if req.get("success") else "Failed")
        with col2:
            st.metric("Response Time", f"{req.get('elapsed', 0):.2f}s")
        with col3:
            st.metric("HTTP Status", str(req.get("status_code", "N/A")))
        st.code(f"Authorization: {req.get('auth', 'N/A')}", language=None)
        if req.get("error"):
            st.error(req["error"])
        if req.get("text"):
            st.code(req["text"], language=None)