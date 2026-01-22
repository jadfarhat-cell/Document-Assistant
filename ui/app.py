import gradio as gr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import settings

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0a0a0f",
    body_background_fill_dark="#0a0a0f",
    background_fill_primary="#101014",
    background_fill_primary_dark="#101014",
    background_fill_secondary="#18181c",
    background_fill_secondary_dark="#18181c",
    border_color_primary="#27272a",
    border_color_primary_dark="#27272a",
    body_text_color="#fafafa",
    body_text_color_dark="#fafafa",
    body_text_color_subdued="#a1a1aa",
    body_text_color_subdued_dark="#a1a1aa",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_dark="#2563eb",
    button_primary_background_fill_hover="#3b82f6",
    button_primary_background_fill_hover_dark="#3b82f6",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#18181c",
    button_secondary_background_fill_dark="#18181c",
    button_secondary_background_fill_hover="#27272a",
    button_secondary_background_fill_hover_dark="#27272a",
    button_secondary_text_color="#fafafa",
    button_secondary_text_color_dark="#fafafa",
    input_background_fill="#18181c",
    input_background_fill_dark="#18181c",
    input_border_color="#27272a",
    input_border_color_dark="#27272a",
    input_border_color_focus="#2563eb",
    input_border_color_focus_dark="#2563eb",
    block_background_fill="#101014",
    block_background_fill_dark="#101014",
    block_border_color="#27272a",
    block_border_color_dark="#27272a",
    block_label_background_fill="#18181c",
    block_label_background_fill_dark="#18181c",
    block_label_text_color="#a1a1aa",
    block_label_text_color_dark="#a1a1aa",
    block_title_text_color="#fafafa",
    block_title_text_color_dark="#fafafa",
    shadow_drop="0 4px 6px -1px rgba(0, 0, 0, 0.4)",
    shadow_drop_lg="0 10px 15px -3px rgba(0, 0, 0, 0.5)",
    block_shadow="none",
    block_shadow_dark="none",
)

custom_css = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
.main-header { text-align: center; padding: 2rem 0; border-bottom: 1px solid #27272a; margin-bottom: 1.5rem; }
.main-header h1 { background: linear-gradient(135deg, #ffffff 0%, #3b82f6 50%, #60a5fa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 2.5rem !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; }
.main-header p { color: #a1a1aa !important; font-size: 1.1rem !important; }
.tab-nav { border-bottom: 1px solid #27272a !important; gap: 0 !important; }
.tab-nav button { border: none !important; background: transparent !important; padding: 0.75rem 1.5rem !important; color: #a1a1aa !important; font-weight: 500 !important; border-bottom: 2px solid transparent !important; margin-bottom: -1px !important; transition: all 0.2s ease !important; }
.tab-nav button.selected { color: #ffffff !important; border-bottom-color: #2563eb !important; background: transparent !important; }
.tab-nav button:hover { color: #fafafa !important; }
.toast-container { position: fixed; top: 20px; right: 20px; z-index: 9999; pointer-events: none; }
.toast { background: #18181c !important; border: 1px solid #27272a !important; border-radius: 10px !important; padding: 1rem 1.25rem !important; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5) !important; animation: toastIn 0.3s ease-out, toastOut 0.3s ease-in 2.7s forwards !important; min-width: 280px; color: #fafafa !important; }
.toast.success { border-left: 3px solid #22c55e !important; }
.toast.warning { border-left: 3px solid #eab308 !important; }
.toast.error { border-left: 3px solid #ef4444 !important; }
.toast-title { font-weight: 600 !important; margin-bottom: 0.25rem !important; color: #ffffff !important; }
.toast-message { font-size: 0.875rem !important; color: #a1a1aa !important; }
@keyframes toastIn { from { opacity: 0; transform: translateX(100px); } to { opacity: 1; transform: translateX(0); } }
@keyframes toastOut { from { opacity: 1; transform: translateX(0); } to { opacity: 0; transform: translateX(100px); } }
.status-row { display: none !important; }
textarea, input[type="text"] { background: #18181c !important; border: 1px solid #27272a !important; border-radius: 8px !important; color: #fafafa !important; transition: border-color 0.2s ease !important; }
textarea:focus, input[type="text"]:focus { border-color: #2563eb !important; outline: none !important; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important; }
.primary { background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important; border: none !important; box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.35) !important; transition: all 0.2s ease !important; }
.primary:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.45) !important; }
input[type="range"] { accent-color: #2563eb !important; }
.upload-area { border: 2px dashed #27272a !important; border-radius: 12px !important; background: #101014 !important; transition: all 0.2s ease !important; }
.upload-area:hover { border-color: #2563eb !important; background: #18181c !important; }
.markdown-text { color: #fafafa !important; }
.markdown-text code { background: #18181c !important; padding: 0.2rem 0.4rem !important; border-radius: 4px !important; color: #60a5fa !important; }
.markdown-text table { border-collapse: collapse !important; width: 100% !important; }
.markdown-text th, .markdown-text td { border: 1px solid #27272a !important; padding: 0.75rem !important; text-align: left !important; }
.markdown-text th { background: #18181c !important; color: #a1a1aa !important; font-weight: 500 !important; }
.footer { text-align: center; padding: 1.5rem; border-top: 1px solid #27272a; margin-top: 2rem; color: #71717a !important; }
.footer a { color: #60a5fa !important; text-decoration: none !important; }
.footer a:hover { text-decoration: underline !important; }
"""

pipeline = RAGPipeline(
    collection_name=settings.chroma_collection,
    persist_directory=settings.chroma_persist_dir,
    model=settings.ollama_model,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    top_k=settings.top_k,
)


def check_status():
    stats = pipeline.get_stats()
    if not stats["ollama_connected"]:
        return f"""<div class="toast warning" id="status-toast">
            <div class="toast-title">Warning</div>
            <div class="toast-message">Ollama not connected. Run 'ollama serve'</div>
        </div>
        <script>setTimeout(() => document.getElementById('status-toast')?.remove(), 3000);</script>"""
    return f"""<div class="toast success" id="status-toast">
        <div class="toast-title">System Ready</div>
        <div class="toast-message">{stats['vector_store']['count']} chunks indexed · Model: {stats['current_model']}</div>
    </div>
    <script>setTimeout(() => document.getElementById('status-toast')?.remove(), 3000);</script>"""


def make_toast(title, message, status="success"):
    return f"""<div class="toast {status}" id="toast-{hash(message) % 10000}">
        <div class="toast-title">{title}</div>
        <div class="toast-message">{message}</div>
    </div>
    <script>setTimeout(() => document.querySelector('.toast')?.remove(), 3500);</script>"""


def upload_file(file):
    if file is None:
        return make_toast("No File", "Please select a file to upload", "warning")
    try:
        result = pipeline.ingest_file(file.name)
        return make_toast("Upload Complete", f"{result['filename']} · {result['chunks']} chunks indexed", "success")
    except Exception as e:
        return make_toast("Upload Failed", str(e), "error")


def upload_directory(dir_path):
    if not dir_path:
        return make_toast("No Path", "Please enter a directory path", "warning")
    path = Path(dir_path)
    if not path.exists():
        return make_toast("Not Found", f"Directory not found: {dir_path}", "error")
    try:
        results = pipeline.ingest_directory(dir_path)
        if not results:
            return make_toast("No Files", "No supported files found in directory", "warning")
        return make_toast("Upload Complete", f"{len(results)} files indexed successfully", "success")
    except Exception as e:
        return make_toast("Upload Failed", str(e), "error")


def query_documents(question, top_k):
    if not question:
        return "Please enter a question.", ""
    stats = pipeline.get_stats()
    if not stats["ollama_connected"]:
        return "Ollama not connected. Run `ollama serve`", ""
    if stats["vector_store"]["count"] == 0:
        return "No documents indexed. Upload some documents first.", ""
    try:
        result = pipeline.query(question, top_k=int(top_k), stream=False)
        sources_text = "**Sources:**\n"
        for source in result["sources"]:
            sources_text += f"- {source}\n"
        return result["answer"], sources_text
    except Exception as e:
        return f"Error: {str(e)}", ""


def clear_index():
    pipeline.clear()
    return make_toast("Cleared", "All indexed documents have been removed", "success")


with gr.Blocks(title="RAG Document Assistant", theme=theme, css=custom_css) as demo:
    toast_output = gr.HTML(elem_classes=["toast-container"])

    gr.HTML("""
    <div class="main-header">
        <h1>RAG Document Assistant</h1>
        <p>Upload documents and ask questions — powered by local LLMs</p>
    </div>
    """)

    with gr.Row():
        status_btn = gr.Button("Check Status", variant="secondary", size="sm")
        status_btn.click(check_status, outputs=toast_output)

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(label="Your Question", placeholder="Ask a question about your documents...", lines=2)
                    top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of sources to retrieve")
                    query_btn = gr.Button("Ask", variant="primary")
                with gr.Column(scale=4):
                    answer_output = gr.Textbox(label="Answer", lines=10, interactive=False)
                    sources_output = gr.Markdown(label="Sources")
            query_btn.click(query_documents, inputs=[question_input, top_k_slider], outputs=[answer_output, sources_output])
            question_input.submit(query_documents, inputs=[question_input, top_k_slider], outputs=[answer_output, sources_output])

        with gr.Tab("Upload"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload a File")
                    file_input = gr.File(label="Select a document", file_types=[".pdf", ".docx", ".md", ".txt", ".py", ".js", ".ts", ".json"])
                    upload_btn = gr.Button("Upload File", variant="primary")
                    upload_btn.click(upload_file, inputs=file_input, outputs=toast_output)
                with gr.Column():
                    gr.Markdown("### Upload a Directory")
                    dir_input = gr.Textbox(label="Directory Path", placeholder="C:/path/to/your/documents")
                    dir_btn = gr.Button("Ingest Directory", variant="primary")
                    dir_btn.click(upload_directory, inputs=dir_input, outputs=toast_output)

        with gr.Tab("Settings"):
            gr.Markdown(f"""
### Current Configuration

| Setting | Value |
|---------|-------|
| Model | `{settings.ollama_model}` |
| Chunk Size | {settings.chunk_size} |
| Chunk Overlap | {settings.chunk_overlap} |
| Top K | {settings.top_k} |

*Edit `.env` file or set `RAG_*` environment variables to customize.*
            """)
            clear_btn = gr.Button("Clear All Documents", variant="stop")
            clear_btn.click(clear_index, outputs=toast_output)

    gr.HTML("""
    <div class="footer">
        <p><strong>Supported formats:</strong> PDF, DOCX, Markdown, TXT, Python, JavaScript, TypeScript, JSON</p>
        <p style="margin-top: 0.5rem;">Powered by <a href="https://ollama.com" target="_blank">Ollama</a> · Built with <a href="https://gradio.app" target="_blank">Gradio</a></p>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
