# ============================================================
# BDA_Project2_GroupNo2 - MLii Online Course Grant AI Assistant
# RAG Pipeline: FAISS + HuggingFace Embeddings + Groq LLM
# รองรับ: PDF | JPG/PNG (Vision) | Word + Google Drive links
# ============================================================
# Group Members:
# 6631501006 Kanyakrit Bowonsuwan
# 6631501014 Koontorn Koonkue
# 6631501167 Yossawat Bukbun
# ============================================================

import streamlit as st
import os
import fitz
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

st.set_page_config(page_title="BDA_Project2_GroupNo2", page_icon="🎓", layout="wide")

# ── Header (ตาม Requirement 3.4) ─────────────────────────────
st.title("🎓 BDA_Project2_GroupNo2")
st.markdown("""
### MLii Online Course Grant AI Assistant

**Group Members:**
| Student ID | Name |
|---|---|
| 6631501006 | Kanyakrit Bowonsuwan |
| 6631501014 | Koontorn Koonkue |
| 6631501167 | Yossawat Bukbun |

*1501316 Business Data Analytics | Mae Fah Luang University*
""")
st.divider()

# ── Load FAISS Vector DB ─────────────────────────────────────
DB_PATH = "mlii_faiss_index"

@st.cache_resource
def load_vector_db():
    if not os.path.exists(DB_PATH):
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# ── Groq LLM ─────────────────────────────────────────────────
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "") or st.secrets.get("GROQ_API_KEY", "")
    return Groq(api_key=api_key) if api_key else None

def generate_answer(question, context, client):
    # ภาษาไทย 1 ตัวอักษร ≈ 2-3 tokens → จำกัด context ไว้ 1500 ตัวอักษร
    context_trimmed = context[:1500]

    prompt = f"""ตอบคำถามต่อไปนี้โดยใช้ข้อมูลจาก Context เท่านั้น
ถ้าไม่มีข้อมูลให้บอกว่าไม่ทราบ ตอบเป็นภาษาไทย

Context:
{context_trimmed}

คำถาม: {question}
คำตอบ:"""
    resp = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    return resp.choices[0].message.content

# ── Load Resources ────────────────────────────────────────────
with st.spinner("⏳ Loading AI model..."):
    db     = load_vector_db()
    client = get_groq_client()

if db is None:
    st.error(f"❌ ไม่พบ Vector Database ที่ '{DB_PATH}'")
    st.stop()

st.success("✅ AI model loaded and ready!")

# ── Chat Interface ────────────────────────────────────────────
st.subheader("💬 ถามคำถามเกี่ยวกับทุน MLii Online Course")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📋 เอกสารการขอรับทุน"):
        st.session_state.preset_query = "มีลิงค์เอกสารการขอรับทุนไหม"
with col2:
    if st.button("💰 รายละเอียดคอร์ส"):
        st.session_state.preset_query = "คอร์สเรียนมีกี่คอร์ส อะไรบ้าง"
with col3:
    if st.button("📅 ขั้นตอนการสมัคร"):
        st.session_state.preset_query = "ขั้นตอนการสมัครทุนมีอะไรบ้าง"

if "messages"     not in st.session_state: st.session_state.messages = []
if "preset_query" not in st.session_state: st.session_state.preset_query = ""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("พิมพ์คำถามเกี่ยวกับทุน MLii...")
if not query and st.session_state.preset_query:
    query = st.session_state.preset_query
    st.session_state.preset_query = ""

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            results = db.similarity_search(query, k=3)
            # จำกัดแต่ละ chunk ไม่เกิน 400 ตัวอักษร (ภาษาไทยกิน token เยอะ)
            context = "\n\n".join([doc.page_content[:400] for doc in results])

        if client:
            with st.spinner("🤖 Generating answer..."):
                answer = generate_answer(query, context, client)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("⚠️ ไม่พบ GROQ_API_KEY — แสดงผลเฉพาะ Retrieved Documents")
            st.write(context[:600] + "...")
            st.session_state.messages.append({"role": "assistant", "content": context[:600]})

        with st.expander("📄 View Source Documents"):
            for i, doc in enumerate(results):
                source   = doc.metadata.get('source', '')
                page_num = doc.metadata.get('page', 0)
                doc_type = doc.metadata.get('type', 'pdf')
                title    = os.path.basename(source)

                st.markdown(
                    f"<span style='background:#007bff;color:white;padding:4px 10px;"
                    f"border-radius:12px;font-weight:bold;'>Result {i+1}</span> "
                    f"&nbsp;<b>{title}</b> | Page {page_num+1} | Type: {doc_type}",
                    unsafe_allow_html=True
                )

                if os.path.exists(source) and source.lower().endswith('.pdf'):
                    try:
                        pdf_doc = fitz.open(source)
                        pg  = pdf_doc.load_page(page_num)
                        pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2))
                        st.image(pix.tobytes("png"), use_container_width=True)
                        pdf_doc.close()
                    except Exception as e:
                        st.error(f"Could not render page: {e}")

                elif os.path.exists(source) and source.lower().endswith(('.jpg', '.jpeg', '.png')):
                    st.image(source, use_container_width=True)

                with st.expander("View Raw Text"):
                    st.write(doc.page_content)
                st.divider()
