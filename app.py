# ============================================================
# BDA_Project2_GroupNo2 - MLii Online Course Grant AI Assistant
# RAG Pipeline: FAISS + HuggingFace Embeddings + Groq LLM
# ============================================================

import streamlit as st
import os
import fitz
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

st.set_page_config(page_title="BDA_Project2_GroupNo2", page_icon="🎓", layout="wide")

# ── Header ─────────────────────────────
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
    # ปรับ Prompt ให้สั้นลงเพื่อป้องกัน Token Limit Error
    prompt = f"""คุณเป็น AI ผู้ช่วยตอบคำถามเกี่ยวกับทุน MLii
ข้อมูลอ้างอิง:
{context}

คำถาม: {question}

คำแนะนำ: ตรวจสอบข้อมูลอ้างอิงและสรุปรายชื่อคอร์สทั้งหมดที่พบออกมาเป็นข้อๆ ให้ครบถ้วน"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ API: {str(e)}\n\nคำแนะนำ: ลองลดความซับซ้อนของคำถาม หรือรอสักครู่แล้วลองใหม่ครับ"

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

if "messages"      not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("พิมพ์คำถามเกี่ยวกับทุน MLii...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            # ปรับ k เป็น 15 เพื่อความสมดุลระหว่างความครบถ้วนและ API Limit
            results = db.similarity_search(query, k=15)
            context = "\n\n".join([doc.page_content for doc in results])

        if client:
            with st.spinner("🤖 Generating answer..."):
                answer = generate_answer(query, context, client)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("⚠️ ไม่พบ GROQ_API_KEY")
            st.write(context[:1000] + "...")
