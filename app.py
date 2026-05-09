# ============================================================
# BDA_Project2_GroupNo2 - MLii Online Course Grant AI Assistant
# RAG Pipeline: FAISS + HuggingFace Embeddings + Groq LLM
# รองรับ: PDF | JPG/PNG (OCR) | Word + Google Drive links
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
    prompt = f"""คุณเป็น AI ผู้ช่วยที่ตอบคำถามเกี่ยวกับทุน MLii Online Course Grant
ตอบตามข้อมูลใน Context เท่านั้น ถ้าไม่มีข้อมูลให้บอกว่าไม่ทราบ
ตอบเป็นภาษาเดียวกับคำถาม (ไทยหรืออังกฤษ)

Context:
{context}

คำถาม: {question}

คำตอบ:"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
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
        st.session_state.preset_query = "เอกสารแบบฟอร์มการขอรับทุนมีอะไรบ้าง"
with col2:
    if st.button("💰 รายละเอียดคอร์ส"):
        st.session_state.preset_query = "คอร์สเรียนมีกี่คอร์ส อะไรบ้าง"
with col3:
    if st.button("📅 ขั้นตอนการสมัคร"):
        st.session_state.preset_query = "ขั้นตอนการสมัครทุนมีอะไรบ้าง"

if "messages"      not in st.session_state: st.session_state.messages = []
if "preset_query"  not in st.session_state: st.session_state.preset_query = ""

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
            context = "\n\n".join([doc.page_content for doc in results])

        if client:
            with st.spinner("🤖 Generating answer..."):
                answer = generate_answer(query, context, client)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("⚠️ ไม่พบ GROQ_API_KEY — แสดงผลเฉพาะ Retrieved Documents")
            st.write(context[:600] + "...")
            st.session_state.messages.append({"role": "assistant", "content": context[:600]})

        # ---------------------------------------------------------
        # ดึงรูปภาพมาแสดงทันทีหลังคำตอบ (ไม่ต้องซ่อนใน expander)
        # ---------------------------------------------------------
        st.markdown("🖼️ **ภาพเอกสารที่เกี่ยวข้อง:**")
        
        # สร้างคอลัมน์เพื่อจัดเรียงรูปภาพตามแนวนอน
        cols = st.columns(len(results)) 
        
        for i, doc in enumerate(results):
            source   = doc.metadata.get('source', '')
            page_num = doc.metadata.get('page', 0)
            
            with cols[i]:
                # กรณีเป็นไฟล์ PDF
                if os.path.exists(source) and source.lower().endswith('.pdf'):
                    try:
                        pdf_doc = fitz.open(source)
                        pg  = pdf_doc.load_page(page_num)
                        pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2)) # ซูมภาพให้ชัดขึ้น 2 เท่า
                        st.image(pix.tobytes("png"), caption=f"หน้า {page_num+1} ({os.path.basename(source)})", use_container_width=True)
                        pdf_doc.close()
                    except Exception as e:
                        pass
                        
                # กรณีเป็นไฟล์รูปภาพ (JPG/PNG)
                elif os.path.exists(source) and source.lower().endswith(('.jpg', '.jpeg', '.png')):
                    st.image(source, caption=os.path.basename(source), use_container_width=True)

        # เก็บ Expander ไว้เผื่อต้องการดูข้อความ Text แบบเต็มๆ (ซ่อนรูปไปแล้ว)
        with st.expander("📄 View Raw Text (ข้อความดิบ)"):
            for i, doc in enumerate(results):
                source   = doc.metadata.get('source', '')
                page_num = doc.metadata.get('page', 0)
                st.markdown(f"**จากไฟล์:** {os.path.basename(source)} (หน้า {page_num+1})")
                st.write(doc.page_content)
                st.divider()
