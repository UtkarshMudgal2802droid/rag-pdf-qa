import gradio as gr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import re
import torch
import traceback
import html

print("Loading RAG models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
qa_model.eval()
print("Models loaded successfully!")

sample_content = """Machine Learning: A Comprehensive Guide

Chapter 1: Introduction to Machine Learning

Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Machine learning algorithms build mathematical models based on sample data, known as training data, to make predictions or decisions.

There are three main types of machine learning:

1. Supervised Learning: The algorithm is trained on labeled data, where the correct output is already known. Examples include classification and regression problems. Common algorithms include linear regression, logistic regression, decision trees, and support vector machines.

2. Unsupervised Learning: The algorithm works with unlabeled data and tries to find hidden patterns or structures. Examples include clustering and dimensionality reduction. Common algorithms include K-means clustering, hierarchical clustering, and principal component analysis (PCA).

3. Reinforcement Learning: The algorithm learns by interacting with an environment and receiving rewards or penalties. The agent learns to make decisions by maximizing cumulative rewards. Common applications include game playing, robotics, and autonomous vehicles.

Chapter 2: Neural Networks and Deep Learning

Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes (neurons) organized in layers. Deep learning uses neural networks with multiple hidden layers.

Key components of neural networks include:
- Input Layer: Receives the input data
- Hidden Layers: Process the data through weighted connections
- Output Layer: Produces the final prediction or classification
- Activation Functions: Introduce non-linearity (ReLU, Sigmoid, Tanh)
- Loss Functions: Measure the difference between predicted and actual values
- Optimizers: Update weights to minimize loss (Adam, SGD, RMSprop)

Chapter 3: Applications of Machine Learning

Machine learning has numerous real-world applications:

1. Healthcare: Disease diagnosis, drug discovery, medical imaging analysis
2. Finance: Fraud detection, algorithmic trading, credit scoring
3. Transportation: Autonomous vehicles, route optimization, traffic prediction
4. E-commerce: Product recommendations, customer segmentation, demand forecasting
5. Natural Language Processing: Sentiment analysis, machine translation, chatbots
6. Computer Vision: Image recognition, object detection, facial recognition

Chapter 4: Challenges and Future Directions

Despite its success, machine learning faces several challenges:
- Data Quality: Garbage in, garbage out. Poor quality data leads to poor models
- Overfitting: Models that work well on training data but fail on new data
- Interpretability: Many ML models are black boxes with unclear decision processes
- Bias and Fairness: Models can perpetuate or amplify biases in training data
- Computational Cost: Training large models requires significant computing resources

Conclusion

Machine learning is transforming how we solve complex problems and make decisions. As algorithms improve and computing power increases, we can expect even more innovative applications in the future.
"""

def safe_html(text):
    return html.escape(text).replace("\n", "<br>")

def extract_text_from_pdf(pdf_file):
    try:
        if pdf_file is None:
            return None, "<p style='color:#ff6b6b;font-weight:700;'>Error: No PDF file uploaded.</p>"
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        if not text.strip():
            return None, "<p style='color:#ff6b6b;font-weight:700;'>Error: No text extracted from PDF.</p>"
        text = re.sub(r"\s+", " ", text).strip()
        return text, f"<p style='color:#7CFC98;font-weight:700;'>Successfully extracted {len(text)} characters from {len(reader.pages)} pages</p>"
    except Exception as e:
        return None, f"<p style='color:#ff6b6b;font-weight:700;'>Error extracting PDF: {safe_html(str(e))}</p>"

def chunk_text(text, chunk_size=600, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            bp = max(last_period, last_newline)
            if bp > chunk_size * 0.5:
                chunk = chunk[:bp + 1]
                end = start + bp + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks if chunks else [text]

def create_embeddings(chunks):
    return embedding_model.encode(chunks, show_progress_bar=False)

def retrieve_relevant_chunks(query, chunks, embeddings, top_k=3):
    q_emb = embedding_model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in idxs:
        score = float(sims[i]) * 100
        if score > 10:
            results.append((chunks[i], score))
    return results

def generate_answer_from_context(query, context):
    try:
        inputs = qa_tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        start = int(torch.argmax(outputs.start_logits))
        end = int(torch.argmax(outputs.end_logits)) + 1
        if end <= start or (end - start) < 4:
            raise ValueError("QA span not informative")
        tokens = inputs["input_ids"][0][start:end]
        ans = qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if query.lower() in ans.lower():
            ans = ans.replace(query, "", 1).strip()
        if len(ans) < 20:
            raise ValueError("Answer too short")
        return ans
    except Exception:
        paras = [p.strip() for p in re.split(r"\n{2,}|\.\s+", context) if p.strip()]
        keywords = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
        matched = []
        for p in paras:
            p_l = p.lower()
            matches = sum(1 for kw in keywords if kw in p_l)
            if matches > 0:
                matched.append(p.strip())
        if matched:
            answer = "\n\n".join(matched)
            return answer[:1400] + ("..." if len(answer) > 1400 else "")
        return context[:1400] + ("..." if len(context) > 1400 else "")

def format_paragraphs(text, max_chars=5000):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = re.split(r"(?<=[.!?])\s+", text)
    out = []
    total = 0
    for p in paras:
        if total >= max_chars:
            out.append("<p style='color:#9aa6b2;font-style:italic;'>...truncated</p>")
            break
        take = p if total + len(p) <= max_chars else p[: max_chars - total]
        out.append(f"<p style='margin:8px 0;line-height:1.65;color:#d8e7ff;'>{safe_html(take)}</p>")
        total += len(take)
    return "".join(out)

def build_answer_box(answer):
    answer = answer.strip()
    if answer and answer[-1] not in ".!?":
        answer += "."
    return f"""
    <div style='background:linear-gradient(180deg, rgba(14,21,34,0.98), rgba(8,14,20,0.98));
                padding:18px;border-radius:12px;border:1px solid rgba(124,252,152,0.35);
                border-left:5px solid #28a745; max-height:430px; overflow:auto; color:#e8f7eb;'>
      <div style='font-size:16px;font-weight:800;color:#eaffef;margin-bottom:12px;'>Answer</div>
      <div style='white-space:pre-wrap;line-height:1.8;font-size:15px;color:#dff5e2;'>{safe_html(answer)}</div>
    </div>
    """

def answer_question(pdf_file, question, top_k=3, use_sample=False):
    try:
        log_lines = []
        if use_sample:
            text = sample_content
            log_lines.append("Using sample document")
            status_html = f"<p style='color:#7CFC98;font-weight:700;'>Loaded sample document ({len(text)} characters)</p>"
        else:
            text, status_html = extract_text_from_pdf(pdf_file)
            if text is None:
                return "", status_html
            log_lines.append("Uploaded PDF processed")
        log_lines.append(f"Text length: {len(text)} characters")
        chunks = chunk_text(text, chunk_size=600, overlap=80)
        log_lines.append(f"Created {len(chunks)} chunks (600 chars, 80 overlap)")
        embeddings = create_embeddings(chunks)
        log_lines.append(f"Embeddings shape: {embeddings.shape}")
        if not question or not question.strip():
            return "", "<p style='color:#ff6b6b;font-weight:700;'>Error: Please enter a question.</p>"
        relevant = retrieve_relevant_chunks(question, chunks, embeddings, top_k=top_k)
        if not relevant:
            return "", "<p style='color:#ff6b6b;font-weight:700;'>No relevant chunks found. Try rephrasing.</p>"
        log_lines.append(f"Found {len(relevant)} relevant chunks")
        for i, (_, score) in enumerate(relevant, start=1):
            log_lines.append(f"Chunk {i} similarity: {score:.2f}%")
        context = " ".join([c for c, _ in relevant])
        log_lines.append(f"Combined context size: {len(context)} characters")
        answer = generate_answer_from_context(question, context)

        context_html = format_paragraphs(context, max_chars=5000)
        process_html = "".join(f"<div style='margin-bottom:7px;color:#aeb8c8;'>{html.escape(line)}</div>" for line in log_lines)
        answer_box = build_answer_box(answer)

        summary_html = f"""
        <div style='background:#071122;padding:14px;border-radius:10px;color:#aeb8c8;border:1px solid rgba(255,255,255,0.08);'>
          <div style='font-weight:800;color:#9ad1ff;margin-bottom:10px;'>Summary</div>
          <div style='line-height:1.7;'><strong style='color:#d9fdd3'>Document pages:</strong> 1</div>
          <div style='line-height:1.7;'><strong style='color:#d9fdd3'>Total chunks:</strong> {len(chunks)}</div>
          <div style='line-height:1.7;'><strong style='color:#d9fdd3'>Relevant chunks found:</strong> {len(relevant)}</div>
          <div style='line-height:1.7; word-break:break-word;'><strong style='color:#d9fdd3'>Question:</strong> {html.escape(question)}</div>
        </div>
        """

        log_html = f"""
        <div style='display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;'>
          <div style='flex:1;min-width:320px;'>
            <div style='font-weight:800;color:#9ad1ff;margin-bottom:6px;'>Context preview</div>
            <div style='background:#0f1724;padding:14px;border-radius:10px;max-height:340px;overflow:auto;color:#d8e7ff;border:1px solid rgba(255,255,255,0.08);'>
              {context_html}
            </div>
            <div style='height:12px;'></div>
            <div style='font-weight:800;color:#9ad1ff;margin-bottom:6px;'>Process log</div>
            <div style='background:#071122;padding:12px;border-radius:10px;color:#aeb8c8;border:1px solid rgba(255,255,255,0.08);'>
              {process_html}
            </div>
          </div>
          <div style='width:430px;flex-shrink:0;'>
            {answer_box}
            <div style='height:12px;'></div>
            {summary_html}
          </div>
        </div>
        """
        return answer, log_html
    except Exception:
        tb = traceback.format_exc()
        err_text = "Error: An unexpected error occurred. See details below."
        err_html = f"<div style='color:#ff6b6b;font-weight:800;'>Error during processing</div><pre style='color:#ffd1d1;background:#2b2730;padding:12px;border-radius:8px;overflow:auto;max-height:420px;'>{html.escape(tb)}</pre>"
        return err_text, err_html

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # PDF Question Answering System using RAG
        Retrieval-Augmented Generation (RAG) application for PDF documents
        Upload a PDF document or use the sample document to ask questions
        Powered by HuggingFace Transformers + Sentence Transformers
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload PDF or Use Sample Document")
            pdf_input = gr.File(label="Upload PDF Document", file_types=[".pdf"], file_count="single")
            use_sample_checkbox = gr.Checkbox(label="Use Sample Document (Machine Learning Guide)", value=False)
            gr.Markdown("*Check this to use sample document, or upload your own PDF*\n\n**You can either:** Upload your own PDF OR check this box and leave PDF empty")
            question_input = gr.Textbox(label="Your Question", placeholder="Try: What are the types of machine learning?", lines=3, max_lines=5)
            top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Chunks to Retrieve")
            submit_btn = gr.Button("Ask Question", variant="primary", size="lg")

        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="Answer (plain text)", lines=8)
            log_output = gr.HTML(value="")

    gr.Markdown(
        """
        ### Sample Questions to Try:
        - What are the types of machine learning?
        - What is supervised learning?
        - What are neural networks?
        - What are applications of machine learning?
        - What challenges does machine learning face?
        """
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[pdf_input, question_input, top_k_slider, use_sample_checkbox],
        outputs=[answer_output, log_output],
    )

    gr.Markdown("---\n**GitHub**: https://github.com/UtkarshMudgal2802droid/rag-pdf-qa")

if __name__ == "__main__":
    demo.launch(share=True)
