

import json
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Connect to OpenAI
client = OpenAI()

PROGRESS_PATH = Path("progress.json")

LESSONS =[
    {"id": 1, "title": "Course Introduction"},
    {"id": 2, "title": "Programming Basics"},
    {"id": 3, "title": "Introduction to Python Programming"},
    {"id": 4, "title": "Python Data Types and Operators"},
    {"id": 5, "title": "Conditional Statements and Loops"},
    {"id": 6, "title": "Python Functions"},

    {"id": 8, "title": "Introduction to Data Science"},
    {"id": 18, "title": "Introduction to Machine Learning"},

    {"id": 25, "title": "Introduction to Deep Learning"},
    {"id": 26, "title": "Artificial Neural Networks"},
    {"id": 27, "title": "Deep Neural Networks"},
    {"id": 28, "title": "TensorFlow"},
    {"id": 29, "title": "PyTorch"},
    {"id": 30, "title": "Model Optimization and Performance Improvement"},
    {"id": 31, "title": "Convolutional Neural Networks"},
    {"id": 32, "title": "Transfer Learning"},
    {"id": 33, "title": "Object Detection"},
    {"id": 34, "title": "Recurrent Neural Networks"},
    {"id": 35, "title": "Transformer Models for NLP"},
    {"id": 36, "title": "Autoencoders"},
    {"id": 37, "title": "Introduction to Natural Language Processing"},
    {"id": 38, "title": "Text Data Analysis"},
    {"id": 39, "title": "NLP Text Vectorization"},
    {"id": 40, "title": "Distributed Representations"},

    {"id": 44, "title": "Introduction to Generative AI Models"},
    {"id": 45, "title": "Large Language Models and LangChain"},
    {"id": 46, "title": "Advanced Prompt Engineering (Part 1)"},
    {"id": 47, "title": "Advanced Prompt Engineering (Part 2)"},
    {"id": 48, "title": "LangChain for LLM Apps (Part 1)"},
    {"id": 49, "title": "LangChain for LLM Apps (Part 2)"},
    {"id": 50, "title": "LLM Fine-Tuning and Customization"},
    {"id": 51, "title": "Benchmarking and Evaluating LLMs"},
]

# Progress schema (LESSONS-based)
def default_progress(): return {
        "current_level": "beginner",
        "sessions_completed": 0,
        "lessons": {
            str(lesson["id"]): {
                "completed_lessons": 0,
                "last_feedback": "",
                "total_questions": 0,
                "correct_questions": 0.0
            }
            for lesson in LESSONS
        }
    }

def load_progress():
    if not PROGRESS_PATH.exists():
        p = default_progress()
        save_progress(p)
        return p

    p = json.loads(PROGRESS_PATH.read_text())

    # Ensure top-level keys exist
    p.setdefault("current_level", "beginner")
    p.setdefault("sessions_completed", 0)

    # Ensure lessons container exists
    if "lessons" not in p or not isinstance(p["lessons"], dict):
        p["lessons"] = {}

    # Ensure each lesson has required fields
    for lesson in LESSONS:
        lid = str(lesson["id"])
        p["lessons"].setdefault(
            lid,
            {
                "completed_lessons": 0,
                "last_feedback": "",
                "total_questions": 0,
                "correct_questions": 0.0
            }
        )

        entry = p["lessons"][lid]
        entry.setdefault("completed_lessons", 0)
        entry.setdefault("last_feedback", "")
        entry.setdefault("total_questions", 0)
        entry.setdefault("correct_questions", 0.0)

    save_progress(p)
    return p

def save_progress(p):
    PROGRESS_PATH.write_text(json.dumps(p, indent=2))

def extract_practice_question(text: str) -> Optional[str]:
    marker = "Practice Question:"
    if marker not in text:
        return None
    return text.split(marker, 1)[1].strip()

def score_from_feedback(feedback: str) -> float:
    t = feedback.lower()
    if "incorrect" in t:
        return 0.0
    if "partially correct" in t:
        return 0.5
    if "correct" in t:
        return 1.0
    return 0.0

def call_llm(system_prompt: str, user_prompt: str) -> str:
    resp = client.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.output_text

# ----- UI -----
def submit_lesson_question():
    q = st.session_state.get("lesson_question", "").strip()
    st.session_state["pending_lesson_question"] = q
    st.session_state["lesson_question"] = ""  # clear input safely via callback

def submit_global_question():
    q = st.session_state.get("global_question", "").strip()
    st.session_state["pending_global_question"] = q
    st.session_state["global_question"] = ""  # clear input safely via callback

def init_chat():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # list of {"role": "user"/"assistant", "content": "..."}

def next_question():
    # Clear only the parts that should reset for a new attempt
    st.session_state.pop("last_feedback", None)
    st.session_state.pop("question", None)
    st.session_state.pop("answer", None)# clear answer box too (only works if the text_area has a key)
st.set_page_config(page_title="ML Tutor Demo", layout="wide")
st.title("🤖 ML Tutor Demo")
st.caption("Curriculum-aligned: Lesson → Practice → Grade → Track progress")

progress = load_progress()

# Build options for the dropdown in lesson order
lesson_options = [f"Lesson {l['id']:02d}: {l['title']}" for l in LESSONS]
selected_label = st.selectbox("Choose a lesson:", lesson_options)

# Convert dropdown selection back to a lesson object
selected_index = lesson_options.index(selected_label)
selected_lesson = LESSONS[selected_index]

lesson_id = selected_lesson["id"]
lesson_title = selected_lesson["title"]
lesson_key = str(lesson_id)

col1, col2 = st.columns(2)

with col1:
    if st.button("Teach this lesson"):
        system_prompt = (
            "You are an AI teaching assistant for a structured Machine Learning course. "
            "Teach ONLY the concepts appropriate for the current lesson. "
            "Do not introduce advanced topics that belong to later lessons. "
            "Use simple explanations and short examples. "
            "End with a clearly labeled practice question on its own line starting with "
            "'Practice Question:'. Do NOT include the answer."
        )

        user_prompt = (
            f"Lesson {lesson_id}: {lesson_title}\n"
            "Teach this lesson clearly and step-by-step for a beginner learner."
        )

        explanation = call_llm(system_prompt, user_prompt)

        st.session_state["explanation"] = explanation
        st.session_state["question"] = extract_practice_question(explanation) or ""

        # Track "lesson taught"
        progress["sessions_completed"] += 1
        progress["lessons"][lesson_key]["completed_lessons"] += 1
        save_progress(progress)

with col2:
    attempts = progress["lessons"][lesson_key]["total_questions"]
    correct = progress["lessons"][lesson_key]["correct_questions"]
    acc = 0 if attempts == 0 else round((correct / attempts) * 100)

    st.metric("Attempts", attempts)
    st.metric("Accuracy", f"{acc}%")

st.write(f"**Current lesson:** Lesson {lesson_id:02d} — {lesson_title}")

if "explanation" in st.session_state:
    st.subheader("Lesson")
    st.write(st.session_state["explanation"])

    st.subheader("Practice Question")
    st.write(st.session_state.get("question", ""))

    st.subheader("Your Answer")
    answer = st.text_area("Type your answer here:", key="answer")

    if st.button("Check my answer"):
        if not st.session_state.get("question") or not answer.strip():
            st.warning("Click 'Teach this lesson' first and enter an answer.")
        else:
            system = (
                "You are a strict but kind grader. You give constructive criticism leading with the positives first then dives into the negatives. In the first sentence say exactly "
                "'correct', 'partially correct', or 'incorrect'. Then explain and give the ideal answer."
            )
            user = f"Question: {st.session_state['question']}\n\nStudent answer: {answer}"
            feedback = call_llm(system, user)

            # Persist feedback
            st.session_state["last_feedback"] = feedback

            # Update progress
            progress["lessons"][lesson_key]["total_questions"] += 1
            progress["lessons"][lesson_key]["correct_questions"] += score_from_feedback(feedback)
            progress["lessons"][lesson_key]["last_feedback"] = feedback
            save_progress(progress)

            st.success("Saved progress!")

    # Show feedback after grading (persists across reruns)
    if "last_feedback" in st.session_state:
        st.subheader("Feedback")
        st.write(st.session_state["last_feedback"])

        st.button("Next question / refresh", on_click=next_question)
    st.divider()
    st.header("💬 Ask Questions")

    init_chat()

    tab_lesson, tab_global = st.tabs(["Ask about this lesson", "Ask anything (ML/AI)"])

    with tab_lesson:
        st.subheader(f"Lesson Q&A — Lesson {lesson_id:02d}: {lesson_title}")

        st.text_area(
            "Ask a question related to this lesson:",
            placeholder="e.g., Can you explain this concept using a real-world example?",
            key="lesson_question"
        )

        # Button triggers callback that copies text into pending + clears the box
        st.button("Ask (Lesson)", key="ask_lesson_btn", on_click=submit_lesson_question)

        # Process pending question AFTER the click
        pending = st.session_state.get("pending_lesson_question", "")
        if pending:
            st.session_state["chat_history"].append(
                {"role": "user", "content": f"[Lesson {lesson_id:02d}] {pending}"}
            )

            system_prompt = (
                "You are a helpful AI teaching assistant for a structured ML/AI course. "
                "Answer clearly in plain English. "
                "Stay aligned to the current lesson when possible. "
                "If the student asks something beyond the lesson, give a brief helpful answer and "
                "suggest what lesson/topic it belongs to."
            )

            lesson_context = st.session_state.get("explanation", "")
            user_prompt = (
                f"Current lesson: Lesson {lesson_id:02d} — {lesson_title}\n"
                f"Lesson context (may be empty):\n{lesson_context}\n\n"
                f"Student question: {pending}"
            )

            with st.spinner("Thinking..."):
                answer_text = call_llm(system_prompt, user_prompt)

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer_text}
            )

            # Clear pending so it doesn't re-run on refresh
            st.session_state["pending_lesson_question"] = ""

        st.markdown("### Chat History")
        for msg in st.session_state["chat_history"][-10:]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Tutor:** {msg['content']}")

    with tab_global:
        st.subheader("Global Q&A (ML/AI)")

        st.text_area(
            "Ask anything about ML/AI (not limited to the lesson):",
            placeholder="e.g., What's the difference between overfitting and underfitting?",
            key="global_question"
        )

        st.button("Ask (Global)", key="ask_global_btn", on_click=submit_global_question)

        pending = st.session_state.get("pending_global_question", "")
        if pending:
            st.session_state["chat_history"].append({"role": "user", "content": pending})

            system_prompt = (
                "You are a helpful ML/AI tutor. "
                "Answer clearly in plain English. "
                "Use short examples when helpful. "
                "If the question is vague, ask ONE clarifying question and provide a best-effort answer."
            )

            with st.spinner("Thinking..."):
                answer_text = call_llm(system_prompt, pending)

            st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})

            st.session_state["pending_global_question"] = ""

        st.markdown("### Chat History")
        for msg in st.session_state["chat_history"][-10:]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Tutor:** {msg['content']}")

    # Optional: clear chat
    if st.button("Clear chat history"):
        st.session_state["chat_history"] = []
        st.success("Chat cleared.")
    st.divider()
if st.button("Reset progress"):
    progress = default_progress()
    save_progress(progress)
    st.success("Progress reset.")
    st.rerun()
