import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

load_dotenv()

st.set_page_config(    page_title="AI Chatbot Mentor", page_icon="🤖", layout="wide")


st.sidebar.title("📘 Learning Modules")

modules = [
    "Python",
    "SQL",
    "Power BI",
    "Exploratory Data Analysis (EDA)",
    "Machine Learning (ML)",
    "Deep Learning (DL)",
    "Generative AI (Gen AI)",
    "Agentic AI"
]

selected_module = st.sidebar.radio("Select a module:", modules)
st.sidebar.markdown("---")


if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.clear()
    st.rerun()


st.title("🤖 AI Chatbot Mentor")
st.caption(f"📌 Current Module: **{selected_module}**")


llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash",temperature=0.7)


domain_check_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer ONLY with YES or NO."),
        (
            "human",
            "Is this question related to the module '{module}'?\n\nQuestion: {question}"
        )
    ]
)

domain_check_chain = domain_check_prompt | llm


mentor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a dedicated AI mentor for the module: {module}. "
            "Answer clearly, structurally, and educationally. "
            "Do not answer questions outside this module."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

mentor_chain = mentor_prompt | llm


chat_history = StreamlitChatMessageHistory(key="chat_history")

def get_limited_history(session_id):
    chat_history.messages = chat_history.messages[-15:]
    return chat_history

conversation = RunnableWithMessageHistory(
    mentor_chain,
    get_limited_history,
    input_messages_key="input",
    history_messages_key="history"
)


welcome_messages = {
    "Python": "🐍 **Welcome to Python AI Mentor**\n\nI am your dedicated mentor for Python.\nHow can I help you today?",
    "SQL": "🗄️ **Welcome to SQL AI Mentor**\n\nI am your dedicated mentor for SQL.\nHow can I help you today?",
    "Power BI": "📊 **Welcome to Power BI AI Mentor**\n\nI am your dedicated mentor for Power BI.\nHow can I help you today?",
    "Exploratory Data Analysis (EDA)": "📈 **Welcome to EDA AI Mentor**\n\nI am your dedicated mentor for Exploratory Data Analysis.\nHow can I help you today?",
    "Machine Learning (ML)": "🤖 **Welcome to Machine Learning AI Mentor**\n\nI am your dedicated mentor for ML.\nHow can I help you today?",
    "Deep Learning (DL)": "🧠 **Welcome to Deep Learning AI Mentor**\n\nI am your dedicated mentor for DL.\nHow can I help you today?",
    "Generative AI (Gen AI)": "✨ **Welcome to Generative AI Mentor**\n\nI am your dedicated mentor for Gen AI.\nHow can I help you today?",
    "Agentic AI": "🧩 **Welcome to Agentic AI Mentor**\n\nI am your dedicated mentor for Agentic AI.\nHow can I help you today?",
}


if "last_module" not in st.session_state:
    st.session_state.last_module = None

if st.session_state.last_module != selected_module:
    chat_history.clear()
    st.chat_message("assistant").write(welcome_messages[selected_module])
    st.session_state.last_module = selected_module


for msg in chat_history.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    elif msg.type == "ai":
        st.chat_message("assistant").write(msg.content)


user_input = st.chat_input("Ask me anything...")

if user_input:
    st.chat_message("user").write(user_input)

    domain_result = domain_check_chain.invoke(
        {
            "module": selected_module,
            "question": user_input
        }
    ).content.strip().upper()

    if domain_result != "YES":
        rejection_msg = (
            "Sorry, I don’t know about this question. "
            "Please ask something related to the selected module."
        )
        st.chat_message("assistant").write(rejection_msg)

    else:
        response = conversation.invoke(
            {
                "input": user_input,
                "module": selected_module
            },
            config={"configurable": {"session_id": "default"}}
        )

        st.chat_message("assistant").write(response.content)


st.sidebar.markdown("---")
st.sidebar.subheader("📥 Download Conversation")

def build_chat_text():
    lines = []
    lines.append(f"AI Chatbot Mentor - Conversation Log")
    lines.append(f"Module: {selected_module}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n" + "=" * 50 + "\n")

    for msg in chat_history.messages:
        if msg.type == "human":
            lines.append(f"User: {msg.content}\n")
        elif msg.type == "ai":
            lines.append(f"Mentor: {msg.content}\n")

    return "\n".join(lines)

chat_text = build_chat_text()

st.sidebar.download_button(
    label="⬇️ Download Chat (.txt)",
    data=chat_text,
    file_name=f"AI_Mentor_{selected_module.replace(' ', '_')}_Chat.txt",
    mime="text/plain"
)
