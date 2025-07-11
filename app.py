import streamlit as st
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit page config
st.set_page_config(page_title="🛍️ E-Commerce Chatbot", page_icon="🛒", layout="wide")
st.title("🛒 E-Commerce Chatbot Assistant")
st.markdown("Hi! I’m your smart shop assistant. Ask me anything about products, shipping, returns, or payments.")

# Add basic styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { text-align: center; color: #7c0a02; font-size: 40px; }
    .stButton>button {
        background-color: maroon;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Gemini model with memory
@st.cache_resource
def init_bot():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    memory = ConversationBufferMemory()
    return ConversationChain(llm=llm, memory=memory, verbose=False)

conversation = init_bot()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Ask me a question...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Custom prompt with FAQs included
    faq_prompt = f"""
You are a helpful and friendly AI assistant for an online store.

Here are some frequently asked questions and their correct answers:

**Shipping FAQs**  
Q: How long does delivery take?  
A: Delivery usually takes 3–5 business days.

Q: Do you offer free shipping?  
A: Yes, we offer free shipping on orders over $50.

Q: Can I track my order?  
A: Yes! Use your order ID on our tracking page to check the status.

**Returns FAQs**  
Q: Can I return a product if I don’t like it?  
A: Yes, returns are accepted within 14 days of delivery.

Q: What is your return policy?  
A: You can return unused products within 14 days for a full refund.

Q: How do I send something back?  
A: Visit our returns page, fill out the form, and follow the shipping instructions.

**Payments FAQs**  
Q: What payment methods do you accept?  
A: We accept Visa, MasterCard, PayPal, and Cash on Delivery.

Q: Is it safe to pay online?  
A: Yes, we use secure payment gateways and SSL encryption for all transactions.

Now answer this customer question in a friendly and short tone:

Customer: {user_input}
Assistant:
"""

    # Get model response
    response = conversation.predict(input=faq_prompt)

    # Show assistant reply
    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
