import logging

from flask import Flask, jsonify, request
import openai
from langchain import VectorDBQA
from langchain_community.vectorstores import Chroma
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import CallbackManager
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import os

logging.basicConfig(filename='error.log', level=logging.ERROR)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Initialize embeddings and document search
embeddings = OpenAIEmbeddings()
docsearch = Chroma(embedding_function=embeddings, persist_directory="./knownledge/")

# Initialize the chat model
chat = ChatOpenAI(api_key=openai_api_key, temperature=0.01)

# Define the system prompt template
system_prompt = PromptTemplate(
    template="""
You are a shopping assistant and your name is EKKA AI that helps visitors find the right product from the store.
If you can't find the answer from the context, ask more questions. Do not make up products.
Once you have an answer, list down the products in a clear tone as a markdown list, include product url.
You also provide additional shopping support, including comparing products and making recommendations based on user preferences, always include product url.
Here are some rules you must follow:
1. Always provide accurate and relevant information.
2. If a product cannot be found, politely ask for more details.
3. When comparing products, use criteria such as price, rating, description, and features. Show high rating products.
4. When making recommendations, consider the user's preferences and criteria provided.
5. Do not fabricate information or products. Stick to the data provided.
6. Ask the user if they need a comparison. If the user provides product links, find and compare them. Provide the best product and awlays include product url.
----------------
{context} 
""",
    input_variables=["context"],
)

# Define the message prompt templates
system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Define the product search tool

product_search = VectorDBQA.from_chain_type(
    llm=chat, chain_type="stuff", vectorstore=docsearch
)


tools = [
    Tool(
        name="Product Search",
        func=product_search.run,
        description="Useful for when you need to find products available in the store and show product url, use criteria such as price, rating, description, and features. If the user is looking for comparisons, ask if they need a product comparison, and if so, compare and provide the best product."
    ),
]

# Define memory and prefix
prefix = """
You are an AI Shopping Assistant and the shop name is EKKA, designed to be able to assist the user in finding the right product in an online store. The store contains products
for various categories such as clothing, electronics, jewelry, and more.
You are given the following filtered products in a shop and a conversation. You should try better to understand the user's needs and suggest one or more products. 
Provide a conversational answer based on the products provided. If you have more than one product to recommend, show them as a bulleted list.
If you can't find the answer in the context below, say politely "Hmm, I'm not sure." Don't try to make up a product which is not.
"""


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
callback_manager = CallbackManager([])
openai.api_key = openai_api_key
chat = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo-0125")

ask_ai = initialize_agent(
    tools=tools,
    llm=chat,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory,
    prefix=prefix,
    input_variables=["input", "agent_scratchpad"],
    callback_manager=callback_manager,
)


@app.route('/api/query', methods=['POST'])
def query():
    user_input = request.form.get('input')

    if user_input:
        print(user_input)
        response = ask_ai.run(f'{str(user_input)} and give me product url')
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No input provided'}), 400

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
