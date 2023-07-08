# https://betterprogramming.pub/private-llms-on-local-and-in-the-cloud-with-langchain-gpt4all-and-cerebrium-6dade79f45f6
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """
You are a friendly chatbot assistant that responds in a conversational
manner to users questions. Keep the answers short, unless specifically
asked by the user to elaborate on something.

Question: {question}

Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm = GPT4All(
    model='./models/orca-mini-13b.ggmlv3.q4_0.bin',
    callbacks=[StreamingStdOutCallbackHandler()]
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

query = input("Prompt: ")
llm_chain(query)