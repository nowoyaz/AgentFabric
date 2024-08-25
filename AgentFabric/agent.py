import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from operator import itemgetter
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

class Agent(ABC):
    @abstractmethod
    def run_agent(self, question: str, chat_history: Optional[List[str]] = None) -> tuple[str, List[str]]:
        pass

class LangChainAgent(Agent):
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstores: Dict[str, FAISS] = {}
        self.tools: List[Tool] = []
        self.agent_executor: Optional[AgentExecutor] = None

    def load_data_and_create_faiss(self, file_paths: List[str], source_columns: Optional[Dict[str, str]] = None) -> None:
        source_columns = source_columns or {}

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()
            file_name = os.path.splitext(file_name)[0].lower()

            if file_extension == '.csv':
                source_column = source_columns.get(file_name.split('.')[0])
                if not source_column:
                    raise ValueError(f"Source column for {file_name} not provided.")
                loader = CSVLoader(file_path=file_path, source_column=source_column)
                data = loader.load()
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                data = loader.load_and_split()
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Only .csv and .pdf files are supported.")

            self.vectorstores[file_name] = FAISS.from_documents(data, self.embeddings)

    def generate_tools(self, 
                       context_key: str = "question", 
                       tool_data: Optional[Dict[str, str]] = None,
                       tool_system_prompts: Optional[Dict[str, str]] = None) -> None:
        tool_data = tool_data or {}
        tool_system_prompts = tool_system_prompts or {}
        
        for file_name, vectorstore in self.vectorstores.items():
            retriever = vectorstore.as_retriever()

            setup_and_retrieval = RunnablePassthrough.assign(
                context=itemgetter(context_key) | retriever
            )

            system_prompt = tool_system_prompts.get(file_name, "You are a helpful assistant.")
            dynamic_system_prompt_template = SystemMessagePromptTemplate(
                prompt=PromptTemplate(template=system_prompt)
            )

            review_human_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["question", "context"],
                    template="""You need to answer the [question] using data from the [context].
                    [context]
                    {context}

                    [question]
                    {question}""",
                )
            )
            messages = [dynamic_system_prompt_template, review_human_prompt]
            review_prompt_template = ChatPromptTemplate(
                input_variables=["context", "question"],
                messages=messages,
            )

            review_chain = setup_and_retrieval | review_prompt_template | self.llm | StrOutputParser()

            tool_name = file_name
            tool_description = tool_data.get(tool_name, "Default tool description")

            @tool()
            def generated_tool(question: str) -> str:
                "temp docstring"
                return review_chain.invoke({"question": question})

            generated_tool.name = tool_name
            generated_tool.description = tool_description

            self.tools.append(generated_tool)

    def create_agent(self, system_prompt: str) -> None:
        agent = create_tool_calling_agent(self.llm, self.tools, system_prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)

    def run_agent(self, question: str, chat_history: Optional[List[str]] = None) -> tuple[str, List[str]]:
        if not self.agent_executor:
            raise RuntimeError("Agent not created. Call create_agent() before using.")
        
        chat_history = chat_history or []
        
        langchain_chat_history = []
        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                langchain_chat_history.append(HumanMessage(content=message))
            else:
                langchain_chat_history.append(AIMessage(content=message))

        response = self.agent_executor.invoke(
            {
                "input": question,
                "chat_history": langchain_chat_history
            }
        )

        chat_history.append(question)
        chat_history.append(response["output"])
        
        return response["output"], chat_history