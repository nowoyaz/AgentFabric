from typing import List, Dict, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from .agent import Agent, LangChainAgent

class AgentFactory:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_complete_agent(self, 
                              agent_type: str, 
                              model_name: str, 
                              file_paths: List[str], 
                              source_columns: Dict[str, str], 
                              system_prompt: str, 
                              tool_data: Optional[Dict[str, str]] = None,
                              tool_system_prompts: Optional[Dict[str, str]] = None) -> Agent:
        if agent_type == "LangChainAgent":
            agent_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),  
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            agent = LangChainAgent(api_key=self.api_key, model_name=model_name)
            agent.load_data_and_create_faiss(file_paths, source_columns)
            agent.generate_tools(tool_data=tool_data, tool_system_prompts=tool_system_prompts)
            agent.create_agent(agent_prompt)
            return agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")