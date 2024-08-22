from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class AnthropicLLMTool:
    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20240620"):
        self.model = ChatAnthropic(anthropic_api_key=api_key, model=model_name)
    
    def process_request(self, user_prompt: str, system_prompt: str, ) -> str:
        prompt = ChatPromptTemplate.from_messages(
           [
            (
                "system",
                system_prompt,
            ),
            ("human", "{input}"),
            ]
        )
        chain = prompt | self.model | StrOutputParser()
        return chain.invoke({"input":user_prompt})



llm = AnthropicLLMTool("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")
print(llm.process_request("hi", "you are assistant"))

