import os
from dotenv import load_dotenv
from AgentFabric import AgentFactory

def main():
    load_dotenv()
    
    factory = AgentFactory(api_key=os.getenv("OPENAI_API_KEY"))

    source_columns = {
        "motor_data": "model",
    }

    tool_data = {
        "motor_data": "Tool for retrieving motor data based on model",
    }

    tool_system_prompts = {
        "motor_data": "You are an expert in motor specifications. Provide accurate and detailed information about motors when asked."
    }

    file_paths = [
        "./data/motor_data.csv",
    ]

    system_prompt = "You are a very powerful assistant, but don't know current events"

    agent = factory.create_complete_agent(
        agent_type="LangChainAgent",
        model_name="gpt-4o",
        file_paths=file_paths,
        source_columns=source_columns,
        system_prompt=system_prompt,
        tool_data=tool_data,
        tool_system_prompts=tool_system_prompts
    )

    initial_chat_history = ["Меня зовут Олег", "Привет"]
    response, updated_chat_history = agent.run_agent("Как меня зовут?", initial_chat_history)
    print(f"Response: {response}")
    print(f"Updated chat history: {updated_chat_history}")

if __name__ == "__main__":
    main()