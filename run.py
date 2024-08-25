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
        "fiber":"use this tool to answer about fiber. If somebody asks 'оптоволокно' - используй этот инструмент"
    }

    tool_system_prompts = {
        "motor_data": "You are an expert in motor specifications. Provide accurate and detailed information about motors when asked.",
        "fiber": "you arre an expert in fiber for drones"
    }

    file_paths = [
        "./data/motor_data.csv",
        "./data/fiber.pdf"
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

    initial_chat_history = []
    response, updated_chat_history = agent.run_agent("Какие размеры оптоволокна у вас есть", initial_chat_history)
    print(f"Response: {response}")
    print(f"Updated chat history: {updated_chat_history}")

if __name__ == "__main__":
    main()