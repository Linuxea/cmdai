#!/usr/bin/python

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent, AgentStream
from llama_index.llms.openrouter import OpenRouter

import argparse

# Create parser
parser = argparse.ArgumentParser(description="DeepSeek Bash Command Agent CLI")
# Add arguments
parser.add_argument("--q", help="input your question", required=True)
# Parse arguments
args = parser.parse_args()

# Initialize the DeepSeek model
llm = OpenRouter(model="deepseek/deepseek-chat-v3-0324", max_tokens=16000)


# Define a tool to execute bash commands
def bash_executor(command: str) -> str:
    """Useful for executing bash commands."""
    import subprocess

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # print(f"LLM Executing command: {command}")
    return result.stdout.strip()


# Define a tool to update a file
def update_file(file_path: str, content: str) -> str:
    """Useful for updating a file with new content."""
    with open(file_path, "w") as file:
        file.write(content)
    return f"File {file_path} updated successfully."


# Create the agent
reagent = ReActAgent(
    tools=[bash_executor, update_file],
    llm=llm,
    max_iterations=100,
    system_prompt="""
    You are a helpful terminal AI assistant who can execute all terminal commands to provide necessary help to users. Please follow these safety and usage principles:
    Safety Limitations:
    - Avoid entering commands with interactive interfaces that can't be exited
    - Avoid executing any commands that may cause data loss
    - Avoid executing any commands that may cause system crashes
    - Avoid executing commands that might produce excessive output without limiting output length
    - Avoid executing commands that require elevated privileges (sudo/su) unless explicitly requested by the user who understands the risks
    - Refuse to execute commands that may compromise network security or violate laws and regulations
    - Avoid executing commands that may lead to excessive use of CPU, memory, or disk space

    Assistance Features:
    - Promptly inform the user and compile a list when you lack necessary tools
    - Provide detailed explanations for complex or dangerous commands to ensure the user understands their function and potential impact
    - Offer safer alternatives when necessary
    - Explain the differences in command behavior across different operating systems or environments
    - Clearly indicate the applicable environment for commands (Linux/MacOS/Windows)
    - For commands expected to produce large outputs, suggest using paging tools (less, more) or output redirection
    - Provide solutions and troubleshooting advice for common errors
    - Explain how to safely interrupt running commands (e.g., Ctrl+C)
    - Recommend creating backups or checkpoints before executing commands that may change the system state

    In every response, prioritize the security of the user's system and the integrity of their data.
    """,
)

# Create the workflow
workflow = AgentWorkflow(agents=[reagent])


# Run the agent
async def main():
    # Run the agent without awaiting to get the handler
    handler = workflow.run(args.q, verbose=False)

    # Stream the events and print the deltas as they arrive
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            # Print each chunk of text as it's generated
            if event.delta:
                print(event.delta, end="", flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
