#!/usr/bin/env python3
"""
Test client for the Question Server MCP.
This script tests all functionality provided by question_server.py.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Create parameters for connecting to the server
    server_params = StdioServerParameters(
        command="python",
        args=["question_server.py"],
        env=None  # Use environment variables from the current process
    )
    
    # Connect to the server using the stdio transport
    async with stdio_client(server_params) as (read, write):
        # Create a client session
        async with ClientSession(read, write) as session:
            # Initialize the connection
            print("🚀 Initializing connection to Question Server...")
            await session.initialize()
            
            # List available resources
            print("\n🔍 Available Resources:")
            resources = await session.list_resources()
            # Check if resources is a list/tuple or has a different structure
            if hasattr(resources, 'resources'):
                for resource in resources.resources:
                    print(f"- {resource.uri}: {getattr(resource, 'description', 'No description')}")
            else:
                for resource in resources:
                    if isinstance(resource, tuple):
                        if len(resource) >= 3:
                            print(f"- {resource[0]}: {resource[2]}")
                        else:
                            print(f"- {resource[0]}")
                    elif hasattr(resource, 'uri'):
                        print(f"- {resource.uri}: {getattr(resource, 'description', 'No description')}")
                    else:
                        print(f"- {resource}")
            
            # List available tools
            print("\n🔧 Available Tools:")
            tools = await session.list_tools()
            # Adapt for possible different return formats
            if hasattr(tools, 'tools'):
                for tool in tools.tools:
                    print(f"- {tool.name}: {getattr(tool, 'description', 'No description')}")
            else:
                for tool in tools:
                    if isinstance(tool, tuple):
                        if len(tool) >= 2:
                            print(f"- {tool[0]}: {tool[1] if len(tool) > 1 else 'No description'}")
                        else:
                            print(f"- {tool[0]}")
                    elif hasattr(tool, 'name'):
                        print(f"- {tool.name}: {getattr(tool, 'description', 'No description')}")
                    else:
                        print(f"- {tool}")
            
            # List available prompts
            print("\n📝 Available Prompts:")
            try:
                prompts = await session.list_prompts()
                # Adapt for possible different return formats
                if hasattr(prompts, 'prompts'):
                    for prompt in prompts.prompts:
                        print(f"- {prompt.name}: {getattr(prompt, 'description', 'No description')}")
                else:
                    for prompt in prompts:
                        if isinstance(prompt, tuple):
                            if len(prompt) >= 2:
                                print(f"- {prompt[0]}: {prompt[1] if len(prompt) > 1 else 'No description'}")
                            else:
                                print(f"- {prompt[0]}")
                        elif hasattr(prompt, 'name'):
                            print(f"- {prompt.name}: {getattr(prompt, 'description', 'No description')}")
                        else:
                            print(f"- {prompt}")
            except Exception as e:
                print(f"Error listing prompts: {e}")
            
            # Test reading resources
            try:
                print("\n📊 Testing resource: questions://count")
                count_result = await session.read_resource("questions://count")
                print_result(count_result, "Question count")
                
                print("\n📚 Testing resource: questions://all")
                all_questions = await session.read_resource("questions://all")
                # For all questions, just show the first few lines to avoid overwhelming output
                print_result(all_questions, "All questions (first 3 lines)")
                if isinstance(all_questions, str):
                    lines = all_questions.split("\n")
                    for i in range(min(3, len(lines))):
                        print(f"  {lines[i]}")
                    print(f"  ... and {len(lines) - 3} more lines")
            except Exception as e:
                print(f"Error reading resource: {e}")
                import traceback
                traceback.print_exc()
            
            # Test calling tools
            try:
                print("\n🎲 Testing get_random_question tool:")
                random_result = await session.call_tool("get_random_question")
                print_result(random_result, "Random question")
                
                # Get the question count to test get_question_by_number with a valid number
                count_result = await session.read_resource("questions://count")
                if isinstance(count_result, str):
                    try:
                        question_count = int(count_result)
                        test_number = min(question_count, 5)  # Use question #5 or the max available
                        
                        print(f"\n🔢 Testing get_question_by_number tool (question #{test_number}):")
                        number_result = await session.call_tool("get_question_by_number", arguments={"number": test_number})
                        print_result(number_result, f"Question #{test_number}")
                        
                        # Test with an invalid number
                        invalid_number = question_count + 100
                        print(f"\n⚠️ Testing get_question_by_number with invalid number ({invalid_number}):")
                        invalid_result = await session.call_tool("get_question_by_number", arguments={"number": invalid_number})
                        print_result(invalid_result, "Invalid question number result")
                    except ValueError:
                        print(f"Could not parse question count: {count_result}")
            except Exception as e:
                print(f"Error calling tool: {e}")
                import traceback
                traceback.print_exc()
            
            # Test getting the interview prompt
            try:
                print("\n🎭 Testing ConductInterview prompt:")
                prompt_result = await session.get_prompt("ConductInterview")
                print("Interview prompt structure:")
                for i, message in enumerate(prompt_result.messages):
                    print(f"Message {i+1} - Role: {message.role}")
                    if hasattr(message.content, 'text'):
                        # Just show a snippet of the content to avoid overwhelming output
                        text = message.content.text
                        preview = text[:100] + "..." if len(text) > 100 else text
                        print(f"Content preview: {preview}")
            except Exception as e:
                print(f"Error getting prompt: {e}")

def print_result(result, label):
    """Helper function to print results with consistent formatting."""
    print(f"{label}:")
    
    # Extract the result data based on the structure
    if hasattr(result, 'result'):
        result_data = result.result
    elif hasattr(result, 'return_value'):
        result_data = result.return_value
    elif hasattr(result, 'resource_contents'):
        result_data = result.resource_contents
    else:
        result_data = result
    
    # Print the result based on its type
    if isinstance(result_data, dict):
        print(json.dumps(result_data, indent=2))
    elif isinstance(result_data, str):
        print(f"  {result_data}")
    else:
        print(f"  {result_data}")

if __name__ == "__main__":
    asyncio.run(main())