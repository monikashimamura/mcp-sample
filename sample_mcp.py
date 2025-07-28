
import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import AsyncOpenAI

my_dic = {
    "key": "sk-NyiTFkWTnHGuF25INh29qiWM3KPONCoRV08o4Sht0kcCedUN",
    "url": "https://poloai.top/v1",
    "model": "gpt-3.5-turbo"
}

mcp_amap = "https://mcp.amap.com/sse?key=e33b4a10248322655ba354b65efa0169"


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI(
            api_key=my_dic["key"],
            base_url=my_dic["url"]
            )

    async def connect_to_sse_server(self, server_url: str):
        """使用SSE方法连接到服务器"""
        # 存储上下文信息
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # 初始化
        await self.session.initialize()

        print("Initialized SSE client...")
        print("Listing tools...")
        # 列出mcp server的元数据的可用的工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """清除上下文"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        """进行对话查询并处理工具调用"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        # mcp server的元数据的可用的工具
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # 询问LLM需要使用哪些工具
        completion = await self.openai.chat.completions.create(
            model=my_dic["model"],
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        tool_results = []
        final_text = []

        assistant_message = completion.choices[0].message

        # 工具调用
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                messages.extend([
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.content[0].text
                    }
                ])

                print(f"Tool {tool_name} returned: {result.content[0].text}")

                # 整理消息
                completion = await self.openai.chat.completions.create(
                    model=my_dic["model"],
                    max_tokens=1000,
                    messages=messages,
                )
                if isinstance(completion.choices[0].message.content, (dict, list)):
                    final_text.append(str(completion.choices[0].message.content))
                else:
                    final_text.append(completion.choices[0].message.content)
        else:
            if isinstance(assistant_message.content, (dict, list)):
                final_text.append(str(assistant_message.content))
            else:
                final_text.append(assistant_message.content)

        return "\n".join(final_text)


async def main(task):

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=mcp_amap)
        response = await client.process_query(task)
        print(response)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    _query = "请告诉我从武汉大学南大门到华中科技大学的路线。"
    asyncio.run(main(_query))
