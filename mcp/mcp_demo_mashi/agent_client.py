import asyncio

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient

from zhipu_ai import llm

# 2. 定义远程 MCP 服务器
mcp_server_config = {
    "url": "http://localhost:8008/sse",
    "transport": "sse"
}

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个智能助手，尽可能的调用工具回答用户的问题'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),
])


async def client_call():
    """客户端，去访问服务区中的工具和资源"""

    # 创建MCP的客户端连接
    async with MultiServerMCPClient({'lx_mcp': mcp_server_config}) as client:
        tools = client.get_tools()
        print(tools)
        resource = await client.get_resources('lx_mcp', uris='datas://users/567/email')
        print(resource[0].model_dump().get('data'))

        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        # resp = await executor.ainvoke({'input': '今天，长沙的天气情况'})
        resp = await executor.ainvoke({'input': '请计算:10和89的乘积'})
        print(resp)


if __name__ == '__main__':
    asyncio.run(client_call())