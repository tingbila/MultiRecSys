import asyncio

from fastmcp import Client
from fastmcp.client import SSETransport


async def test_client():
    """采用fastmcp库中的客户端，来调用工具"""
    async with Client(SSETransport(url='http://localhost:8008/sse')) as client:
        tools = await client.list_tools()
        print(tools)
        # resource = await client.list_resources()
        # print(resource)

        email = await client.read_resource('datas://users/567/email')
        print(email)

        # 调用一个工具
        result = await client.call_tool(name='add', arguments={"a": 23, "b": 11})
        print(result)

if __name__ == '__main__':
    asyncio.run(test_client())
