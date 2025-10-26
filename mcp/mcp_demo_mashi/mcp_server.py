from mcp.server.fastmcp import FastMCP
from zhipu_ai import zhipuai_client

mcp_server = FastMCP(name='lx-mcp', instructions='我自己的MCP服务', port=8008)


@mcp_server.tool('my_search_tool', description='专门搜索互联网中的内容')
def my_search(query: str) -> str:
    """搜索互联网上的内容"""
    try:
        response = zhipuai_client.web_search.web_search(
            search_engine="search-std",
            search_query=query
        )
        print(response)
        if response.search_result:
            return "\n\n".join([d.content for d in response.search_result])
    except Exception as e:
        print(e)
        return '没有搜索到任何内容！'


@mcp_server.resource("datas://users/{user_id}/email", name='get_user_email')
async def get_user_email(user_id: str) -> str:
    """检索给定用户ID的电子邮件地址。"""
    emails = {"123": "alice@example.com", "456": "bob@example.com"}
    return emails.get(user_id, "not_found@example.com")


@mcp_server.resource("data://product-categories")
async def get_categories() -> list[str]:
    """返回一个类型的列表."""
    return ["Electronics", "Books", "Home Goods"]
