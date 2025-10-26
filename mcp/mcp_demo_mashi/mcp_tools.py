from mcp_demo.mcp_server import mcp_server

@mcp_server.tool()
def add(a: int, b: int) -> int:
    """加法运算: 计算两个数字相加"""
    return a + b


@mcp_server.tool()
def multiply(a: int, b: int) -> int:
    """乘法运算：计算两个数字相乘"""
    return a * b
