import ast
import asyncio
import json
import pprint
import sys
from typing import Dict, Any, List, Optional
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

pp = pprint.PrettyPrinter(indent=2, width=100)

def unwrap_tool_result(resp):
    """Unwrap tool result from MCP response"""
    if hasattr(resp, "structured_content") and resp.structured_content:
        return resp.structured_content.get("result", resp.structured_content)
    if hasattr(resp, "content") and resp.content:
        text = resp.content
        try:
            return ast.literal_eval(text)
        except Exception:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return resp


class MCPClient:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.client = None
        self.tools = []
        self.is_connected = False

    async def start(self):
        """Initialize MCP client connection"""
        try:
            transport = StdioTransport(
                command=sys.executable,
                args=[self.server_script],
            )
            self.client = Client(transport)
            print(f"🚀 Connecting to MCP server: {self.server_script}")
            await self.client.__aenter__()
            
            # Discover available tools
            await self.discover_tools()
            self.is_connected = True
            tool_names = [t.name for t in self.tools]
            print(f"✅ Connected! Available tools: {tool_names}")
            
        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {e}")
            raise

    async def stop(self):
        """Close MCP client connection"""
        if self.client and self.is_connected:
            await self.client.__aexit__(None, None, None)
            self.is_connected = False
            print("🔌 Disconnected from MCP server")

    async def discover_tools(self):
        """Discover available tools from the MCP server"""
        if not self.client:
            raise RuntimeError("Client not connected. Call start() first.")
        
        print("🛠️  Discovering tools...")
        self.tools = await self.client.list_tools()
        return self.tools

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool with error handling and result unwrapping"""
        if not self.client or not self.is_connected:
            raise RuntimeError("Client not connected. Call start() first.")
        
        try:
            print(f"🔧 Calling tool: {tool_name} with params: {params}")
            result = await self.client.call_tool(tool_name, params)
            unwrapped = unwrap_tool_result(result)
            print(f"✅ Tool {tool_name} completed successfully")
            return {"success": True, "result": unwrapped}
            
        except Exception as e:
            print(f"❌ Tool {tool_name} failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool"""
        for tool in self.tools:
            if tool.name == tool_name:
                return getattr(tool, "inputSchema", {})
        return None

    def list_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]

    async def interactive_prompt(self):
        """Interactive mode for testing tools"""
        print("\n--- MCP Client Interactive ---")
        print("Commands: list, call <number>, schema <number>, exit")
        
        while True:
            try:
                cmd = input("Enter command> ").strip()
                
                if cmd == "exit":
                    print("Exiting client...")
                    await self.stop()
                    break
                    
                elif cmd == "list":
                    for i, t in enumerate(self.tools):
                        desc = getattr(t, "description", None) or getattr(t, "title", "")
                        print(f"{i+1}. {t.name} - {desc}")
                        
                elif cmd.startswith("schema "):
                    try:
                        idx = int(cmd.split()[1]) - 1
                        if idx < 0 or idx >= len(self.tools):
                            print("Invalid tool number")
                            continue
                        tool = self.tools[idx]
                        schema = getattr(tool, "inputSchema", {})
                        print(f"Schema for {tool.name}:")
                        pp.pprint(schema)
                    except Exception as e:
                        print(f"Error: {e}")
                        
                elif cmd.startswith("call "):
                    await self._handle_tool_call(cmd)
                    
                else:
                    print("Unknown command. Available: list, call <number>, schema <number>, exit")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    async def _handle_tool_call(self, cmd: str):
        """Handle interactive tool calling"""
        try:
            idx = int(cmd.split()[1]) - 1
            if idx < 0 or idx >= len(self.tools):
                print("Invalid tool number")
                return
                
            tool = self.tools[idx]
            schema = getattr(tool, "inputSchema", {})
            
            if schema and "properties" in schema:
                params = {}
                for pname, pinfo in schema["properties"].items():
                    required = pname in schema.get("required", [])
                    ptype = pinfo.get("type", "string")
                    default = pinfo.get("default", None)
                    description = pinfo.get("description", "")
                    
                    if not required and default is not None:
                        prompt = f"Enter '{pname}' ({ptype}, default={default}): {description}\n> "
                    else:
                        prompt = f"Enter '{pname}' ({ptype}{'*' if required else ''}): {description}\n> "
                        
                    val = input(prompt).strip()
                    
                    if val == "" and not required and default is not None:
                        val = default
                    elif val == "" and required:
                        print(f"'{pname}' is required, try again.")
                        return
                        
                    # Type conversion
                    if ptype == "integer" and val:
                        try:
                            val = int(val)
                        except ValueError:
                            print(f"Invalid integer for '{pname}', try again.")
                            return
                    elif ptype == "number" and val:
                        try:
                            val = float(val)
                        except ValueError:
                            print(f"Invalid number for '{pname}', try again.")
                            return
                    elif ptype == "boolean" and val:
                        val = str(val).lower() in ("true", "yes", "1", "on")
                    elif ptype == "array" and val:
                        try:
                            val = json.loads(val) if val.startswith('[') else val.split(',')
                        except json.JSONDecodeError:
                            val = val.split(',')
                            
                    params[pname] = val
                    
                print(f"\n🔧 Calling {tool.name} with params: {params}")
                result = await self.call_tool(tool.name, params)
                print("\n📋 Result:")
                pp.pprint(result)
                
            else:
                print("Tool has no input schema - calling with no parameters.")
                result = await self.call_tool(tool.name, {})
                print("\n📋 Result:")
                pp.pprint(result)
                
        except Exception as e:
            print(f"Error calling tool: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


async def main():
    """Main function for standalone testing"""
    if len(sys.argv) < 2:
        print("Usage: python client.py mcp_server.py ")
        print("Example: python client.py mcp_server.py")
        sys.exit(1)
        
    server_script_path = sys.argv[1]
    
    async with MCPClient(server_script_path) as client:
        await client.interactive_prompt()


if __name__ == "__main__":
    asyncio.run(main())