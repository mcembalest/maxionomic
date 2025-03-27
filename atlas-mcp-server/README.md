# Nomic Atlas MCP Server

This is a [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) server that provides tools for interacting with [Nomic Atlas](https://atlas.nomic.ai), allowing LLMs to search and maintain datasets.

## Installation

Install the MCP command-line tool:
```bash
pip install mcp
```

Install the Nomic Atlas MCP server:
```bash
mcp install atlas_mcp_server.py --env-var NOMIC_API_KEY=nk-...
```

## Available Tools

The server provides the following tools:

- `list_datasets`: List all datasets accessible to your account
- `upload_dataset`: Upload a new dataset to Atlas
- `vector_search`: Perform vector search on a dataset
- `add_data_to_dataset`: Add new data to an existing dataset
- `create_index`: Create a new index/map for a dataset
- `query_with_selections`: Query a dataset with selection filters
- `vector_search_with_selections`: Perform vector search with selection filters

