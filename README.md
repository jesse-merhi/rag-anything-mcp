# RAG Anything MCP Server

An MCP (Model Context Protocol) server that provides comprehensive RAG (Retrieval-Augmented Generation) capabilities for processing and querying directories of documents using the `raganything` library with full multimodal support.

## Features

- **End-to-End Document Processing**: Complete document parsing with multimodal content extraction
- **Multimodal RAG**: Support for images, tables, equations, and text processing
- **Batch Processing**: Process entire directories with multiple file types
- **Advanced Querying**: Both pure text and multimodal-enhanced queries
- **Multiple Query Modes**: hybrid, local, global, naive, mix, and bypass modes
- **Vision Processing**: Advanced image analysis using GPT-4V
- **Persistent Storage**: RAG instances maintained per directory for efficient querying

## Available Tools

### `process_directory`
Process all files in a directory for comprehensive RAG indexing with multimodal support.

**Required Parameters:**
- `directory_path`: Path to the directory containing files to process
- `api_key`: OpenAI API key for LLM and embedding functions

**Optional Parameters:**
- `working_dir`: Custom working directory for RAG storage
- `base_url`: OpenAI API base URL (for custom endpoints)
- `file_extensions`: List of file extensions to process (default: ['.pdf', '.docx', '.pptx', '.txt', '.md'])
- `recursive`: Process subdirectories (default: True)
- `enable_image_processing`: Enable image analysis (default: True)
- `enable_table_processing`: Enable table extraction (default: True)
- `enable_equation_processing`: Enable equation processing (default: True)
- `max_workers`: Concurrent processing workers (default: 4)

### `process_single_document`
Process a single document with full multimodal analysis.

**Required Parameters:**
- `file_path`: Path to the document to process
- `api_key`: OpenAI API key

**Optional Parameters:**
- `working_dir`: Custom working directory for RAG storage
- `base_url`: OpenAI API base URL
- `output_dir`: Output directory for parsed content
- `parse_method`: Document parsing method (default: "auto")
- `enable_image_processing`: Enable image analysis (default: True)
- `enable_table_processing`: Enable table extraction (default: True)
- `enable_equation_processing`: Enable equation processing (default: True)

### `query_directory`
Pure text query against processed documents using LightRAG.

**Parameters:**
- `directory_path`: Path to the processed directory
- `query`: The question to ask about the documents
- `mode`: Query mode - "hybrid", "local", "global", "naive", "mix", or "bypass" (default: "hybrid")

### `query_with_multimodal_content`
Enhanced query with additional multimodal content (tables, equations, etc.).

**Parameters:**
- `directory_path`: Path to the processed directory
- `query`: The question to ask
- `multimodal_content`: List of multimodal content dictionaries
- `mode`: Query mode (default: "hybrid")

**Example multimodal_content:**
```json
[
  {
    "type": "table",
    "table_data": "Method,Accuracy\\nRAGAnything,95.2%\\nBaseline,87.3%",
    "table_caption": "Performance comparison"
  },
  {
    "type": "equation",
    "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
    "equation_caption": "Document relevance probability"
  }
]
```

### `list_processed_directories`
List all directories that have been processed and are available for querying.

### `get_rag_info`
Get detailed information about the RAG configuration and status for a directory.

## Usage Examples

### 1. Basic Directory Processing
```
process_directory(
  directory_path="/path/to/documents",
  api_key="your-openai-api-key"
)
```

### 2. Advanced Directory Processing
```
process_directory(
  directory_path="/path/to/research_papers",
  api_key="your-openai-api-key",
  file_extensions=[".pdf", ".docx"],
  enable_image_processing=true,
  enable_table_processing=true,
  max_workers=6
)
```

### 3. Pure Text Query
```
query_directory(
  directory_path="/path/to/documents",
  query="What are the main findings in these research papers?",
  mode="hybrid"
)
```

### 4. Multimodal Query with Table Data
```
query_with_multimodal_content(
  directory_path="/path/to/documents",
  query="Compare these results with the document findings",
  multimodal_content=[{
    "type": "table",
    "table_data": "Method,Accuracy,Speed\\nRAGAnything,95.2%,120ms\\nBaseline,87.3%,180ms",
    "table_caption": "Performance comparison"
  }],
  mode="hybrid"
)
```

### 5. Single Document Processing
```
process_single_document(
  file_path="/path/to/important_paper.pdf",
  api_key="your-openai-api-key",
  enable_image_processing=true
)
```

## Setup Requirements

### 1. Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Run the MCP Server
```bash
python main.py
```

## Query Modes Explained

- **hybrid**: Combines local and global search (recommended for most use cases)
- **local**: Focuses on local context and entity relationships
- **global**: Provides broader, document-level insights and summaries
- **naive**: Simple keyword-based search without graph reasoning
- **mix**: Combines multiple approaches for comprehensive results
- **bypass**: Direct access without RAG processing

## Multimodal Content Types

The server supports processing and querying with:

- **Images**: Automatic caption generation and visual analysis
- **Tables**: Structure extraction and content analysis
- **Equations**: LaTeX parsing and mathematical reasoning
- **Charts/Graphs**: Visual data interpretation
- **Mixed Content**: Combined analysis of multiple content types

## API Configuration

The server uses OpenAI's APIs by default:
- **LLM**: GPT-4o-mini for text processing
- **Vision**: GPT-4o for image analysis
- **Embeddings**: text-embedding-3-large (3072 dimensions)

You can customize the `base_url` parameter to use:
- Azure OpenAI
- OpenAI-compatible APIs
- Custom model endpoints

## File Support

Supported file formats include:
- PDF documents
- Microsoft Word (.docx)
- PowerPoint presentations (.pptx)
- Text files (.txt)
- Markdown files (.md)
- And more via the raganything library

## Performance Notes

- **Concurrent Processing**: Use `max_workers` to control parallel document processing
- **Memory Usage**: Large documents with many images may require significant memory
- **API Costs**: Vision processing (GPT-4o) is more expensive than text processing
- **Storage**: Processed data is stored locally for efficient re-querying