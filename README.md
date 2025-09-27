# Textbook PDF Analysis MCP Server

A Model Context Protocol (MCP) server that provides comprehensive PDF analysis capabilities including OCR, table of contents extraction, summary generation, flashcard creation, and quiz generation.

## Purpose

This MCP server provides a secure interface for AI assistants to analyze textbook PDFs with advanced text processing capabilities. It supports both regular PDF text extraction and OCR for scanned documents.

## Features

### Current Implementation

- **`extract_toc`** - Extract table of contents from PDF files with pattern recognition
- **`chapter_summary`** - Generate comprehensive summaries for chapter page ranges
- **`section_summary`** - Create individual page summaries within a section
- **`page_summary`** - Generate detailed summary for a specific page
- **`flashcards`** - Create study flashcards from PDF content
- **`quiz_gen`** - Generate fill-in-the-blank quiz questions from text

### Security & Configuration

- File type validation and sanitization
- Configurable page limits and file size restrictions
- Rate limiting (50 requests per hour per tool)
- Non-root container execution
- Environment-based configuration

## Prerequisites

- Docker Desktop with MCP Toolkit enabled
- Docker MCP CLI plugin (`docker mcp` command)
- Sufficient disk space for PDF processing
- Optional: CUDA support for faster AI processing

## Configuration Options

Set these environment variables when running:

- `MODEL_PATH`: AI model path (default: "microsoft/DialoGPT-medium")
- `MAX_PAGES`: Maximum pages to process (default: 500)
- `OCR_LANG`: OCR language code (default: "eng")
- `ALLOWED_UPLOAD_DIR`: Upload directory (default: "/app/uploads")
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 100)

## Installation

See the step-by-step instructions provided with the files.

## Usage Examples

In Claude Desktop, you can ask:

- "Extract the table of contents from textbook.pdf"
- "Summarize chapter 3 which spans pages 45-67"
- "Create flashcards from pages 10-20 of the biology textbook"
- "Generate quiz questions from the first 5 pages"
- "Summarize page 123 using OCR since it's a scanned document"

### Tool Parameters

**extract_toc**:
- `file_path`: Name of PDF file in upload directory
- `use_ocr`: "true" or "false" for OCR processing

**chapter_summary**:
- `file_path`: Name of PDF file
- `chapter_pages`: Page range (e.g., "10-25" or "10,15,20-25")
- `use_ocr`: Enable OCR if needed

**flashcards**:
- `file_path`: Name of PDF file
- `pages`: Page range to process
- `count`: Number of flashcards to generate (max 20)
- `use_ocr`: Enable OCR if needed

## Architecture

```
Claude Desktop → MCP Gateway → Textbook MCP Server → PDF Processing
                                      ↓
                              OCR (Tesseract) + NLP Models
                              (Text extraction, summarization, Q&A generation)
```

## Development

### Local Testing

```bash
# Set environment variables for testing
export ALLOWED_UPLOAD_DIR="/tmp/pdfs"
export MAX_PAGES="100"
export OCR_LANG="eng"

# Run directly
python textbook_server.py

# Test MCP protocol
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python textbook_server.py
```

### Adding New Tools

1. Add the function to `textbook_server.py`
2. Decorate with `@mcp.tool()`
3. Update the catalog entry with the new tool name
4. Rebuild the Docker image

## Supported Languages

OCR supports multiple languages via Tesseract:

- English (eng) - Default
- French (fra)
- Spanish (spa) 
- German (deu)
- Additional languages can be added by modifying the Dockerfile

## Troubleshooting

### Tools Not Appearing

- Verify Docker image built successfully
- Check catalog and registry files
- Ensure Claude Desktop config includes custom catalog
- Restart Claude Desktop

### PDF Processing Errors

- Ensure PDF file is in the upload directory
- Check file permissions and size limits
- For scanned PDFs, use `use_ocr: "true"`
- Verify sufficient memory for large files

### OCR Issues

- Check if Tesseract language packs are installed
- Verify image quality for scanned documents
- Consider preprocessing images for better OCR results

### Performance Issues

- Reduce page ranges for large documents
- Lower flashcard/quiz counts for faster processing
- Consider using GPU acceleration for AI models

## Security Considerations

- All files processed within container sandbox
- Input sanitization prevents path traversal
- Rate limiting prevents abuse
- No persistent storage of uploaded content
- Running as non-root user
- File type validation

## API Rate Limits

Each tool has a rate limit of 50 requests per hour to prevent abuse and ensure fair usage.

## File Format Support

- PDF files (native text and scanned images)
- Maximum file size: 100MB (configurable)
- Maximum pages: 500 (configurable)

## License

MIT License