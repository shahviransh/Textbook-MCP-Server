#!/usr/bin/env python3
"""
Textbook PDF Analysis MCP Server - Analyze PDFs with OCR, extract TOC, generate summaries, flashcards and quizzes
"""
import os
import sys
import logging
import json
import tempfile
import shutil
import re
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import aiofiles
import magic
import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import pipeline
import nltk
from mcp.server.fastmcp import FastMCP
import textwrap
from typing import List, Dict

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("textbook-server")

# Initialize MCP server
mcp = FastMCP("textbook")

# Configuration from environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "microsoft/DialoGPT-medium")
MAX_PAGES = int(os.environ.get("MAX_PAGES", "500"))
OCR_LANG = os.environ.get("OCR_LANG", "eng")
ALLOWED_UPLOAD_DIR = os.environ.get("ALLOWED_UPLOAD_DIR", "/app/uploads")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "100"))

# Rate limiting
REQUEST_COUNT = {}
RATE_LIMIT_REQUESTS = 50
RATE_LIMIT_WINDOW = 3600  # 1 hour

# Initialize NLP components
summarizer = None
nltk_downloaded = False

def get_summarizer():
    global summarizer
    if summarizer is None:
        from transformers import pipeline
        logger.info("Loading summarization model on first request...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def lazy_nltk_download():
    global nltk_downloaded
    if not nltk_downloaded:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk_downloaded = True

# === UTILITY FUNCTIONS ===

def sanitize_filename(filename):
    """Sanitize filename to prevent path traversal."""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    return filename

def validate_file_type(file_path):
    """Validate file is a PDF."""
    try:
        mime_type = magic.from_file(file_path, mime=True)
        return mime_type == 'application/pdf'
    except Exception:
        return file_path.lower().endswith('.pdf')

def validate_page_range(page_range, total_pages):
    """Validate and parse page range string."""
    if not page_range.strip():
        return list(range(1, min(total_pages + 1, MAX_PAGES + 1)))
    
    pages = []
    try:
        for part in page_range.split(','):
            if '-' in part:
                start, end = map(int, part.split('-', 1))
                pages.extend(range(max(1, start), min(total_pages + 1, end + 1)))
            else:
                page = int(part.strip())
                if 1 <= page <= total_pages:
                    pages.append(page)
        return sorted(list(set(pages)))[:MAX_PAGES]
    except ValueError:
        raise ValueError("Invalid page range format")

def check_rate_limit(client_id):
    """Simple rate limiting check."""
    now = datetime.now().timestamp()
    if client_id not in REQUEST_COUNT:
        REQUEST_COUNT[client_id] = []
    
    # Clean old requests
    REQUEST_COUNT[client_id] = [
        req_time for req_time in REQUEST_COUNT[client_id] 
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    if len(REQUEST_COUNT[client_id]) >= RATE_LIMIT_REQUESTS:
        return False
    
    REQUEST_COUNT[client_id].append(now)
    return True

async def extract_text_from_pdf(file_path, use_ocr=False, pages=None):
    """Extract text from PDF with optional OCR."""
    text_content = {}
    
    try:
        if use_ocr:
            # Convert PDF to images and OCR
            images = convert_from_path(file_path)
            for i, image in enumerate(images, 1):
                if pages and i not in pages:
                    continue
                try:
                    page_text = pytesseract.image_to_string(image, lang=OCR_LANG)
                    text_content[i] = page_text
                except Exception as e:
                    logger.warning(f"OCR failed for page {i}: {e}")
                    text_content[i] = ""
        else:
            # Extract text directly from PDF
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    if pages and i not in pages:
                        continue
                    try:
                        page_text = page.extract_text() or ""
                        text_content[i] = page_text
                    except Exception as e:
                        logger.warning(f"Text extraction failed for page {i}: {e}")
                        text_content[i] = ""
        
        return text_content
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise

def detect_toc_patterns(text_content):
    """Detect table of contents patterns in text."""
    toc_entries = []
    
    # Common TOC patterns
    patterns = [
        r'^\s*(\d+\.?\d*\.?\d*)\s+(.+?)\s+(\d+)\s*$',  # 1.2.3 Title 123
        r'^\s*([A-Z][a-z]+\s+\d+)\s+(.+?)\s+(\d+)\s*$',  # Chapter 1 Title 123
        r'^\s*([IVX]+\.?)\s+(.+?)\s+(\d+)\s*$',  # Roman numerals
    ]
    
    for page_num, text in text_content.items():
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    toc_entries.append({
                        'section': match.group(1),
                        'title': match.group(2).strip(),
                        'page': int(match.group(3)),
                        'found_on_page': page_num
                    })
    
    return toc_entries

def generate_summary(text, max_length=150):
    """Generate summary using transformer model."""
    if not text.strip():
        return "Summary not available"

    lazy_nltk_download()  # ensure NLTK is ready

    try:
        text = text.strip()[:1024]
        if len(text) < 50:
            return text[:max_length] + "..." if len(text) > max_length else text

        summ = get_summarizer()  # lazy load model here
        if summ:
            summary = summ(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        else:
            # Fallback extractive summarization
            sentences = text.split('. ')
            summary_sentences = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > max_length:
                    break
                summary_sentences.append(sentence.strip())
                current_length += len(sentence)
            
            summary = '. '.join(summary_sentences)
            if summary and not summary.endswith('.'):
                summary += '.'
                
            return summary if summary else text[:max_length] + "..."
            
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return text[:max_length] + "..." if len(text) > max_length else text

def generate_flashcards(text, count=5):
    """Generate flashcards from text content."""
    if not text.strip():
        return []
    
    # Simple flashcard generation based on sentences
    sentences = text.split('.')
    flashcards = []
    
    for i, sentence in enumerate(sentences[:count]):
        sentence = sentence.strip()
        if len(sentence) > 20:
            # Create question by removing key terms
            words = sentence.split()
            if len(words) > 5:
                # Remove middle portion as answer
                mid = len(words) // 2
                question = ' '.join(words[:mid-1]) + " _____ " + ' '.join(words[mid+1:])
                answer = words[mid-1:mid+1]
                
                flashcards.append({
                    'question': question,
                    'answer': ' '.join(answer),
                    'full_context': sentence
                })
    
    return flashcards

def generate_quiz(text, count=3):
    """Generate quiz questions from text content."""
    if not text.strip():
        return []
    
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30][:count]
    quiz_questions = []
    
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        if len(words) > 8:
            # Create fill-in-the-blank question
            key_word_idx = len(words) // 2
            key_word = words[key_word_idx]
            
            question_text = ' '.join(words[:key_word_idx]) + " _____ " + ' '.join(words[key_word_idx+1:])
            
            quiz_questions.append({
                'question': question_text,
                'answer': key_word,
                'type': 'fill_in_blank',
                'context': sentence
            })
    
    return quiz_questions

# === MCP TOOLS ===

@mcp.tool()
async def extract_toc(file_path: str = "", use_ocr: str = "false") -> str:
    """Extract table of contents from a PDF file."""
    if not check_rate_limit("extract_toc"):
        return "⏱️ Rate limit exceeded. Please try again later."
    
    logger.info(f"Extracting TOC from {file_path}")
    
    if not file_path.strip():
        return "❌ Error: file_path is required"
    
    # Sanitize file path
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"❌ Error: File not found: {file_path}"
    
    if not validate_file_type(safe_path):
        return "❌ Error: File must be a PDF"
    
    try:
        use_ocr_bool = use_ocr.lower() in ['true', '1', 'yes']
        
        # Extract text from first 20 pages to find TOC
        text_content = await extract_text_from_pdf(safe_path, use_ocr_bool, list(range(1, 21)))
        
        # Detect TOC patterns
        toc_entries = detect_toc_patterns(text_content)
        
        if not toc_entries:
            return "📋 No clear table of contents pattern detected in the first 20 pages"
        
        # Format TOC output
        result = "📚 **Table of Contents Extracted:**\n\n"
        for entry in toc_entries[:50]:  # Limit output
            result += f"**{entry['section']}** {entry['title']} - Page {entry['page']}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"TOC extraction error: {e}")
        return f"❌ Error extracting TOC: {str(e)}"

@mcp.tool()
async def chapter_summary(file_path: str = "", chapter_pages: str = "", use_ocr: str = "false") -> str:
    """Generate summary for specific chapter pages."""
    if not check_rate_limit("chapter_summary"):
        return "⏱️ Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating chapter summary from {file_path}, pages: {chapter_pages}")
    
    if not file_path.strip():
        return "❌ Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"❌ Error: File not found: {file_path}"
    
    try:
        use_ocr_bool = use_ocr.lower() in ['true', '1', 'yes']
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page range
        pages = validate_page_range(chapter_pages, total_pages)
        if not pages:
            return "❌ Error: Invalid page range"
        
        # Extract text from specified pages
        text_content = await extract_text_from_pdf(safe_path, use_ocr_bool, pages)
        
        # Combine text from all pages
        combined_text = '\n'.join(text_content.values())
        
        if not combined_text.strip():
            return "❌ Error: No text found in specified pages"
        
        # Generate summary
        summary = generate_summary(combined_text, max_length=300)
        
        result = f"📖 **Chapter Summary** (Pages {min(pages)}-{max(pages)}):\n\n{summary}\n\n"
        result += f"**Pages processed:** {len(pages)}\n**Total characters:** {len(combined_text)}"
        
        return result
        
    except Exception as e:
        logger.error(f"Chapter summary error: {e}")
        return f"❌ Error generating chapter summary: {str(e)}"

@mcp.tool()
async def section_summary(file_path: str = "", section_pages: str = "", use_ocr: str = "false") -> str:
    """Generate summary for specific section pages."""
    if not check_rate_limit("section_summary"):
        return "⏱️ Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating section summary from {file_path}, pages: {section_pages}")
    
    if not file_path.strip():
        return "❌ Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"❌ Error: File not found: {file_path}"
    
    try:
        use_ocr_bool = use_ocr.lower() in ['true', '1', 'yes']
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page range
        pages = validate_page_range(section_pages, total_pages)
        if not pages:
            return "❌ Error: Invalid page range"
        
        # Extract text from specified pages
        text_content = await extract_text_from_pdf(safe_path, use_ocr_bool, pages)
        
        # Generate summary for each page
        summaries = []
        for page_num, text in text_content.items():
            if text.strip():
                page_summary = generate_summary(text, max_length=150)
                summaries.append(f"**Page {page_num}:** {page_summary}")
        
        if not summaries:
            return "❌ Error: No text found in specified pages"
        
        result = f"📄 **Section Summary** (Pages {min(pages)}-{max(pages)}):\n\n"
        result += '\n\n'.join(summaries)
        
        return result
        
    except Exception as e:
        logger.error(f"Section summary error: {e}")
        return f"❌ Error generating section summary: {str(e)}"

@mcp.tool()
async def page_summary(file_path: str = "", page_number: str = "1", use_ocr: str = "false") -> str:
    """Generate summary for a specific page."""
    if not check_rate_limit("page_summary"):
        return "⏱️ Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating page summary from {file_path}, page: {page_number}")
    
    if not file_path.strip():
        return "❌ Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"❌ Error: File not found: {file_path}"
    
    try:
        page_num = int(page_number) if page_number.strip() else 1
        use_ocr_bool = use_ocr.lower() in ['true', '1', 'yes']
        
        # Extract text from specific page
        text_content = await extract_text_from_pdf(safe_path, use_ocr_bool, [page_num])
        
        if page_num not in text_content:
            return f"❌ Error: Page {page_num} not found in document"
        
        page_text = text_content[page_num]
        if not page_text.strip():
            return f"📄 Page {page_num} contains no extractable text"
        
        # Generate summary
        summary = generate_summary(page_text, max_length=200)
        
        result = f"📄 **Page {page_num} Summary:**\n\n{summary}\n\n"
        result += f"**Character count:** {len(page_text)}\n**Word count:** {len(page_text.split())}"
        
        return result
        
    except ValueError:
        return f"❌ Error: Invalid page number: {page_number}"
    except Exception as e:
        logger.error(f"Page summary error: {e}")
        return f"❌ Error generating page summary: {str(e)}"

@mcp.tool()
async def flashcards(file_path: str = "", pages: str = "", count: str = "5", use_ocr: str = "false") -> str:
    """Generate flashcards from PDF content."""
    if not check_rate_limit("flashcards"):
        return "⏱️ Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating flashcards from {file_path}, pages: {pages}")
    
    if not file_path.strip():
        return "❌ Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"❌ Error: File not found: {file_path}"
    
    try:
        card_count = int(count) if count.strip() else 5
        card_count = min(card_count, 20)  # Limit to 20 cards
        use_ocr_bool = use_ocr.lower() in ['true', '1', 'yes']
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page range
        page_list = validate_page_range(pages, total_pages) if pages.strip() else list(range(1, min(11, total_pages + 1)))
        
        # Extract text from specified pages
        text_content = await extract_text_from_pdf(safe_path, use_ocr_bool, page_list)
        
        # Combine text from all pages
        combined_text = '\n'.join(text_content.values())
        
        if not combined_text.strip():
            return "❌ Error: No text found in specified pages"
        
        # Generate flashcards
        flashcard_list = generate_flashcards(combined_text, card_count)
        
        if not flashcard_list:
            return "❌ Error: Could not generate flashcards from the content"
        
        result = f"🎓 **Generated {len(flashcard_list)} Flashcards:**\n\n"
        
        for i, card in enumerate(flashcard_list, 1):
            result += f"**Card {i}:**\n"
            result += f"❓ **Question:** {card['question']}\n"
            result += f"✅ **Answer:** {card['answer']}\n\n"
        
        return result
        
    except ValueError:
        return f"❌ Error: Invalid count value: {count}"
    except Exception as e:
        logger.error(f"Flashcard generation error: {e}")
        return f"❌ Error generating flashcards: {str(e)}"

@mcp.tool()
async def quiz_gen(file_path: str = "", pages: str = "", count: str = "3", use_ocr: str = "false") -> str:
    """Generate quiz questions from PDF content."""
    if not check_rate_limit("quiz_gen"):
        return "⏱️ Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating quiz from {file_path}, pages: {pages}")
    
    if not file_path.strip():
        return "❌ Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"❌ Error: File not found: {file_path}"
    
    try:
        question_count = int(count) if count.strip() else 3
        question_count = min(question_count, 10)  # Limit to 10 questions
        use_ocr_bool = use_ocr.lower() in ['true', '1', 'yes']
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page range
        page_list = validate_page_range(pages, total_pages) if pages.strip() else list(range(1, min(11, total_pages + 1)))
        
        # Extract text from specified pages
        text_content = await extract_text_from_pdf(safe_path, use_ocr_bool, page_list)
        
        # Combine text from all pages
        combined_text = '\n'.join(text_content.values())
        
        if not combined_text.strip():
            return "❌ Error: No text found in specified pages"
        
        # Generate quiz questions
        quiz_questions = generate_quiz(combined_text, question_count)
        
        if not quiz_questions:
            return "❌ Error: Could not generate quiz questions from the content"
        
        result = f"📝 **Generated {len(quiz_questions)} Quiz Questions:**\n\n"
        
        for i, question in enumerate(quiz_questions, 1):
            result += f"**Question {i}:**\n"
            result += f"❓ {question['question']}\n"
            result += f"✅ **Answer:** {question['answer']}\n"
            result += f"📖 **Context:** {question['context'][:100]}...\n\n"
        
        return result
        
    except ValueError:
        return f"❌ Error: Invalid count value: {count}"
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        return f"❌ Error generating quiz: {str(e)}"

@mcp.tool()
async def extract_from_pdf(
    file_path: str = "",
    focus_pages: str = "",
    supporting_files: str = "",
    use_ocr: str = "false"
) -> str:
    """
    Read entire PDF (or specified pages)
    
    Args:
        file_path: Path to the PDF file
        focus_pages: Optional page range to focus on (e.g., "1-10" or "5,10,15-20"). If empty, reads entire document
        supporting_files: Comma-separated list of supporting file paths
        use_ocr: Whether to use OCR for scanned PDFs
    
    Returns:
        Combined text content from main and supporting files
    """
    if not check_rate_limit("extract_text_from_pdf"):
        return "⏱️ Rate limit exceeded. Please try again later."

    logger.info(f"Extracting text from PDF: {file_path}")

    if not file_path.strip():
        return "❌ Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"❌ Error: File not found: {file_path}"
    
    if not validate_file_type(safe_path):
        return "❌ Error: File must be a PDF"
    
    try:
        use_ocr_bool = use_ocr.lower() in ['true', '1', 'yes']
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        logger.info(f"PDF has {total_pages} pages")
        
        # Determine which pages to read
        if focus_pages.strip():
            page_list = validate_page_range(focus_pages, total_pages)
            logger.info(f"Reading specified pages: {len(page_list)} pages")
        else:
            # Read entire document (up to MAX_PAGES limit)
            page_list = list(range(1, min(total_pages + 1, MAX_PAGES + 1)))
            logger.info(f"Reading entire document: {len(page_list)} pages")
        
        if not page_list:
            return "❌ Error: Invalid page range"
        
        # Extract text from main PDF
        logger.info("Extracting text from main PDF...")
        text_content = await extract_text_from_pdf(safe_path, use_ocr_bool, page_list)
        
        # Combine all text from main file
        combined_text = '\n\n'.join([f"=== PAGE {page_num} ===\n{text}" 
                                     for page_num, text in text_content.items() if text.strip()])
        
        if not combined_text.strip():
            return "❌ Error: No text found in main document"
        
        logger.info(f"Extracted {len(combined_text)} characters from {len(page_list)} pages")
        
        # Process supporting files
        supporting_text = ""
        supporting_file_list = [f.strip() for f in supporting_files.split(',') if f.strip()]
        
        if supporting_file_list:
            logger.info(f"Processing {len(supporting_file_list)} supporting files")
            
            for idx, support_file in enumerate(supporting_file_list, 1):
                safe_support_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(support_file))
                
                if not os.path.exists(safe_support_path):
                    logger.warning(f"Supporting file not found: {support_file}")
                    supporting_text += f"\n\n=== SUPPORTING FILE {idx}: {support_file} ===\n❌ File not found\n"
                    continue
                
                if not validate_file_type(safe_support_path):
                    logger.warning(f"Supporting file is not a PDF: {support_file}")
                    supporting_text += f"\n\n=== SUPPORTING FILE {idx}: {support_file} ===\n❌ File must be a PDF\n"
                    continue
                
                try:
                    # Get total pages of supporting file
                    with open(safe_support_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        support_total_pages = len(pdf_reader.pages)
                    
                    # Read all pages (up to MAX_PAGES limit)
                    support_page_list = list(range(1, min(support_total_pages + 1, MAX_PAGES + 1)))
                    
                    logger.info(f"Extracting text from supporting file: {support_file} ({len(support_page_list)} pages)")
                    support_content = await extract_text_from_pdf(safe_support_path, use_ocr_bool, support_page_list)
                    
                    # Combine text from supporting file
                    support_combined = '\n\n'.join([f"=== PAGE {page_num} ===\n{text}" 
                                                   for page_num, text in support_content.items() if text.strip()])
                    
                    if support_combined.strip():
                        supporting_text += f"\n\n=== SUPPORTING FILE {idx}: {support_file} ===\n{support_combined}"
                        logger.info(f"Extracted {len(support_combined)} characters from {support_file}")
                    else:
                        supporting_text += f"\n\n=== SUPPORTING FILE {idx}: {support_file} ===\n❌ No text found in document\n"
                        logger.warning(f"No text found in supporting file: {support_file}")
                
                except Exception as e:
                    logger.error(f"Error processing supporting file {support_file}: {e}")
                    supporting_text += f"\n\n=== SUPPORTING FILE {idx}: {support_file} ===\n❌ Error: {str(e)}\n"
        
        # Combine main text and supporting text
        final_output = f"=== MAIN FILE: {file_path} ===\n{combined_text}"
        if supporting_text:
            final_output += f"\n\n{supporting_text}"
        
        return final_output
        
    except Exception as e:
        logger.error(f"PDF text extraction error: {e}", exc_info=True)
        return f"❌ Error extracting text from PDF: {str(e)}"

# Add this utility function with the other utility functions
def extract_key_points(text: str, max_points: int = 15) -> List[str]:
    """Extract key points from text for cheatsheet."""
    if not text.strip():
        return []
    
    # Split into sentences
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    
    key_points = []
    for sentence in sentences[:max_points]:
        # Filter for meaningful sentences (avoid headers, page numbers, etc.)
        if len(sentence) > 20 and len(sentence.split()) > 3:
            # Truncate long sentences
            if len(sentence) > 100:
                sentence = sentence[:97] + "..."
            key_points.append(sentence)
    
    return key_points

def create_cheatsheet_html(title: str, sections: List[Dict[str, any]]) -> str:
    """Generate handwritten-looking cheatsheet HTML."""
    
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;600;700&family=Indie+Flower&family=Permanent+Marker&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Caveat', cursive;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .notebook {{
            max-width: 900px;
            margin: 0 auto;
            background: #fff;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            position: relative;
        }}
        
        /* Notebook binding holes */
        .notebook::before {{
            content: '';
            position: absolute;
            left: 40px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: repeating-linear-gradient(
                to bottom,
                transparent,
                transparent 40px,
                #e74c3c 40px,
                #e74c3c 42px
            );
        }}
        
        /* Top binding */
        .binding {{
            height: 30px;
            background: linear-gradient(to bottom, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
            border-bottom: 3px solid #1a252f;
            position: relative;
        }}
        
        .binding::after {{
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            background: repeating-linear-gradient(
                to right,
                transparent,
                transparent 45px,
                rgba(255,255,255,0.1) 45px,
                rgba(255,255,255,0.1) 50px
            );
        }}
        
        .page {{
            padding: 60px 80px 60px 80px;
            background: 
                linear-gradient(90deg, transparent 79px, #abced4 79px, #abced4 81px, transparent 81px),
                linear-gradient(#fff 0%, #fafafa 100%);
            background-size: 100% 100%;
            line-height: 35px;
            position: relative;
        }}
        
        /* Lined paper effect */
        .page::after {{
            content: '';
            position: absolute;
            left: 0;
            right: 0;
            top: 60px;
            bottom: 60px;
            background: repeating-linear-gradient(
                transparent,
                transparent 34px,
                #e8e8e8 34px,
                #e8e8e8 35px
            );
            pointer-events: none;
            z-index: 1;
        }}
        
        .content {{
            position: relative;
            z-index: 2;
        }}
        
        h1 {{
            font-family: 'Permanent Marker', cursive;
            font-size: 42px;
            color: #2c3e50;
            margin-bottom: 30px;
            text-align: center;
            transform: rotate(-1deg);
            text-shadow: 2px 2px 0px rgba(255,255,255,0.8);
        }}
        
        h2 {{
            font-family: 'Indie Flower', cursive;
            font-size: 32px;
            color: #e74c3c;
            margin-top: 35px;
            margin-bottom: 20px;
            border-bottom: 3px solid #e74c3c;
            padding-bottom: 5px;
            transform: rotate(-0.5deg);
            text-decoration: underline wavy #f39c12;
        }}
        
        h3 {{
            font-family: 'Caveat', cursive;
            font-size: 28px;
            color: #3498db;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: 700;
        }}
        
        ul {{
            list-style: none;
            padding-left: 30px;
        }}
        
        li {{
            font-size: 24px;
            color: #2c3e50;
            margin-bottom: 15px;
            position: relative;
            line-height: 32px;
        }}
        
        li::before {{
            content: '✓';
            position: absolute;
            left: -25px;
            color: #27ae60;
            font-weight: bold;
            font-size: 28px;
        }}
        
        .highlight {{
            background: linear-gradient(180deg, transparent 50%, #fff59d 50%);
            padding: 2px 4px;
            display: inline;
        }}
        
        .important {{
            background: linear-gradient(180deg, transparent 50%, #ffccbc 50%);
            padding: 2px 4px;
            display: inline;
            font-weight: 600;
        }}
        
        .formula {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            font-size: 20px;
            border-radius: 5px;
            transform: rotate(-0.5deg);
        }}
        
        .note {{
            background: #fffde7;
            border: 2px dashed #fbc02d;
            padding: 15px;
            margin: 20px 0;
            font-size: 22px;
            transform: rotate(0.5deg);
            box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
        }}
        
        .note::before {{
            content: '📌 Note: ';
            font-weight: bold;
            color: #f57c00;
        }}
        
        .doodle {{
            position: absolute;
            opacity: 0.3;
            z-index: 0;
        }}
        
        .star {{
            position: absolute;
            font-size: 30px;
            opacity: 0.4;
            z-index: 0;
        }}
        
        /* Handwritten emphasis */
        .underline {{
            text-decoration: underline wavy #e74c3c;
        }}
        
        .circle {{
            border: 2px solid #9c27b0;
            border-radius: 50%;
            padding: 5px 15px;
            display: inline-block;
            transform: rotate(-5deg);
        }}
        
        /* Footer */
        .footer {{
            padding: 20px 80px;
            text-align: center;
            font-size: 20px;
            color: #7f8c8d;
            border-top: 2px dashed #bdc3c7;
        }}
        
        /* Print styles */
        @media print {{
            body {{
                background: white;
            }}
            .notebook {{
                box-shadow: none;
            }}
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .page {{
                padding: 40px 40px 40px 60px;
            }}
            h1 {{
                font-size: 32px;
            }}
            h2 {{
                font-size: 26px;
            }}
            h3 {{
                font-size: 22px;
            }}
            li {{
                font-size: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="notebook">
        <div class="binding"></div>
        <div class="page">
            <div class="content">
                <h1>{title}</h1>
                {content}
            </div>
        </div>
        <div class="footer">
            📚 Created: {date} | Happy Studying! ✨
        </div>
    </div>
    
    <script>
        // Add random doodles
        function addDoodles() {{
            const page = document.querySelector('.page');
            const doodles = ['⭐', '🌟', '💡', '📝', '✨', '🎯'];
            
            for (let i = 0; i < 5; i++) {{
                const star = document.createElement('div');
                star.className = 'star';
                star.textContent = doodles[Math.floor(Math.random() * doodles.length)];
                star.style.left = Math.random() * 80 + 10 + '%';
                star.style.top = Math.random() * 80 + 10 + '%';
                page.appendChild(star);
            }}
        }}
        
        // Add slight rotation to list items for handwritten feel
        document.querySelectorAll('li').forEach((li, index) => {{
            const rotation = (Math.random() - 0.5) * 1.5;
            li.style.transform = `rotate(${{rotation}}deg)`;
        }});
        
        addDoodles();
    </script>
</body>
</html>'''
    
    return html

# Add this new MCP tool
@mcp.tool()
async def create_cheatsheet(
    file_path: str = "",
    pages: str = "",
    title: str = "",
    use_ocr: str = "false",
    style: str = "handwritten"
) -> str:
    """
    Create a handwritten-looking cheatsheet from PDF content.
    
    Args:
        file_path: Path to the PDF file
        pages: Page range to extract from (e.g., "1-10" or "5,10,15-20")
        title: Title for the cheatsheet (auto-generated if empty)
        use_ocr: Whether to use OCR for scanned PDFs
        style: Cheatsheet style - "handwritten" (default), "minimal", or "colorful"
    
    Returns:
        HTML content for handwritten cheatsheet
    """
    if not check_rate_limit("create_cheatsheet"):
        return "⏱️ Rate limit exceeded. Please try again later."
    
    logger.info(f"Creating cheatsheet from {file_path}, pages: {pages}")
    
    if not file_path.strip():
        return "❌ Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"❌ Error: File not found: {file_path}"
    
    if not validate_file_type(safe_path):
        return "❌ Error: File must be a PDF"
    
    try:
        use_ocr_bool = use_ocr.lower() in ['true', '1', 'yes']
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page range
        if pages.strip():
            page_list = validate_page_range(pages, total_pages)
        else:
            page_list = list(range(1, min(21, total_pages + 1)))  # First 20 pages by default
        
        if not page_list:
            return "❌ Error: Invalid page range"
        
        logger.info(f"Extracting text from {len(page_list)} pages for cheatsheet")
        
        # Extract text from specified pages
        text_content = await extract_text_from_pdf(safe_path, use_ocr_bool, page_list)
        
        # Combine text from all pages
        combined_text = '\n'.join(text_content.values())
        
        if not combined_text.strip():
            return "❌ Error: No text found in specified pages"
        
        # Auto-generate title if not provided
        if not title.strip():
            # Try to extract title from first few lines
            first_lines = combined_text.split('\n')[:5]
            title = next((line.strip() for line in first_lines if len(line.strip()) > 5), "Study Notes")
            if len(title) > 50:
                title = title[:47] + "..."
        
        logger.info(f"Generating cheatsheet with title: {title}")
        
        # Split content into logical sections
        sections = []
        current_section = {"title": "Overview", "points": []}
        
        # Simple section detection based on headers (lines with fewer than 60 chars, capitalized)
        lines = combined_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect potential headers
            if len(line) < 60 and (line.isupper() or line.istitle()) and not line.endswith('.'):
                # Save current section if it has points
                if current_section["points"]:
                    sections.append(current_section)
                # Start new section
                current_section = {"title": line, "points": []}
            else:
                # Add as point to current section
                if len(line) > 20 and len(current_section["points"]) < 10:  # Limit points per section
                    current_section["points"].append(line)
        
        # Add last section
        if current_section["points"]:
            sections.append(current_section)
        
        # If no sections detected, create one big section from key points
        if not sections:
            key_points = extract_key_points(combined_text, max_points=20)
            if key_points:
                sections = [{"title": "Key Points", "points": key_points}]
        
        if not sections:
            return "❌ Error: Could not extract meaningful content for cheatsheet"
        
        logger.info(f"Created {len(sections)} sections for cheatsheet")
        
        # Build HTML content
        content_html = ""
        for i, section in enumerate(sections[:8]):  # Limit to 8 sections for readability
            section_title = section["title"]
            points = section["points"][:8]  # Limit to 8 points per section
            
            # Vary the header level
            if i == 0:
                content_html += f'<h2>{section_title}</h2>\n'
            else:
                content_html += f'<h3>{section_title}</h3>\n'
            
            content_html += '<ul>\n'
            for j, point in enumerate(points):
                # Add variety with highlighting and emphasis
                if j % 3 == 0 and len(point) > 40:
                    # Highlight every 3rd point
                    content_html += f'<li><span class="highlight">{point}</span></li>\n'
                elif 'important' in point.lower() or 'key' in point.lower() or 'critical' in point.lower():
                    content_html += f'<li><span class="important">{point}</span></li>\n'
                else:
                    content_html += f'<li>{point}</li>\n'
            content_html += '</ul>\n'
            
            # Add a note box every 3 sections
            if (i + 1) % 3 == 0 and i < len(sections) - 1:
                content_html += '<div class="note">Review the above sections before moving forward!</div>\n'
        
        # Generate final HTML
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        final_html = create_cheatsheet_html(
            title=title,
            sections=sections
        ).format(
            title=title,
            content=content_html,
            date=current_date
        )
        
        logger.info("Cheatsheet HTML generated successfully")
        
        # Return as formatted response
        result = f"""📝 **Handwritten Cheatsheet Created!**

**Title:** {title}
**Pages processed:** {len(page_list)} pages
**Sections:** {len(sections)}
**Total points:** {sum(len(s['points']) for s in sections)}

The cheatsheet has been generated in handwritten style with:
- Notebook-style lined paper background
- Handwritten fonts (Caveat, Indie Flower, Permanent Marker)
- Color-coded sections and highlights
- Important points emphasized
- Study notes and doodles for visual appeal

**The HTML cheatsheet is ready to view!** It includes:
✓ Responsive design (mobile-friendly)
✓ Print-friendly CSS
✓ Interactive elements
✓ Handwritten aesthetic

You can save this as an HTML file and open it in any web browser, or print it out for physical study notes!

---
{final_html}"""
        
        return result
        
    except Exception as e:
        logger.error(f"Cheatsheet creation error: {e}", exc_info=True)
        return f"❌ Error creating cheatsheet: {str(e)}"

# === SERVER STARTUP ===

if __name__ == "__main__":
    logger.info("Starting Textbook PDF Analysis MCP server...")
    
    # Create upload directory if it doesn't exist
    os.makedirs(ALLOWED_UPLOAD_DIR, exist_ok=True)
    
    # Startup checks
    logger.info(f"Configuration: MAX_PAGES={MAX_PAGES}, OCR_LANG={OCR_LANG}")
    logger.info(f"Upload directory: {ALLOWED_UPLOAD_DIR}")
    
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)