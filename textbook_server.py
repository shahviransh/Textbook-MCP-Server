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

# Configure Tesseract path for Docker
if os.path.exists('/usr/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

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
MAX_OCR_PAGES = int(os.environ.get("MAX_OCR_PAGES", "20"))  # Strict limit for OCR
OCR_LANG = os.environ.get("OCR_LANG", "eng")
ALLOWED_UPLOAD_DIR = os.environ.get("ALLOWED_UPLOAD_DIR", "/app/uploads")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "100"))
OCR_TIMEOUT_PER_PAGE = int(os.environ.get("OCR_TIMEOUT_PER_PAGE", "30"))  # seconds
OCR_DPI = int(os.environ.get("OCR_DPI", "200"))  # Lower DPI for faster processing

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

def page_has_images(file_path, page_num):
    """Detect if a page contains images."""
    try:
        with pdfplumber.open(file_path) as pdf:
            if page_num > len(pdf.pages):
                return False
            page = pdf.pages[page_num - 1]
            # Check if page has images
            if hasattr(page, 'images') and page.images:
                return True
            # Also check if extracted text is very sparse (might indicate scanned page)
            text = page.extract_text() or ""
            # If very little text is extracted, it might be an image-based page
            if len(text.strip()) < 50:
                return True
        return False
    except Exception as e:
        logger.warning(f"Could not detect images on page {page_num}: {e}")
        return False

async def extract_text_from_pdf(file_path, use_ocr=False, pages=None):
    """Extract text from PDF with optional OCR. 
    
    If use_ocr='auto', intelligently detects which pages have images and only OCRs those.
    If use_ocr=True, OCRs all specified pages (with MAX_OCR_PAGES limit).
    If use_ocr=False, uses regular text extraction only.
    """
    text_content = {}
    
    try:
        if use_ocr == 'auto' or use_ocr is True:
            # Smart OCR: detect pages with images and OCR only those
            loop = asyncio.get_event_loop()
            page_list = pages if pages else list(range(1, MAX_OCR_PAGES + 1))
            
            logger.info(f"Starting intelligent OCR: scanning {len(page_list)} pages for images...")
            
            # First pass: extract text normally and detect which pages need OCR
            pages_needing_ocr = []
            with pdfplumber.open(file_path) as pdf:
                for page_num in page_list:
                    if page_num > len(pdf.pages):
                        continue
                    
                    try:
                        page = pdf.pages[page_num - 1]
                        page_text = page.extract_text() or ""
                        
                        # Check if page has images or very little text (scanned page)
                        has_images = hasattr(page, 'images') and len(page.images) > 0
                        has_sparse_text = len(page_text.strip()) < 50
                        
                        if has_images or has_sparse_text:
                            pages_needing_ocr.append(page_num)
                            logger.info(f"Page {page_num}: Images detected or sparse text - will use OCR")
                        else:
                            # Good text extraction, use it
                            text_content[page_num] = page_text
                            logger.debug(f"Page {page_num}: Clean text extraction ({len(page_text)} chars)")
                    
                    except Exception as e:
                        logger.warning(f"Error analyzing page {page_num}: {e}")
                        pages_needing_ocr.append(page_num)
            
            # Second pass: OCR only pages that need it
            if pages_needing_ocr:
                # Enforce OCR page limit
                if len(pages_needing_ocr) > MAX_OCR_PAGES:
                    logger.warning(f"Found {len(pages_needing_ocr)} pages with images, limiting OCR to first {MAX_OCR_PAGES} pages")
                    pages_needing_ocr = pages_needing_ocr[:MAX_OCR_PAGES]
                
                logger.info(f"🔍 Applying OCR to {len(pages_needing_ocr)} pages with images...")
                
                for page_num in pages_needing_ocr:
                    try:
                        logger.info(f"Converting page {page_num} to image for OCR...")
                        
                        # Convert single page to image in executor
                        try:
                            images = await loop.run_in_executor(
                                None,
                                lambda p=page_num: convert_from_path(
                                    file_path,
                                    first_page=p,
                                    last_page=p,
                                    dpi=OCR_DPI
                                )
                            )
                        except Exception as e:
                            logger.info(f"Retrying with system poppler path")
                            images = await loop.run_in_executor(
                                None,
                                lambda p=page_num: convert_from_path(
                                    file_path,
                                    first_page=p,
                                    last_page=p,
                                    poppler_path='/usr/bin',
                                    dpi=OCR_DPI
                                )
                            )
                        
                        if not images:
                            logger.warning(f"No image generated for page {page_num}")
                            text_content[page_num] = "[Image conversion failed]"
                            continue
                        
                        # Yield after image conversion
                        await asyncio.sleep(0.05)
                        
                        # Perform OCR on the converted image
                        logger.info(f"Performing OCR on page {page_num}...")
                        try:
                            page_text = await asyncio.wait_for(
                                loop.run_in_executor(
                                    None,
                                    lambda img=images[0]: pytesseract.image_to_string(
                                        img,
                                        lang=OCR_LANG,
                                        config='--psm 1 --oem 3'
                                    )
                                ),
                                timeout=OCR_TIMEOUT_PER_PAGE
                            )
                            text_content[page_num] = page_text
                            logger.info(f"Page {page_num} OCR completed: {len(page_text)} characters")
                        except asyncio.TimeoutError:
                            logger.warning(f"OCR timeout for page {page_num} after {OCR_TIMEOUT_PER_PAGE}s")
                            text_content[page_num] = "[OCR timeout - page skipped]"
                        
                        # Yield control after each page
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num}: {e}")
                        text_content[page_num] = f"[OCR error: {str(e)}]"
            
            logger.info(f"Intelligent OCR complete: {len(text_content)} pages processed ({len(pages_needing_ocr)} with OCR)")
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
async def extract_toc(file_path: str = "", use_ocr: str = "auto") -> str:
    """Extract table of contents from a PDF file with intelligent OCR."""
    if not check_rate_limit("extract_toc"):
        return "Rate limit exceeded. Please try again later."
    
    logger.info(f"Extracting TOC from {file_path}")
    
    if not file_path.strip():
        return "Error: file_path is required"
    
    # Sanitize file path
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"Error: File not found: {file_path}"
    
    if not validate_file_type(safe_path):
        return "Error: File must be a PDF"
    
    try:
        # Parse OCR mode
        ocr_mode = use_ocr.lower().strip()
        use_ocr_mode = 'auto' if ocr_mode in ['auto', 'automatic', 'smart'] else (True if ocr_mode in ['true', '1', 'yes'] else False)
        
        # Extract text from first 20 pages to find TOC
        text_content = await extract_text_from_pdf(safe_path, use_ocr_bool, list(range(1, 21)))
        
        # Detect TOC patterns
        toc_entries = detect_toc_patterns(text_content)
        
        if not toc_entries:
            return "No clear table of contents pattern detected in the first 20 pages"
        
        # Format TOC output
        result = "**Table of Contents Extracted:**\n\n"
        for entry in toc_entries[:50]:  # Limit output
            result += f"**{entry['section']}** {entry['title']} - Page {entry['page']}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"TOC extraction error: {e}")
        return f"Error extracting TOC: {str(e)}"

@mcp.tool()
async def chapter_summary(file_path: str = "", chapter_pages: str = "", use_ocr: str = "auto") -> str:
    """Generate summary for specific chapter pages with intelligent OCR."""
    if not check_rate_limit("chapter_summary"):
        return "Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating chapter summary from {file_path}, pages: {chapter_pages}")
    
    if not file_path.strip():
        return "Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"Error: File not found: {file_path}"
    
    try:
        # Parse OCR mode
        ocr_mode = use_ocr.lower().strip()
        use_ocr_mode = 'auto' if ocr_mode in ['auto', 'automatic', 'smart'] else (True if ocr_mode in ['true', '1', 'yes'] else False)
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page range
        pages = validate_page_range(chapter_pages, total_pages)
        if not pages:
            return "Error: Invalid page range"
        
        # Extract text from specified pages
        text_content = await extract_text_from_pdf(safe_path, use_ocr_mode, pages)
        
        # Combine text from all pages
        combined_text = '\n'.join(text_content.values())
        
        if not combined_text.strip():
            return "Error: No text found in specified pages"
        
        # Generate summary
        summary = generate_summary(combined_text, max_length=300)
        
        result = f"**Chapter Summary** (Pages {min(pages)}-{max(pages)}):\n\n{summary}\n\n"
        result += f"**Pages processed:** {len(pages)}\n**Total characters:** {len(combined_text)}"
        
        return result
        
    except Exception as e:
        logger.error(f"Chapter summary error: {e}")
        return f"Error generating chapter summary: {str(e)}"

@mcp.tool()
async def section_summary(file_path: str = "", section_pages: str = "", use_ocr: str = "auto") -> str:
    """Generate summary for specific section pages with intelligent OCR."""
    if not check_rate_limit("section_summary"):
        return "Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating section summary from {file_path}, pages: {section_pages}")
    
    if not file_path.strip():
        return "Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"Error: File not found: {file_path}"
    
    try:
        # Parse OCR mode
        ocr_mode = use_ocr.lower().strip()
        use_ocr_mode = 'auto' if ocr_mode in ['auto', 'automatic', 'smart'] else (True if ocr_mode in ['true', '1', 'yes'] else False)
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page range
        pages = validate_page_range(section_pages, total_pages)
        if not pages:
            return "Error: Invalid page range"
        
        # Extract text from specified pages
        text_content = await extract_text_from_pdf(safe_path, use_ocr_mode, pages)
        
        # Generate summary for each page
        summaries = []
        for page_num, text in text_content.items():
            if text.strip():
                page_summary = generate_summary(text, max_length=150)
                summaries.append(f"**Page {page_num}:** {page_summary}")
        
        if not summaries:
            return "Error: No text found in specified pages"
        
        result = f"**Section Summary** (Pages {min(pages)}-{max(pages)}):\n\n"
        result += '\n\n'.join(summaries)
        
        return result
        
    except Exception as e:
        logger.error(f"Section summary error: {e}")
        return f"Error generating section summary: {str(e)}"

@mcp.tool()
async def page_summary(file_path: str = "", page_number: str = "1", use_ocr: str = "auto") -> str:
    """Generate summary for a specific page with intelligent OCR."""
    if not check_rate_limit("page_summary"):
        return "Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating page summary from {file_path}, page: {page_number}")
    
    if not file_path.strip():
        return "Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"Error: File not found: {file_path}"
    
    try:
        page_num = int(page_number) if page_number.strip() else 1
        # Parse OCR mode
        ocr_mode = use_ocr.lower().strip()
        use_ocr_mode = 'auto' if ocr_mode in ['auto', 'automatic', 'smart'] else (True if ocr_mode in ['true', '1', 'yes'] else False)
        
        # Extract text from specific page
        text_content = await extract_text_from_pdf(safe_path, use_ocr_mode, [page_num])
        
        if page_num not in text_content:
            return f"Error: Page {page_num} not found in document"
        
        page_text = text_content[page_num]
        if not page_text.strip():
            return f"Page {page_num} contains no extractable text"
        
        # Generate summary
        summary = generate_summary(page_text, max_length=200)
        
        result = f"**Page {page_num} Summary:**\n\n{summary}\n\n"
        result += f"**Character count:** {len(page_text)}\n**Word count:** {len(page_text.split())}"
        
        return result
        
    except ValueError:
        return f"Error: Invalid page number: {page_number}"
    except Exception as e:
        logger.error(f"Page summary error: {e}")
        return f"Error generating page summary: {str(e)}"

@mcp.tool()
async def flashcards(file_path: str = "", pages: str = "", count: str = "5", use_ocr: str = "auto") -> str:
    """Generate flashcards from PDF content with intelligent OCR."""
    if not check_rate_limit("flashcards"):
        return "Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating flashcards from {file_path}, pages: {pages}")
    
    if not file_path.strip():
        return "Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"Error: File not found: {file_path}"
    
    try:
        card_count = int(count) if count.strip() else 5
        card_count = min(card_count, 20)  # Limit to 20 cards
        # Parse OCR mode
        ocr_mode = use_ocr.lower().strip()
        use_ocr_mode = 'auto' if ocr_mode in ['auto', 'automatic', 'smart'] else (True if ocr_mode in ['true', '1', 'yes'] else False)
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page range
        page_list = validate_page_range(pages, total_pages) if pages.strip() else list(range(1, min(11, total_pages + 1)))
        
        # Extract text from specified pages
        text_content = await extract_text_from_pdf(safe_path, use_ocr_mode, page_list)
        
        # Combine text from all pages
        combined_text = '\n'.join(text_content.values())
        
        if not combined_text.strip():
            return "Error: No text found in specified pages"
        
        # Generate flashcards
        flashcard_list = generate_flashcards(combined_text, card_count)
        
        if not flashcard_list:
            return "Error: Could not generate flashcards from the content"
        
        result = f"🎓 **Generated {len(flashcard_list)} Flashcards:**\n\n"
        
        for i, card in enumerate(flashcard_list, 1):
            result += f"**Card {i}:**\n"
            result += f"**Question:** {card['question']}\n"
            result += f"**Answer:** {card['answer']}\n\n"
        
        return result
        
    except ValueError:
        return f"Error: Invalid count value: {count}"
    except Exception as e:
        logger.error(f"Flashcard generation error: {e}")
        return f"Error generating flashcards: {str(e)}"

@mcp.tool()
async def quiz_gen(file_path: str = "", pages: str = "", count: str = "3", use_ocr: str = "auto") -> str:
    """Generate quiz questions from PDF content with intelligent OCR."""
    if not check_rate_limit("quiz_gen"):
        return "Rate limit exceeded. Please try again later."
    
    logger.info(f"Generating quiz from {file_path}, pages: {pages}")
    
    if not file_path.strip():
        return "Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"Error: File not found: {file_path}"
    
    try:
        question_count = int(count) if count.strip() else 3
        question_count = min(question_count, 10)  # Limit to 10 questions
        # Parse OCR mode
        ocr_mode = use_ocr.lower().strip()
        use_ocr_mode = 'auto' if ocr_mode in ['auto', 'automatic', 'smart'] else (True if ocr_mode in ['true', '1', 'yes'] else False)
        
        # Get total pages
        with open(safe_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Validate page range
        page_list = validate_page_range(pages, total_pages) if pages.strip() else list(range(1, min(11, total_pages + 1)))
        
        # Extract text from specified pages
        text_content = await extract_text_from_pdf(safe_path, use_ocr_mode, page_list)
        
        # Combine text from all pages
        combined_text = '\n'.join(text_content.values())
        
        if not combined_text.strip():
            return "Error: No text found in specified pages"
        
        # Generate quiz questions
        quiz_questions = generate_quiz(combined_text, question_count)
        
        if not quiz_questions:
            return "Error: Could not generate quiz questions from the content"
        
        result = f"**Generated {len(quiz_questions)} Quiz Questions:**\n\n"
        
        for i, question in enumerate(quiz_questions, 1):
            result += f"**Question {i}:**\n"
            result += f"{question['question']}\n"
            result += f"**Answer:** {question['answer']}\n"
            result += f"**Context:** {question['context'][:100]}...\n\n"
        
        return result
        
    except ValueError:
        return f"Error: Invalid count value: {count}"
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        return f"Error generating quiz: {str(e)}"

@mcp.tool()
async def extract_from_pdf(
    file_path: str = "",
    focus_pages: str = "",
    supporting_files: str = "",
    use_ocr: str = "auto"
) -> str:
    """
    Read entire PDF (or specified pages) with intelligent OCR
    
    Args:
        file_path: Path to the PDF file
        focus_pages: Optional page range to focus on (e.g., "1-10" or "5,10,15-20"). If empty, reads entire document
        supporting_files: Comma-separated list of supporting file paths
        use_ocr: OCR mode - "auto" (intelligent detection, default), "true" (force all pages), "false" (disable)
    
    Returns:
        Combined text content from main and supporting files
    """
    if not check_rate_limit("extract_text_from_pdf"):
        return "Rate limit exceeded. Please try again later."

    logger.info(f"Extracting text from PDF: {file_path}")

    if not file_path.strip():
        return "Error: file_path is required"
    
    safe_path = os.path.join(ALLOWED_UPLOAD_DIR, sanitize_filename(file_path))
    
    if not os.path.exists(safe_path):
        return f"Error: File not found: {file_path}"
    
    if not validate_file_type(safe_path):
        return "Error: File must be a PDF"
    
    try:
        # Parse OCR mode: 'auto' (default), 'true', or 'false'
        ocr_mode = use_ocr.lower().strip()
        if ocr_mode in ['auto', 'automatic', 'smart']:
            use_ocr_mode = 'auto'
        elif ocr_mode in ['true', '1', 'yes']:
            use_ocr_mode = True
        else:
            use_ocr_mode = False
        
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
            return "Error: Invalid page range"
        
        # Extract text from main PDF
        logger.info(f"Extracting text from main PDF (OCR mode: {use_ocr_mode})...")
        text_content = await extract_text_from_pdf(safe_path, use_ocr_mode, page_list)
        
        # Combine all text from main file
        combined_text = '\n\n'.join([f"=== PAGE {page_num} ===\n{text}" 
                                     for page_num, text in text_content.items() if text.strip()])
        
        if not combined_text.strip():
            return "Error: No text found in main document"
        
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
                    supporting_text += f"\n\n=== SUPPORTING FILE {idx}: {support_file} ===\nFile not found\n"
                    continue
                
                if not validate_file_type(safe_support_path):
                    logger.warning(f"Supporting file is not a PDF: {support_file}")
                    supporting_text += f"\n\n=== SUPPORTING FILE {idx}: {support_file} ===\nFile must be a PDF\n"
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
                        supporting_text += f"\n\n=== SUPPORTING FILE {idx}: {support_file} ===\nNo text found in document\n"
                        logger.warning(f"No text found in supporting file: {support_file}")
                
                except Exception as e:
                    logger.error(f"Error processing supporting file {support_file}: {e}")
                    supporting_text += f"\n\n=== SUPPORTING FILE {idx}: {support_file} ===\nError: {str(e)}\n"
        
        # Combine main text and supporting text
        final_output = f"=== MAIN FILE: {file_path} ===\n{combined_text}"
        if supporting_text:
            final_output += f"\n\n{supporting_text}"
        
        return final_output
        
    except Exception as e:
        logger.error(f"PDF text extraction error: {e}", exc_info=True)
        return f"Error extracting text from PDF: {str(e)}"

# === SERVER STARTUP ===

if __name__ == "__main__":
    logger.info("Starting Textbook PDF Analysis MCP server...")
    
    # Create upload directory if it doesn't exist
    os.makedirs(ALLOWED_UPLOAD_DIR, exist_ok=True)
    
    # Startup checks
    logger.info(f"Configuration: MAX_PAGES={MAX_PAGES}, OCR_LANG={OCR_LANG}")
    logger.info(f"Upload directory: {ALLOWED_UPLOAD_DIR}")
    
    # Verify OCR dependencies
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR detected: version {tesseract_version}")
    except Exception as e:
        logger.warning(f"Tesseract OCR not available: {e}")
        logger.warning("OCR features will not work properly")
    
    # Verify poppler
    try:
        import subprocess
        result = subprocess.run(['pdfinfo', '-v'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("Poppler utilities detected (required for pdf2image)")
        else:
            logger.warning("Poppler utilities may not be properly configured")
    except Exception as e:
        logger.warning(f"Could not verify poppler installation: {e}")
    
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
