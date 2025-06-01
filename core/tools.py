"""
Tools for the BasicAgent using smolagents framework.
"""

import os
import re
import uuid
import tempfile
import logging
from typing import List, Dict, Union, Optional
from pathlib import Path
import requests
from huggingface_hub import InferenceClient

from smolagents import DuckDuckGoSearchTool, VisitWebpageTool, WikipediaSearchTool, Tool

logger = logging.getLogger(__name__)


class FileDownloadTool(Tool):
    """Tool for downloading files from URLs."""

    name = "file_download"
    description = """Downloads a file from a URL and saves it locally. Returns a dictionary with 'path' key containing the file path."""

    inputs = {
        "url": {"type": "string", "description": "The URL to download the file from"},
        "directory": {"type": "string", "description": "Directory to save the file (optional)", "nullable": True},
        "filename": {"type": "string", "description": "Custom filename (optional)", "nullable": True}
    }

    output_type = "object"

    def forward(
            self,
            url: str,
            directory: Optional[str] = None,
            filename: Optional[str] = None
    ) -> Dict[str, Union[str, int, None]]:
        """Download a file from URL."""
        try:
            if directory is None or directory == "":
                directory = tempfile.gettempdir()

            os.makedirs(directory, exist_ok=True)

            logger.info(f"Downloading file from: {url}")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            if filename is None:
                filename = self._get_filename_from_response(response, url)

            file_path = os.path.join(directory, filename)

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = os.path.getsize(file_path)
            content_type = response.headers.get('content-type', 'unknown')

            result = {
                "status": "success",
                "path": file_path,
                "filename": filename,
                "size_bytes": file_size,
                "content_type": content_type,
                "directory": directory
            }

            logger.info(f"File downloaded successfully: {filename} ({file_size} bytes)")
            return result

        except requests.exceptions.RequestException as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "path": None
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "path": None
            }

    def _get_filename_from_response(self, response, url: str) -> str:
        """Extract filename from response headers or URL."""
        cd = response.headers.get('content-disposition', '')
        if cd:
            match = re.search(r"filename\*=UTF-8''(.+)", cd) or re.search(r'filename="?([^"]+)"?', cd)
            if match:
                return match.group(1)

        filename = os.path.basename(url.split('?')[0])
        if filename and '.' in filename:
            return filename

        # Fallback: generate filename based on content type
        content_type = response.headers.get('content-type', '').lower()
        extension = self._get_extension_from_content_type(content_type)
        return f"downloaded_{uuid.uuid4().hex[:8]}{extension}"

    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Get file extension from content type."""
        extensions = {
            'text/plain': '.txt',
            'text/csv': '.csv',
            'application/json': '.json',
            'application/pdf': '.pdf',
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'video/mp4': '.mp4',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'application/vnd.ms-excel': '.xls',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
        }
        return extensions.get(content_type, '.bin')


class FileReaderTool(Tool):
    """Tool for reading various file types."""

    name = "file_reader"
    description = """Reads content from various file types including text, CSV, JSON, and basic analysis. Returns a dictionary with keys: 'status' (success/error), 'content' (file contents or analysis), 'file_path', 'file_size', 'file_extension', and 'encoding'. Use the 'content' value to access the file data."""

    inputs = {
        "file_path": {"type": "string", "description": "Path to the file to read"},
        "encoding": {"type": "string", "description": "Text encoding (defaults to 'utf-8')", "nullable": True},
        "max_lines": {"type": "integer", "description": "Maximum lines to read for large files", "nullable": True}
    }

    output_type = "object"

    def forward(
            self,
            file_path: str,
            encoding: str = 'utf-8',
            max_lines: Optional[int] = None
    ) -> Dict[str, Union[str, int, None]]:
        """Read file content."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return {
                    "status": "error",
                    "error": f"File not found: {file_path}",
                    "content": None
                }

            file_size = file_path.stat().st_size
            file_extension = file_path.suffix.lower()

            logger.info(f"Reading file: {file_path} ({file_size} bytes)")

            if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
                content = self._read_text_file(file_path, encoding, max_lines)
            elif file_extension == '.csv':
                content = self._read_csv_file(file_path, max_lines)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
                content = self._analyze_image_file(file_path)
            elif file_extension == '.pdf':
                content = self._read_pdf_file(file_path, max_lines)  # max_lines -> max_pages
            else: # Try to read as text with error handling
                content = self._read_text_file(file_path, encoding, max_lines, safe_mode=True)

            return {
                "status": "success",
                "content": content,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_extension": file_extension,
                "encoding": encoding
            }

        except Exception as e:
            error_msg = f"Error reading file: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "content": None
            }

    def _read_text_file(self, file_path: Path, encoding: str, max_lines: Optional[int], safe_mode: bool = False) -> str:
        """Read text file content."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            lines.append(f"... (truncated after {max_lines} lines)")
                            break
                        lines.append(line.rstrip())
                    return '\n'.join(lines)
                else:
                    return f.read()
        except UnicodeDecodeError:
            if safe_mode:
                # Try with different encodings or as binary
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                    return f"[Binary/Non-UTF8 file - showing with latin-1 encoding]\n{content[:1000]}..."
                except:
                    return f"[Binary file - cannot display as text. Size: {file_path.stat().st_size} bytes]"
            else:
                raise

    def _read_csv_file(self, file_path: Path, max_lines: Optional[int]) -> Dict[str, Union[str, int, list]]:
        """Read CSV file"""
        import pandas as pd

        dataframe = pd.read_csv(file_path, nrows=max_lines if max_lines else None)

        analysis = [
            f"CSV Analysis for {file_path.name}:",
            f"Dimensions: {len(dataframe)} rows × {len(dataframe.columns)} columns",
            f"Columns: {', '.join(dataframe.columns.tolist())}",
            "",
            "Data types:",
        ]

        for col, dtype in dataframe.dtypes.items():
            analysis.append(f"  {col}: {dtype}")

        result = {
            'analysis': '\n'.join(analysis),
            'rows': len(dataframe),
            'columns': len(dataframe.columns),
            'data': dataframe.to_dict('records')
        }
        logger.debug(f"CSV READ result: {result}")
        return result

    def _analyze_csv_with_llm(self, file_path: Path, dataframe, query: Optional[str] = None) -> str:
        """Analyze CSV data using LLM for insights"""
        try:
            # Get LLM client from environment or create a simple one
            llm_client = self._get_llm_client()
            if not llm_client:
                return "LLM analysis not available - no LLM client configured"

            # Prepare data summary for LLM
            df_summary = f"""CSV File: {file_path.name}
            Dimensions: {len(dataframe)} rows × {len(dataframe.columns)} columns
            Columns: {', '.join(dataframe.columns.tolist())}
        
            Data Types:
            {dataframe.dtypes.to_string()}
        
            Sample Data (first 5 rows):
            {dataframe.head().to_string()}
        
            Statistical Summary:
            {dataframe.describe().to_string()}
            """

            # Create analysis prompt
            if query:
                prompt = f"""Analyze this CSV data and answer the specific question: "{query}"
                {df_summary}
                Please provide:
                1. Direct answer to the question
                2. Supporting analysis and insights
                3. Any relevant patterns or trends
                """
            else:
                prompt = f"""Analyze this CSV data and provide comprehensive insights:
                {df_summary}
            
                Please provide:
                1. Summary of data structure and content
                2. Key patterns and insights
                3. Potential data quality issues
                4. Interesting findings or correlations
                5. Suggestions for further analysis
                Format your response clearly with sections and bullet points."""

            # Get LLM analysis
            result = llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )

            if "error" in result:
                return f"LLM analysis failed: {result['error']}"

            return result.get("content", "No analysis generated")

        except Exception as e:
            return f"Error in LLM analysis: {str(e)}"

    def _get_llm_client(self):
        """Get LLM client from global context or environment"""
        try:
            # Try to import and create a basic LLM client
            from core.llm_client import create_llm_client
            import os

            api_url = os.getenv("LLM_API_URL", "http://host.docker.internal:11434/api/generate")
            default_model = os.getenv("DEFAULT_MODEL", "mistral")

            return create_llm_client(
                api_url=api_url,
                default_model=default_model,
                telemetry_enabled=False
            )
        except:
            return None

    def _analyze_image_file(self, file_path: Path) -> str:
        """Basic image file analysis for image size and dimensions."""
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                return f"""Image Analysis for {file_path.name}:
                    Format: {img.format}
                    Mode: {img.mode}
                    Size: {img.width} × {img.height} pixels
                    File size: {file_path.stat().st_size} bytes
                    
                    [Note: Image content cannot be analyzed without vision model integration]
                    """
        except ImportError:
            return f"Image file detected: {file_path.name} (PIL not available for analysis)"
        except Exception as e:
            return f"Image file: {file_path.name} (Error analyzing: {str(e)})"


def _read_pdf_file(self, file_path: Path, max_pages: Optional[int] = None) -> str:
    """Read PDF file content with text extraction."""
    try:
        import PyPDF2

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            # Limit pages if specified
            pages_to_read = min(total_pages, max_pages) if max_pages else total_pages

            text_content = []
            text_content.append(f"PDF Analysis for {file_path.name}:")
            text_content.append(f"Total pages: {total_pages}")
            text_content.append(f"Reading pages: 1 to {pages_to_read}")
            text_content.append("-" * 50)

            for page_num in range(pages_to_read):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()

                    if page_text.strip():
                        text_content.append(f"\n--- Page {page_num + 1} ---")
                        text_content.append(page_text.strip())
                    else:
                        text_content.append(f"\n--- Page {page_num + 1} ---")
                        text_content.append("[No extractable text found on this page]")

                except Exception as e:
                    text_content.append(f"\n--- Page {page_num + 1} ---")
                    text_content.append(f"[Error reading page: {str(e)}]")

            if pages_to_read < total_pages:
                text_content.append(f"\n... (Remaining {total_pages - pages_to_read} pages not shown)")

            return '\n'.join(text_content)

    except ImportError:
        return f"PDF file detected: {file_path.name} (PyPDF2 not available for text extraction)"
    except Exception as e:
        return f"PDF file: {file_path.name} (Error reading: {str(e)})"


class SpeechToTextTool(Tool):
    """Speech-to-Text tool using API """

    name = "speech_to_text"
    description = """Transcribes downloaded audio files to text. Input should be a file path to an audio file."""

    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to downloaded audio file (wav, mp3, m4a, flac, ogg)"
        },
        "language": {
            "type": "string",
            "description": "Language code for transcription (optional, e.g., 'en', 'es', 'fr')",
            "nullable": True
        }
    }

    output_type = "string"

    def __init__(self, hf_token: str, model: Optional[str] = None):
        """Initialize with HF token and model."""
        super().__init__()
        self.hf_token = hf_token
        self.model = os.getenv("STT_MODEL", "openai/whisper-large-v2")
        self.client = InferenceClient(token=hf_token)

    def forward(self, file_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio file to text."""
        try:
            logger.info(f"Transcribing audio file: {file_path}")
            path = Path(file_path)

            if not path.exists():
                return f"Error: Audio file not found at {file_path}"

            result = self.client.automatic_speech_recognition(
                audio=str(path),
                model=self.model
            )

            transcription = result.text.strip()
            logger.debug(f"Result: {result.text[:100]}")
            logger.info(f"Transcription successful: {len(transcription)} characters")

            return transcription

        except Exception as e:
            error_msg = f"Speech-to-text processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

class ImageToTextTool(Tool):
    """Image-to-Text tool using API """

    name = "image_to_text"
    description = """Transcribes downloaded image files to text. Input should be a file path to an image file."""

    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to downloaded image file (wav, mp3, m4a, flac, ogg)"
        },
        "prompt": {
            "type": "string",
            "description": "Custom prompt for image analysis (optional)",
            "nullable": True
        }
    }

    output_type = "string"

    def __init__(self, open_ai_key: str, model: Optional[str] = None):
        """Initialize with HF token and model."""
        super().__init__()
        self.api_key = open_ai_key
        import openai
        self.client = openai.OpenAI(api_key=open_ai_key)

    def forward(self, file_path: str, prompt: str = "Describe this image in detail") -> str:
        try:
            import base64
            from pathlib import Path

            logger.info(f"Analyzing image: {file_path}")
            path = Path(file_path)

            supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
            if path.suffix.lower() not in supported_extensions:
                return f"Error: Unsupported image format. Supported: {', '.join(supported_extensions)}"

            file_size = path.stat().st_size
            max_size = 20 * 1024 * 1024  # 20MB
            if file_size > max_size:
                return f"Error: Image file too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is 20MB."

            with open(file_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(path.suffix.lower(), 'image/jpeg')
            logger.info(f"Sending {len(base64_image)} character base64 image to OpenAI")

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=500
            )
            description = response.choices[0].message.content.strip()
            logger.info(f"Image analysis successful: {len(description)} characters")
            return description

        except ImportError as e:
            error_msg = f"OpenAI library not available: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}. Please install: pip install openai"

        except Exception as e:
            error_msg = f"Image analysis failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

def get_default_tools() -> List[Tool]:
    """
    Get the default set of tools for the BasicAgent.

    Returns:
        List of configured tool instances
    """
    hf_token = os.getenv("HF_TOKEN")
    openai_key = os.getenv("OPEN_AI_KEY")

    tools = [
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        WikipediaSearchTool(content_type="summary"),
        FileDownloadTool(),
        FileReaderTool()
    ]
    #load HF tools
    if hf_token:
        tools.append(SpeechToTextTool(hf_token))
        logger.info("Speech-to-Text tool integrated")
    else:
        logger.warning("No HF token found, SpeechToTextTool is not enabled")
    #load openai based tools
    if openai_key:
        tools.append(ImageToTextTool(openai_key))
        logger.info("Image-to-Text tool integrated")
    else:
        logger.warning("No OpenAI API key found, ImageToTextTool not enabled")

    return tools

# Tool registry for easy access
AVAILABLE_TOOLS = {
    'search': DuckDuckGoSearchTool,
    'visit': VisitWebpageTool,
    'speech_to_text': SpeechToTextTool,
    'search_wikipedia': WikipediaSearchTool,
    'download': FileDownloadTool,
    'read': FileReaderTool
}


def create_tool_by_name(tool_name: str) -> Optional[Tool]:
    """
    Create a tool instance by name.

    Args:
        tool_name: Name of the tool to create

    Returns:
        Tool instance or None if not found
    """
    tool_class = AVAILABLE_TOOLS.get(tool_name.lower())
    if tool_class:
        return tool_class()
    return None