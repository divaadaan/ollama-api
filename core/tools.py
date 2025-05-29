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
from requests import RequestException
from markdownify import markdownify

from smolagents import DuckDuckGoSearchTool, VisitWebpageTool, SpeechToTextTool, WikipediaSearchTool, Tool

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

            # Handle different file types
            if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
                content = self._read_text_file(file_path, encoding, max_lines)
            elif file_extension == '.csv':
                content = self._read_csv_file(file_path, max_lines)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
                content = self._analyze_image_file(file_path)
            else:
                # Try to read as text with error handling
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

    def _read_csv_file(self, file_path: Path, max_lines: Optional[int]) -> str:
        """Read CSV file with basic analysis."""
        import pandas as pd

        # Read CSV with basic analysis
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

        analysis.extend([
            "",
            "First few rows:",
            dataframe.head().to_string(),
            "",
            "Basic statistics for numeric columns:"
        ])

        numeric_cols = dataframe.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            analysis.append(dataframe[numeric_cols].describe().to_string())
        else:
            analysis.append("No numeric columns found.")

        return '\n'.join(analysis)


    def _analyze_image_file(self, file_path: Path) -> str:
        """Basic image file analysis."""
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


def get_default_tools() -> List[Tool]:
    """
    Get the default set of tools for the BasicAgent.

    Returns:
        List of configured tool instances
    """
    return [
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        SpeechToTextTool(),
        WikipediaSearchTool(),
        FileDownloadTool(),
        FileReaderTool()
    ]


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