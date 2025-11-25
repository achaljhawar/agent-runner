"""Image and video fetching and generation tools.

Implements:
- Image fetching from Pexels stock photo source
- AI image generation with Google Gemini
- Video fetching from Pexels stock video source
- AI video generation with Google Veo 3.1 Fast
"""

import asyncio
import base64
import os
from pathlib import Path

from agentrunner.core.exceptions import E_VALIDATION
from agentrunner.core.tool_protocol import ToolCall, ToolDefinition, ToolResult
from agentrunner.tools.base import BaseTool, ToolContext


class ImageFetchTool(BaseTool):
    """Fetch images from popular stock photo sources.

    Supports Pexels (free stock photos and videos).

    The tool will search for images matching the query and download the best match
    to the specified location in the workspace.

    Note: Requires httpx library and PEXELS_API_KEY environment variable.
    """

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Fetch image from a stock photo source.

        Args:
            call: Tool call with query, source, and save_path
            context: Tool execution context

        Returns:
            ToolResult with image path and metadata
        """
        query = call.arguments.get("query")
        source = call.arguments.get("source", "pexels").lower()
        save_path = call.arguments.get("save_path")
        orientation = call.arguments.get("orientation", "landscape")

        if not query:
            return ToolResult(
                success=False,
                error="query is required - describe the image you want to fetch",
                error_code=E_VALIDATION,
            )

        if not save_path:
            return ToolResult(
                success=False,
                error="save_path is required - specify where to save the image (e.g., 'public/hero.jpg')",
                error_code=E_VALIDATION,
            )

        # Validate source
        supported_sources = ["pexels"]  # Only Pexels is active
        if source not in supported_sources:
            return ToolResult(
                success=False,
                error=f"Unsupported source: {source}. Supported sources: {', '.join(supported_sources)}",
                error_code=E_VALIDATION,
            )

        # Check if httpx is available
        try:
            import httpx  # noqa: F401
        except ImportError:
            return ToolResult(
                success=False,
                error="ImageFetchTool requires httpx library. Install with: pip install httpx",
                error_code=E_VALIDATION,
            )

        # Resolve save path relative to workspace
        abs_save_path = Path(context.workspace.resolve_path(save_path))

        # Create parent directory if it doesn't exist
        abs_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Fetch image based on source
        if source == "pexels":
            return await self._fetch_from_pexels(query, abs_save_path, orientation, context)
        else:
            return ToolResult(
                success=False,
                error=f"Source '{source}' not yet implemented",
                error_code=E_VALIDATION,
            )

    async def _fetch_from_pexels(
        self, query: str, save_path: Path, orientation: str, context: ToolContext
    ) -> ToolResult:
        """Fetch image from Pexels.

        Uses Pexels' free API. Requires PEXELS_API_KEY environment variable.
        Get your key at: https://www.pexels.com/api/
        """
        import httpx

        api_key = os.getenv("PEXELS_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="Pexels requires PEXELS_API_KEY environment variable. Get your key at https://www.pexels.com/api/",
                error_code=E_VALIDATION,
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search for photos
                search_url = "https://api.pexels.com/v1/search"
                headers = {"Authorization": api_key}
                params = {
                    "query": query,
                    "per_page": 1,
                    "orientation": orientation,
                }

                response = await client.get(search_url, headers=headers, params=params)  # type: ignore[arg-type]
                response.raise_for_status()
                data = response.json()

                if not data.get("photos"):
                    return ToolResult(
                        success=False,
                        error=f"No images found for query: {query}",
                        error_code=E_VALIDATION,
                    )

                # Get the first result
                photo = data["photos"][0]
                image_url = photo["src"]["large"]
                photographer = photo["photographer"]
                photo_url = photo["url"]

                # Download the image
                image_response = await client.get(image_url)
                image_response.raise_for_status()

                # Save to file
                save_path.write_bytes(image_response.content)

                context.logger.info(
                    "Image fetched from Pexels",
                    query=query,
                    save_path=str(save_path),
                    photographer=photographer,
                )

                return ToolResult(
                    success=True,
                    output=f"Image fetched from Pexels and saved to: {save_path}\n\nQuery: {query}\nPhotographer: {photographer}\nSource: {photo_url}\n\nIMPORTANT: When using Pexels images, please credit the photographer:\nPhoto by {photographer} on Pexels ({photo_url})",
                    data={
                        "save_path": str(save_path),
                        "source": "pexels",
                        "query": query,
                        "photographer": photographer,
                        "photo_url": photo_url,
                        "width": photo["width"],
                        "height": photo["height"],
                    },
                )
        except httpx.HTTPStatusError as e:
            context.logger.error(
                "Pexels API error", status_code=e.response.status_code, error=str(e)
            )
            return ToolResult(
                success=False,
                error=f"Pexels API error: {e.response.status_code}",
                error_code=E_VALIDATION,
            )
        except httpx.TimeoutException:
            context.logger.error("Pexels API timeout", query=query)
            return ToolResult(
                success=False,
                error="Pexels API request timed out",
                error_code=E_VALIDATION,
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            ToolDefinition with image fetch schema
        """
        return ToolDefinition(
            name="fetch_image",
            description="""Fetches high-quality images from Pexels stock photo library.

Use this tool when you need to:
- Find and download images for websites, apps, or design projects
- Get high-quality stock photos matching specific descriptions
- Fetch images with specific orientations (landscape, portrait, square)

The tool will:
1. Search Pexels for images matching your query
2. Download the best matching image
3. Save it to the specified path in your workspace
4. Provide attribution information for the photographer

Supported source:
- pexels: Free stock photos and videos (requires PEXELS_API_KEY)

Usage notes:
- Be specific in your query for best results (e.g., "modern minimalist workspace" instead of just "workspace")
- The tool will automatically create parent directories if they don't exist
- Always credit photographers when required
- Images are downloaded in high quality suitable for web and print use""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the image you want (e.g., 'mountain sunset', 'business team meeting', 'modern apartment interior')",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["pexels"],
                        "default": "pexels",
                        "description": "Image source to fetch from. Only Pexels is supported.",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Path where to save the image in the workspace (e.g., 'public/images/hero.jpg', 'assets/background.png')",
                    },
                    "orientation": {
                        "type": "string",
                        "enum": ["landscape", "portrait", "squarish"],
                        "default": "landscape",
                        "description": "Image orientation preference. Default: landscape",
                    },
                },
                "required": ["query", "save_path"],
            },
        )


class ImageGenerationTool(BaseTool):
    """Generate images using AI image generation services.

    Supports Google Gemini image generation:
    - gemini-2.5-flash-image (high quality, conversational editing, text rendering)

    The tool will generate an image based on your prompt and save it to the workspace.

    Note: Requires GOOGLE_API_KEY environment variable and httpx library.
    """

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Generate image using AI.

        Args:
            call: Tool call with prompt and save_path
            context: Tool execution context

        Returns:
            ToolResult with generated image path and metadata
        """
        prompt = call.arguments.get("prompt")
        save_path = call.arguments.get("save_path")
        aspect_ratio = call.arguments.get("aspect_ratio", "1:1")

        if not prompt:
            return ToolResult(
                success=False,
                error="prompt is required - describe the image you want to generate",
                error_code=E_VALIDATION,
            )

        if not save_path:
            return ToolResult(
                success=False,
                error="save_path is required - specify where to save the generated image (e.g., 'public/hero.jpg')",
                error_code=E_VALIDATION,
            )

        # Validate aspect ratio
        valid_ratios = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
        if aspect_ratio not in valid_ratios:
            return ToolResult(
                success=False,
                error=f"Invalid aspect_ratio: {aspect_ratio}. Valid ratios: {', '.join(valid_ratios)}",
                error_code=E_VALIDATION,
            )

        # Check if httpx is available
        try:
            import httpx  # noqa: F401
        except ImportError:
            return ToolResult(
                success=False,
                error="ImageGenerationTool requires httpx library. Install with: pip install httpx",
                error_code=E_VALIDATION,
            )

        # Resolve save path relative to workspace
        abs_save_path = Path(context.workspace.resolve_path(save_path))

        # Create parent directory if it doesn't exist
        abs_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate image with Gemini
        return await self._generate_with_gemini(prompt, abs_save_path, aspect_ratio, context)

    async def _generate_with_gemini(
        self,
        prompt: str,
        save_path: Path,
        aspect_ratio: str,
        context: ToolContext,
    ) -> ToolResult:
        """Generate image using Google Gemini.

        Requires GOOGLE_API_KEY environment variable.
        """
        import httpx

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="Gemini image generation requires GOOGLE_API_KEY environment variable. Get your key at https://ai.google.dev/",
                error_code=E_VALIDATION,
            )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"

                headers = {
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                }

                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseModalities": ["Image"],
                        "imageConfig": {"aspectRatio": aspect_ratio},
                    },
                }

                context.logger.info(
                    "Generating image with Gemini",
                    model="gemini-2.5-flash-image",
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )

                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                # Extract image from response
                if not data.get("candidates") or not data["candidates"][0].get("content"):
                    return ToolResult(
                        success=False,
                        error="No image generated in response",
                        error_code=E_VALIDATION,
                    )

                # Find the inline image data in parts
                image_data = None
                for part in data["candidates"][0]["content"]["parts"]:
                    if "inlineData" in part:
                        image_data = part["inlineData"]["data"]
                        break

                if not image_data:
                    return ToolResult(
                        success=False,
                        error="No image data found in response",
                        error_code=E_VALIDATION,
                    )

                # Decode base64 image and save
                image_bytes = base64.b64decode(image_data)
                save_path.write_bytes(image_bytes)

                context.logger.info(
                    "Image generated with Gemini",
                    save_path=str(save_path),
                )

                return ToolResult(
                    success=True,
                    output=f"✓ Image generated with Gemini and saved to: {save_path}\n\nPrompt: {prompt}\nAspect Ratio: {aspect_ratio}\nModel: gemini-2.5-flash-image",
                    data={
                        "save_path": str(save_path),
                        "provider": "gemini",
                        "model": "gemini-2.5-flash-image",
                        "prompt": prompt,
                        "aspect_ratio": aspect_ratio,
                    },
                )

        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            context.logger.error(
                "Gemini API error", status_code=e.response.status_code, error=error_text
            )
            return ToolResult(
                success=False,
                error=f"Gemini API error: {e.response.status_code} - {error_text}",
                error_code=E_VALIDATION,
            )
        except httpx.TimeoutException:
            context.logger.error("Gemini API timeout", prompt=prompt)
            return ToolResult(
                success=False,
                error="Gemini API request timed out",
                error_code=E_VALIDATION,
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            ToolDefinition with image generation schema
        """
        return ToolDefinition(
            name="generate_image",
            description="""Generates images using Google Gemini AI image generation (gemini-2.5-flash-image).

Use this tool when you need to:
- Create custom images, illustrations, or artwork
- Generate images that don't exist in stock photo libraries
- Create specific visuals for websites, apps, or design projects
- Produce images with high-fidelity text rendering (logos, diagrams, posters)

The tool will:
1. Generate an image based on your detailed prompt
2. Save it to the specified path in your workspace
3. Provide metadata about the generation

Key features:
- High-quality image generation with conversational editing
- Accurate text rendering for logos and typography
- Multiple aspect ratios (1:1, 16:9, 9:16, 21:9, and more)
- All generated images include a SynthID watermark

Usage notes:
- Be detailed and specific in your prompts for best results
- The tool will automatically create parent directories if needed
- Images are saved in PNG format
- Requires GOOGLE_API_KEY environment variable""",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image you want to generate (e.g., 'A futuristic city skyline at sunset with flying cars, cyberpunk style, highly detailed')",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Path where to save the generated image (e.g., 'public/images/hero.png', 'assets/logo.png')",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": [
                            "1:1",
                            "2:3",
                            "3:2",
                            "3:4",
                            "4:3",
                            "4:5",
                            "5:4",
                            "9:16",
                            "16:9",
                            "21:9",
                        ],
                        "default": "1:1",
                        "description": "Aspect ratio for the generated image. Default: '1:1' (1024x1024px). Popular: '16:9' (landscape), '9:16' (portrait)",
                    },
                },
                "required": ["prompt", "save_path"],
            },
        )


class VideoFetchTool(BaseTool):
    """Fetch videos from popular stock video sources.

    Supports Pexels (free stock videos).

    The tool will search for videos matching the query and download the best match
    to the specified location in the workspace.

    Note: Requires httpx library and PEXELS_API_KEY environment variable.
    """

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Fetch video from a stock video source.

        Args:
            call: Tool call with query, source, and save_path
            context: Tool execution context

        Returns:
            ToolResult with video path and metadata
        """
        query = call.arguments.get("query")
        source = call.arguments.get("source", "pexels").lower()
        save_path = call.arguments.get("save_path")
        orientation = call.arguments.get("orientation", "landscape")
        min_duration = call.arguments.get("min_duration", 5)
        max_duration = call.arguments.get("max_duration", 30)

        if not query:
            return ToolResult(
                success=False,
                error="query is required - describe the video you want to fetch",
                error_code=E_VALIDATION,
            )

        if not save_path:
            return ToolResult(
                success=False,
                error="save_path is required - specify where to save the video (e.g., 'public/video.mp4')",
                error_code=E_VALIDATION,
            )

        # Validate source
        supported_sources = ["pexels"]
        if source not in supported_sources:
            return ToolResult(
                success=False,
                error=f"Unsupported source: {source}. Supported sources: {', '.join(supported_sources)}",
                error_code=E_VALIDATION,
            )

        # Check if httpx is available
        try:
            import httpx  # noqa: F401
        except ImportError:
            return ToolResult(
                success=False,
                error="VideoFetchTool requires httpx library. Install with: pip install httpx",
                error_code=E_VALIDATION,
            )

        # Resolve save path relative to workspace
        abs_save_path = Path(context.workspace.resolve_path(save_path))

        # Create parent directory if it doesn't exist
        abs_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Fetch video based on source
        if source == "pexels":
            return await self._fetch_from_pexels(
                query, abs_save_path, orientation, min_duration, max_duration, context
            )
        else:
            return ToolResult(
                success=False,
                error=f"Source '{source}' not yet implemented",
                error_code=E_VALIDATION,
            )

    async def _fetch_from_pexels(
        self,
        query: str,
        save_path: Path,
        orientation: str,
        min_duration: int,
        max_duration: int,
        context: ToolContext,
    ) -> ToolResult:
        """Fetch video from Pexels.

        Uses Pexels' free API. Requires PEXELS_API_KEY environment variable.
        Get your key at: https://www.pexels.com/api/
        """
        import httpx

        api_key = os.getenv("PEXELS_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="Pexels requires PEXELS_API_KEY environment variable. Get your key at https://www.pexels.com/api/",
                error_code=E_VALIDATION,
            )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Search for videos
                search_url = "https://api.pexels.com/videos/search"
                headers = {"Authorization": api_key}
                params = {
                    "query": query,
                    "per_page": 10,  # Get more to filter by duration
                    "orientation": orientation,
                }

                response = await client.get(search_url, headers=headers, params=params)  # type: ignore[arg-type]
                response.raise_for_status()
                data = response.json()

                if not data.get("videos"):
                    return ToolResult(
                        success=False,
                        error=f"No videos found for query: {query}",
                        error_code=E_VALIDATION,
                    )

                # Filter videos by duration
                suitable_videos = [
                    v
                    for v in data["videos"]
                    if min_duration <= v.get("duration", 0) <= max_duration
                ]

                if not suitable_videos:
                    return ToolResult(
                        success=False,
                        error=f"No videos found matching duration range {min_duration}-{max_duration} seconds for query: {query}",
                        error_code=E_VALIDATION,
                    )

                # Get the first suitable result
                video = suitable_videos[0]

                # Find the HD video file (prefer 1080p, then 720p, then SD)
                video_file = None
                for file in video.get("video_files", []):
                    if file.get("quality") == "hd" and file.get("width") >= 1920:
                        video_file = file
                        break

                if not video_file:
                    for file in video.get("video_files", []):
                        if file.get("quality") == "hd":
                            video_file = file
                            break

                if not video_file:
                    video_file = video["video_files"][0]  # Fallback to first available

                video_url = video_file["link"]
                videographer = video.get("user", {}).get("name", "Unknown")
                video_page_url = video.get("url", "")
                duration = video.get("duration", 0)
                width = video_file.get("width", 0)
                height = video_file.get("height", 0)

                context.logger.info(
                    "Downloading video from Pexels",
                    query=query,
                    duration=duration,
                    resolution=f"{width}x{height}",
                )

                # Download the video
                video_response = await client.get(video_url)
                video_response.raise_for_status()

                # Save to file
                save_path.write_bytes(video_response.content)

                context.logger.info(
                    "Video fetched from Pexels",
                    query=query,
                    save_path=str(save_path),
                    videographer=videographer,
                )

                return ToolResult(
                    success=True,
                    output=f"✓ Video fetched from Pexels and saved to: {save_path}\n\nQuery: {query}\nVideographer: {videographer}\nDuration: {duration}s\nResolution: {width}x{height}\nSource: {video_page_url}\n\nIMPORTANT: When using Pexels videos, please credit the videographer:\nVideo by {videographer} on Pexels ({video_page_url})",
                    data={
                        "save_path": str(save_path),
                        "source": "pexels",
                        "query": query,
                        "videographer": videographer,
                        "video_url": video_page_url,
                        "duration": duration,
                        "width": width,
                        "height": height,
                    },
                )
        except httpx.HTTPStatusError as e:
            context.logger.error(
                "Pexels API error", status_code=e.response.status_code, error=str(e)
            )
            return ToolResult(
                success=False,
                error=f"Pexels API error: {e.response.status_code}",
                error_code=E_VALIDATION,
            )
        except httpx.TimeoutException:
            context.logger.error("Pexels API timeout", query=query)
            return ToolResult(
                success=False,
                error="Pexels API request timed out",
                error_code=E_VALIDATION,
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            ToolDefinition with video fetch schema
        """
        return ToolDefinition(
            name="fetch_video",
            description="""Fetches high-quality videos from Pexels stock video library.

Use this tool when you need to:
- Find and download videos for websites, apps, or design projects
- Get high-quality stock footage matching specific descriptions
- Fetch videos with specific orientations and durations

The tool will:
1. Search Pexels for videos matching your query
2. Download the best matching HD video
3. Save it to the specified path in your workspace
4. Provide attribution information for the videographer

Supported source:
- pexels: Free stock videos (requires PEXELS_API_KEY)

Usage notes:
- Be specific in your query for best results (e.g., "ocean waves crashing on beach" instead of just "ocean")
- The tool will automatically create parent directories if they don't exist
- Always credit videographers when required
- Videos are downloaded in HD quality (1080p or 720p when available)""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the video you want (e.g., 'city traffic timelapse', 'waterfall in forest', 'coffee being poured')",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["pexels"],
                        "default": "pexels",
                        "description": "Video source to fetch from. Only Pexels is supported.",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Path where to save the video in the workspace (e.g., 'public/videos/hero.mp4', 'assets/background.mp4')",
                    },
                    "orientation": {
                        "type": "string",
                        "enum": ["landscape", "portrait", "square"],
                        "default": "landscape",
                        "description": "Video orientation preference. Default: landscape",
                    },
                    "min_duration": {
                        "type": "integer",
                        "default": 5,
                        "description": "Minimum video duration in seconds. Default: 5",
                    },
                    "max_duration": {
                        "type": "integer",
                        "default": 30,
                        "description": "Maximum video duration in seconds. Default: 30",
                    },
                },
                "required": ["query", "save_path"],
            },
        )


class VideoGenerationTool(BaseTool):
    """Generate videos using AI video generation services.

    Supports Google Veo 3.1 Fast:
    - Fast video generation with high quality
    - 6-second, 720p videos with native audio
    - Text-to-video generation

    The tool will generate a video based on your prompt and save it to the workspace.

    Note: Requires GOOGLE_API_KEY environment variable and httpx library.
    Video generation has a 1.5 minute timeout.
    """

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Generate video using AI.

        Args:
            call: Tool call with prompt and save_path
            context: Tool execution context

        Returns:
            ToolResult with generated video path and metadata
        """
        prompt = call.arguments.get("prompt")
        save_path = call.arguments.get("save_path")

        if not prompt:
            return ToolResult(
                success=False,
                error="prompt is required - describe the video you want to generate",
                error_code=E_VALIDATION,
            )

        if not save_path:
            return ToolResult(
                success=False,
                error="save_path is required - specify where to save the generated video (e.g., 'public/video.mp4')",
                error_code=E_VALIDATION,
            )

        # Check if httpx is available
        try:
            import httpx  # noqa: F401
        except ImportError:
            return ToolResult(
                success=False,
                error="VideoGenerationTool requires httpx library. Install with: pip install httpx",
                error_code=E_VALIDATION,
            )

        # Resolve save path relative to workspace
        abs_save_path = Path(context.workspace.resolve_path(save_path))

        # Create parent directory if it doesn't exist
        abs_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate video with Veo
        return await self._generate_with_veo(prompt, abs_save_path, context)

    async def _generate_with_veo(
        self,
        prompt: str,
        save_path: Path,
        context: ToolContext,
    ) -> ToolResult:
        """Generate video using Google Veo 3.1 Fast.

        Requires GOOGLE_API_KEY environment variable.
        """
        import httpx

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="Veo video generation requires GOOGLE_API_KEY environment variable. Get your key at https://ai.google.dev/",
                error_code=E_VALIDATION,
            )

        try:
            async with httpx.AsyncClient(
                timeout=120.0
            ) as client:  # 2 minute timeout for individual requests
                base_url = "https://generativelanguage.googleapis.com/v1beta"
                generate_url = f"{base_url}/models/veo-3.1-fast-generate-preview:predictLongRunning"

                headers = {
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                }

                payload = {"instances": [{"prompt": prompt}]}

                context.logger.info(
                    "Starting video generation with Veo 3.1 Fast",
                    model="veo-3.1-fast-generate-preview",
                    prompt=prompt,
                )

                # Start the long-running operation
                response = await client.post(generate_url, headers=headers, json=payload)
                response.raise_for_status()
                operation_data = response.json()

                operation_name = operation_data.get("name")
                if not operation_name:
                    return ToolResult(
                        success=False,
                        error="No operation name returned from Veo API",
                        error_code=E_VALIDATION,
                    )

                context.logger.info(
                    "Video generation started, polling for completion",
                    operation=operation_name,
                )

                # Poll the operation status until complete
                operation_url = f"{base_url}/{operation_name}"
                max_polls = 18  # 1.5 minutes max (5 second intervals)
                poll_count = 0

                while poll_count < max_polls:
                    await asyncio.sleep(5)  # Wait 5 seconds between polls
                    poll_count += 1

                    status_response = await client.get(
                        operation_url, headers={"x-goog-api-key": api_key}
                    )
                    status_response.raise_for_status()
                    status_data = status_response.json()

                    if status_data.get("done"):
                        # Check for errors
                        if "error" in status_data:
                            error_msg = status_data["error"].get("message", "Unknown error")
                            return ToolResult(
                                success=False,
                                error=f"Veo video generation failed: {error_msg}",
                                error_code=E_VALIDATION,
                            )

                        # Extract video URI
                        response_data = status_data.get("response", {})
                        video_response = response_data.get("generateVideoResponse", {})
                        generated_samples = video_response.get("generatedSamples", [])

                        if not generated_samples:
                            return ToolResult(
                                success=False,
                                error="No video generated in response",
                                error_code=E_VALIDATION,
                            )

                        video_uri = generated_samples[0].get("video", {}).get("uri")
                        if not video_uri:
                            return ToolResult(
                                success=False,
                                error="No video URI found in response",
                                error_code=E_VALIDATION,
                            )

                        context.logger.info(
                            "Video generation completed, downloading",
                            video_uri=video_uri,
                        )

                        # Download the video
                        video_response = await client.get(
                            video_uri,
                            headers={"x-goog-api-key": api_key},
                            follow_redirects=True,
                        )
                        video_response.raise_for_status()

                        # Save to file
                        save_path.write_bytes(video_response.content)

                        context.logger.info(
                            "Video generated with Veo",
                            save_path=str(save_path),
                            poll_count=poll_count,
                        )

                        return ToolResult(
                            success=True,
                            output=f"✓ Video generated with Veo 3.1 Fast and saved to: {save_path}\n\nPrompt: {prompt}\nModel: veo-3.1-fast-generate-preview\nDuration: 6 seconds\nResolution: 720p with native audio\n\nNote: Videos created by Veo are watermarked with SynthID.",
                            data={
                                "save_path": str(save_path),
                                "provider": "google",
                                "model": "veo-3.1-fast-generate-preview",
                                "prompt": prompt,
                                "generation_time_seconds": poll_count * 5,
                            },
                        )

                    if poll_count % 6 == 0:  # Log every 30 seconds
                        context.logger.info(
                            "Still waiting for video generation",
                            elapsed_seconds=poll_count * 5,
                        )

                # Timeout
                return ToolResult(
                    success=False,
                    error=f"Video generation timed out after {max_polls * 5} seconds (1.5 minutes). The operation may still be processing on the server.",
                    error_code=E_VALIDATION,
                )

        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            context.logger.error(
                "Veo API error", status_code=e.response.status_code, error=error_text
            )
            return ToolResult(
                success=False,
                error=f"Veo API error: {e.response.status_code} - {error_text}",
                error_code=E_VALIDATION,
            )
        except httpx.TimeoutException:
            context.logger.error("Veo API timeout", prompt=prompt)
            return ToolResult(
                success=False,
                error="Veo API request timed out",
                error_code=E_VALIDATION,
            )

    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM.

        Returns:
            ToolDefinition with video generation schema
        """
        return ToolDefinition(
            name="generate_video",
            description="""Generates videos using Google Veo 3.1 Fast AI video generation.

Use this tool when you need to:
- Create custom videos, animations, or cinematic content
- Generate videos that don't exist in stock video libraries
- Create specific video content for websites, apps, or social media
- Produce realistic videos with dialogue, sound effects, and native audio

The tool will:
1. Generate a 6-second, 720p video based on your detailed prompt
2. Poll the server until the video is ready (timeout: 1.5 minutes)
3. Save it to the specified path in your workspace
4. Provide metadata about the generation

Key features:
- Fast, high-quality video generation with native audio
- Wide range of visual and cinematic styles
- Natural dialogue and sound effects generation
- 720p resolution at 24fps
- SynthID watermarking for AI-generated content verification

Usage notes:
- Be detailed and specific in your prompts for best results
- Include dialogue in quotes if you want characters to speak specific words
- Specify camera movements, lighting, and style for better control
- The tool will automatically create parent directories if needed
- Videos are saved in MP4 format
- Requires GOOGLE_API_KEY environment variable
- Generation timeout is set to 1.5 minutes

Example prompts:
- "A close up of two people staring at a cryptic drawing on a wall, torchlight flickering. A man murmurs, 'This must be it.'"
- "Aerial shot of a majestic waterfall in a lush Hawaiian rainforest, camera slowly panning to reveal the full height of the falls"
- "Time-lapse of a city skyline transitioning from day to night, with lights turning on in buildings"
""",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the video you want to generate. Include camera movements, dialogue (in quotes), visual style, and atmosphere. Maximum 1024 tokens.",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Path where to save the generated video (e.g., 'public/videos/hero.mp4', 'assets/background.mp4')",
                    },
                },
                "required": ["prompt", "save_path"],
            },
        )
