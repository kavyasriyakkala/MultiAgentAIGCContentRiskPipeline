"""
Tool 3: LLM-as-Judge Tool
Uses Google Gemini API for structured verdict on ambiguous reviews.
Only called when classifier confidence < threshold or tools disagree.
"""
import os
import time
import json


class JudgeTool:
    """
    Calls Google Gemini API (free tier) for a structured verdict.
    Gracefully degrades if API key is not set.
    Judge is blind to classifier prediction to avoid anchoring.
    """

    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.available = self.api_key is not None
        self.model_name = "gemini-3.8-flash"
        self.client = None

        if self.available:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                print(f"Gemini API initialized with model: {self.model_name}")
            except Exception as e:
                print(f"Failed to initialize Gemini API: {e}")
                self.available = False
        else:
            print("GEMINI_API_KEY not set; LLM-as-Judge will be skipped")

    def __call__(self, review_text: str, tool1_signal: float, tool2_prediction: int,
                 tool2_confidence: float) -> dict:
        """
        Query Gemini for a structured verdict on a review.

        Returns:
            dict with:
                - 'verdict': int (0=genuine, 1=fake, -1=uncertain/failed)
                - 'reasoning': str
                - 'latency_seconds': float
                - 'tool_name': 'judge'
                - 'api_called': bool (True only if API actually responded)
        """
        if not self.available:
            return {
                'verdict': -1,
                'reasoning': 'LLM-as-Judge unavailable (no API key)',
                'latency_seconds': 0.0,
                'tool_name': 'judge',
                'api_called': False
            }

        start_time = time.time()

        # Blind to classifier prediction — only similarity signal provided to avoid anchoring
        prompt = f"""You are an expert at detecting AI-generated fake product reviews.
Analyze the following review and determine if it is likely human-written (genuine) or AI-generated (fake).

Review text:
"{review_text}"

Context from prior analysis:
- Similarity clustering risk score (0-1, higher = more templated/bot-like): {tool1_signal:.2f}

Respond in this exact JSON format:
{{
    "verdict": 0 or 1 (0=genuine, 1=fake),
    "reasoning": "brief explanation (1-2 sentences)"
}}

Consider:
1. Writing quality, grammar, and naturalness
2. Specificity and personal experience (genuine reviews usually have concrete details)
3. Emotional tone (AI often struggles with authentic emotion)
4. Review length and structure (AI tends toward templated patterns)
5. Clustering risk signal above

Respond only with the JSON object."""

        try:
            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={"temperature": 0.3}
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    is_overload = "503" in err_str or "UNAVAILABLE" in err_str
                    if (is_rate_limit or is_overload) and attempt < max_retries - 1:
                        # Parse suggested retry delay from error if present
                        import re as _re
                        match = _re.search(r'retryDelay.*?(\d+)s', err_str)
                        suggested = int(match.group(1)) + 2 if match else retry_delay
                        wait = min(suggested, 65)
                        print(f"Rate limited. Retrying in {wait}s...")
                        time.sleep(wait)
                        retry_delay = min(retry_delay * 2, 30)
                    else:
                        raise

            response_text = response.text.strip()

            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text

            result = json.loads(json_str)
            verdict = int(result.get('verdict', -1))
            reasoning = result.get('reasoning', 'No reasoning provided')

            return {
                'verdict': verdict,
                'reasoning': reasoning,
                'latency_seconds': time.time() - start_time,
                'tool_name': 'judge',
                'api_called': True
            }

        except (json.JSONDecodeError, IndexError, ValueError, KeyError) as e:
            # Parsing error — response came back but was malformed
            return {
                'verdict': -1,
                'reasoning': f'Response parse error: {str(e)[:100]}',
                'latency_seconds': time.time() - start_time,
                'tool_name': 'judge',
                'api_called': True  # API responded, we just couldn't parse it
            }
        except Exception as e:
            print(f"Gemini API call failed: {e}")
            return {
                'verdict': -1,
                'reasoning': f'API call failed: {str(e)[:100]}',
                'latency_seconds': time.time() - start_time,
                'tool_name': 'judge',
                'api_called': False
            }
