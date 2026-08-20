"""
Tool 3: LLM-as-Judge Tool
Uses Google Gemini API for structured verdict on ambiguous reviews.
Only called when classifier confidence < 0.65.
"""
import os
import time
import json
from typing import Optional

class JudgeTool:
    """
    Calls Google Gemini API (free tier) for a structured verdict.
    Gracefully degrades if API key is not set.
    """

    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.available = self.api_key is not None
        self.model_name = None
        self.client = None

        if self.available:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)

                # Use stable Flash model designed for high-throughput workloads
                self.model_name = "gemini-3.5-flash"

                self.client = genai
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

        Args:
            review_text: the review to judge
            tool1_signal: risk score from similarity tool (0-1)
            tool2_prediction: classifier prediction (0=genuine, 1=fake)
            tool2_confidence: classifier confidence (0-1)

        Returns:
            dict with:
                - 'verdict': int (0=genuine, 1=fake, -1=uncertain)
                - 'reasoning': str (brief explanation)
                - 'latency_seconds': float
                - 'tool_name': 'judge'
                - 'api_called': bool (whether API was actually called)
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

        # Build prompt
        prompt = f"""You are an expert at detecting AI-generated fake product reviews.
Analyze the following review and determine if it is likely human-written (genuine) or AI-generated (fake).

Review text:
"{review_text}"

Context from prior analysis:
- Similarity clustering risk score (0-1, higher = more templated): {tool1_signal:.2f}
- Classifier prediction (0=genuine, 1=fake): {tool2_prediction}
- Classifier confidence: {tool2_confidence:.2f}

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
5. Prior risk signals from clustering and classifier

Respond only with the JSON object."""

        try:
            # Use correct Google Generative AI API with retry logic
            import time as time_module
            from google.api_core import retry as google_retry

            model = self.client.GenerativeModel(self.model_name)

            # Retry with exponential backoff for rate limits
            max_retries = 5
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config={"temperature": 0.3}
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"Rate limited. Retrying in {retry_delay}s... (attempt {attempt+1}/{max_retries})")
                            time_module.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, 60)  # Exponential backoff, max 60s
                        else:
                            raise  # Re-raise on final attempt
                    else:
                        raise

            # Parse response
            response_text = response.text.strip()

            # Try to extract JSON
            try:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response_text

                result = json.loads(json_str)
                verdict = int(result.get('verdict', -1))
                reasoning = result.get('reasoning', 'No reasoning provided')
            except (json.JSONDecodeError, IndexError, ValueError) as e:
                # Fallback: use classifier prediction if parsing fails
                verdict = tool2_prediction
                reasoning = f"Parsing error in API response; using classifier prediction as fallback"

            latency = time.time() - start_time

            return {
                'verdict': verdict,
                'reasoning': reasoning,
                'latency_seconds': latency,
                'tool_name': 'judge',
                'api_called': True
            }

        except Exception as e:
            print(f"Gemini API call failed: {e}")
            return {
                'verdict': tool2_prediction,
                'reasoning': f'API call failed; using classifier prediction as fallback: {str(e)[:100]}',
                'latency_seconds': time.time() - start_time,
                'tool_name': 'judge',
                'api_called': False
            }
