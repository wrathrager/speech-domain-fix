# GPT API post-processor
"""
llm_correction.py
-----------------
Module for correcting ASR (Automatic Speech Recognition) transcripts
using a Large Language Model (LLM) like OpenAI's GPT.

Usage:
    from llm_correction import LLMCorrection

    corrector = LLMCorrection(model="gpt-4o-mini")
    corrected_text = corrector.correct("i goed to school yestarday")
    print(corrected_text)
"""

import os
from typing import Optional
from openai import OpenAI


class LLMCorrection:
    """
    A class to perform text correction using an LLM.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the correction module.

        Args:
            model (str): Model name to use for correction.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your environment."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def correct(self, text: str, system_prompt: Optional[str] = None) -> str:
        """
        Correct the given text using the LLM.

        Args:
            text (str): The input text to correct.
            system_prompt (Optional[str]): Optional instruction for correction style.

        Returns:
            str: Corrected text.
        """
        if not system_prompt:
            system_prompt = (
                "You are a text correction system. "
                "Fix grammar, spelling, and clarity without changing meaning."
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Example usage
    corrector = LLMCorrection()
    sample_text = "i goed to school yestarday and meeted my freind"
    print("Original:", sample_text)
    print("Corrected:", corrector.correct(sample_text))
