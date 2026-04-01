#!/usr/bin/env python3
"""
01_generate_synthetic_GSS.py

Query LLMs with COMPREHENSIVE attitudinal battery including:
- All 30 original culture-war items (abortion, guns, immigration, etc.)
- 22 new non-culture-war items (spending, trust, economic outlook)

Total: 52 attitudinal items to test whether constraint/inflation patterns
extend beyond hot-button identity politics.

Features:
- ✅ Proper resume functionality (skips already-completed persona/question/run combinations)
- ✅ Retry logic for failed API calls with exponential backoff
- ✅ Year-specific persona loading
- ✅ Batch saving with progress tracking

Usage:
    python 01_generate_synthetic_GSS.py --year 2024 --all-models --personas 600
    python 01_generate_synthetic_GSS.py --year 2016 --models openai/gpt-4o --personas 300

Set OPENROUTER_API_KEY environment variable before running.
"""

import os
import json
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Set, Tuple
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent  # generation/ directory (one level up from scripts/)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# === COMPREHENSIVE GSS QUESTIONS ===
# Part 1: Original 30 culture-war items
GSS_QUESTIONS_CULTUREWAR = {

    # Economic (2)
    "affrmact": {
        "text": "Some people say that because of past discrimination, Black people should be given preference in hiring and promotion. Others say that such preference in hiring and promotion of Black people is wrong because it discriminates against whites. What about your opinion--are you for or against preferential hiring and promotion of Black people?",
        "options": {1: "Strongly favor", 2: "Not strongly favor", 3: "Not strongly oppose", 4: "Strongly oppose"}
    },
    "eqwlth": {
        "text": "Some people think that the government in Washington ought to reduce the income differences between the rich and the poor, perhaps by raising the taxes of wealthy families or by giving income assistance to the poor. Others think that the government should not concern itself with reducing this income difference between the rich and the poor. Here is a card with a scale from 1 to 7. Think of a score of 1 as meaning that the government ought to reduce the income differences between rich and poor, and a score of 7 meaning that the government should not concern itself with reducing income differences. What score between 1 and 7 comes closest to the way you feel?",
        "options": {1: "The Government should reduce differences", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "The government should not concern itself with reducing income differences"}
    },
    # Confidence in institutions (3)
    "confed": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?: Executive branch of the federal government.",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conpress": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?: The Press.",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "consci": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?: The Scientific Community.",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
}

# Part 2: Non-culture-war items (spending, trust, economic)
GSS_QUESTIONS_NONCULTUREWAR = {

    # === INSTITUTIONAL CONFIDENCE (CON* series) ===
    "coneduc": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Education?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conbus": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Major companies?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "confinan": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Banks and financial institutions?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conlegis": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Congress?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conarmy": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Military?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },
    "conmedic": {
        "text": "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them: Medicine?",
        "options": {1: "A great deal", 2: "Only some", 3: "Hardly any"}
    },

    # === SOCIAL TRUST ===
    "trust": {
        "text": "Generally speaking, would you say that most people can be trusted or that you can't be too careful in dealing with people?",
        "options": {1: "Can trust", 2: "Can't be too careful", 3: "Depends"}
    },
    "fair": {
        "text": "Do you think most people would try to take advantage of you if they got a chance, or would they try to be fair?",
        "options": {1: "Would take advantage of you", 2: "Would try to be fair", 3: "Depends"}
    },
    "helpful": {
        "text": "Would you say that most of the time people try to be helpful, or that they are mostly just looking out for themselves?",
        "options": {1: "Try to be helpful", 2: "Looking out for themselves", 3: "Depends"}
    },

    # === ECONOMIC OUTLOOK & CLASS ===
    "satfin": {
        "text": "We are interested in how people are getting along financially these days. So far as you and your family are concerned, would you say that you are pretty well satisfied with your present financial situation, more or less satisfied, or not satisfied at all?",
        "options": {1: "Pretty well satisfied", 2: "More or less satisfied", 3: "Not satisfied at all"}
    },
    "finalter": {
        "text": "During the last few years, has your financial situation been getting better, worse, or has it stayed the same?",
        "options": {1: "Getting better", 2: "Stayed the same", 3: "Getting worse"}
    },
    "getahead": {
        "text": "Some people say that people get ahead by their own hard work; others say that lucky breaks or help from other people are more important. Which do you think is most important?",
        "options": {1: "Hard work most important", 2: "Hard work, luck equally important", 3: "Luck most important"}
    },
}

# Combine both dictionaries into comprehensive battery
GSS_QUESTIONS_COMPREHENSIVE = {**GSS_QUESTIONS_CULTUREWAR, **GSS_QUESTIONS_NONCULTUREWAR}

# Popular models on OpenRouter
POPULAR_MODELS = [
    # existing models
    # "deepseek/deepseek-chat-v3.1",
    # "google/gemini-2.5-flash",
    # "google/gemma-3-12b-it",
    # "openai/gpt-4o-mini",
    # "openai/gpt-oss-120b",
    # "meta-llama/llama-3.1-8b-instruct",
    # "meta-llama/llama-3.3-70b-instruct",
    # "meta-llama/llama-4-maverick",
    "mistralai/mistral-small-3.2-24b-instruct",
    "mistralai/mistral-medium-3.1",
    "mistralai/mistral-nemo",
    "z-ai/glm-4.6",
    "qwen/qwen-2.5-72b-instruct",
    "arcee-ai/trinity-mini",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4-fast",

    # --- new high-usage / frontier models from OpenRouter ---

    # OpenAI 5-series
    "openai/gpt-5-mini",
    "openai/gpt-5",

    # Gemini tiers
    "google/gemini-2.5-flash-lite",

    # Newer DeepSeek chat models
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-v3.2-20251201",

    # New Mistral flagship
    "mistralai/mistral-large-2512",

    # New Qwen3 large instruct
    "qwen/qwen3-235b-a22b-2507",

    # Popular non-US giant
    "minimax/minimax-m2",

    # non-mainstream additions
    "moonshotai/kimi-k2",
    "ai21/jamba-large-1.7",
    "cohere/command-r-plus-08-2024",
    "allenai/olmo-3-7b-instruct",
    "deepcogito/cogito-v2-preview-llama-405b",
]


def query_openrouter(
    model: str,
    persona: str,
    question: str,
    options: Dict[int, str],
    api_key: str,
    year: int,
    timeout: int = 30,
    max_retries: int = 3
) -> Dict:
    """
    Query OpenRouter API for a single question with retry logic.

    Args:
        model: Model identifier
        persona: Natural language persona description
        question: Survey question text
        options: Dict mapping option numbers to text
        api_key: OpenRouter API key
        year: Survey year for temporal context
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        Dict with 'answer', 'error', token counts, and 'raw_response'
    """

    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

    prompt = f"""It is now {year}. You are answering survey questions as the following person, who is living in the United States:

{persona}

Question: {question}

Options:
{options_text}

Respond with ONLY the number of your answer (e.g., "1" or "2"). Do not explain your reasoning."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 1.0, 
        "max_tokens": 50
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()

            result = response.json()
            answer_text = result['choices'][0]['message']['content'].strip()

            try:
                answer = int(answer_text)
                if answer not in options:
                    return {
                        "answer": None,
                        "error": f"Invalid answer: {answer_text}",
                        "prompt_tokens": result.get('usage', {}).get('prompt_tokens', 0),
                        "completion_tokens": result.get('usage', {}).get('completion_tokens', 0),
                        "raw_response": answer_text
                    }
            except ValueError:
                return {
                    "answer": None,
                    "error": f"Could not parse answer: {answer_text}",
                    "prompt_tokens": result.get('usage', {}).get('prompt_tokens', 0),
                    "completion_tokens": result.get('usage', {}).get('completion_tokens', 0),
                    "raw_response": answer_text
                }

            return {
                "answer": answer,
                "error": None,
                "prompt_tokens": result.get('usage', {}).get('prompt_tokens', 0),
                "completion_tokens": result.get('usage', {}).get('completion_tokens', 0),
                "raw_response": answer_text
            }

        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            break

    return {
        "answer": None,
        "error": last_error,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "raw_response": ""
    }


def load_completed_tasks(output_file: Path) -> Set[Tuple[int, str, int]]:
    """
    Load already-completed tasks from existing output file.

    Returns:
        Set of (persona_id, variable, run) tuples
    """
    if not output_file.exists():
        return set()

    try:
        df = pd.read_csv(output_file)
        completed = set(
            df[df['answer'].notna()][['persona_id', 'variable', 'run']]
            .apply(tuple, axis=1)
        )
        return completed
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Query LLMs with comprehensive GSS attitudinal items"
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        choices=[2024, 2016, 2008, 2000],
        help="Survey year to simulate (2024, 2016, 2008, or 2000)"
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Use all popular models"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per persona-question pair (default: 1)"
    )
    parser.add_argument(
        "--personas",
        type=int,
        default=600,
        help="Number of personas to query (default: 600)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of results to accumulate before saving (default: 100)"
    )

    args = parser.parse_args()
    year = args.year

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Get your API key from https://openrouter.ai/keys"
        )

    # Determine models to use
    if args.all_models:
        models = POPULAR_MODELS
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        raise ValueError("Must specify either --models or --all-models")

    # Create output directory
    output_base = PROJECT_DIR / "synthetic_data"
    output_dir = output_base / f"year_{year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Comprehensive GSS Survey - Year {year}")
    print("=" * 70)
    print()
    print(f"Models: {len(models)}")
    for model in models:
        print(f"  - {model}")
    print(f"\nPersonas: {args.personas}")
    print(f"Questions: {len(GSS_QUESTIONS_COMPREHENSIVE)}")
    print(f"  - Culture-war items: {len(GSS_QUESTIONS_CULTUREWAR)}")
    print(f"  - Non-culture-war items: {len(GSS_QUESTIONS_NONCULTUREWAR)}")
    print(f"Runs per persona-question: {args.runs}")
    print(f"Max workers: {args.max_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}")
    print()

    # Load personas - USE YEAR-SPECIFIC PERSONAS
    personas_file = PROJECT_DIR / "data" / f"gss{year}_personas.csv"
    if not personas_file.exists():
        # Fallback to 2024 if year-specific doesn't exist
        print(f"Warning: {personas_file} not found, falling back to 2024 personas")
        personas_file = PROJECT_DIR / "data" / "gss2024_personas.csv"

    if not personas_file.exists():
        raise FileNotFoundError(f"Personas file not found: {personas_file}")

    personas_df = pd.read_csv(personas_file)

    # Sample personas
    if args.personas and args.personas > 0 and len(personas_df) > args.personas:
        random.seed(42)
        personas_df = personas_df.sample(n=args.personas, random_state=42)

    print(f"Loaded {len(personas_df)} personas from {personas_file.name}")
    print()

    # Calculate total requests
    total_requests = len(models) * len(personas_df) * len(GSS_QUESTIONS_COMPREHENSIVE) * args.runs
    print(f"Total requests: {total_requests:,}")
    print()

    # Process each model
    for model in models:
        print("=" * 70)
        print(f"Processing: {model}")
        print("=" * 70)
        print()

        # Clean model name for filename
        model_filename = model.replace("/", "_")
        output_file = output_dir / f"{model_filename}.csv"

        # Load already-completed tasks for resume functionality
        completed_tasks = load_completed_tasks(output_file)

        if completed_tasks:
            print(f"Found {len(completed_tasks)} already-completed tasks")
            print("Resuming from previous run...")

        # Build task list (excluding already-completed)
        tasks = []
        for _, persona_row in personas_df.iterrows():
            persona_id = persona_row['respondent_id']
            persona_text = persona_row['persona']

            for var_name, question_data in GSS_QUESTIONS_COMPREHENSIVE.items():
                for run in range(1, args.runs + 1):
                    task_key = (persona_id, var_name, run)

                    # Skip if already completed
                    if task_key in completed_tasks:
                        continue

                    tasks.append({
                        'persona_id': persona_id,
                        'persona_text': persona_text,
                        'variable': var_name,
                        'question': question_data['text'],
                        'options': question_data['options'],
                        'run': run
                    })

        total_tasks = len(tasks) + len(completed_tasks)

        if not tasks:
            print(f"All tasks already completed for {model}")
            print(f"Output file: {output_file}\n")
            continue

        print(f"Tasks: {len(tasks)} remaining, {len(completed_tasks)} already done ({total_tasks} total)")
        print()

        results = []

        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    query_openrouter,
                    model,
                    task['persona_text'],
                    task['question'],
                    task['options'],
                    api_key,
                    year
                ): task for task in tasks
            }

            pbar = tqdm(total=len(tasks), desc=model.split('/')[-1])

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()

                results.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'model': model,
                    'persona_id': task['persona_id'],
                    'variable': task['variable'],
                    'question_short': task['question'][:50] + '...',
                    'run': task['run'],
                    'answer': result.get('answer'),
                    'prompt_tokens': result.get('prompt_tokens', 0),
                    'completion_tokens': result.get('completion_tokens', 0),
                    'total_tokens': result.get('prompt_tokens', 0) + result.get('completion_tokens', 0),
                    'error': result.get('error', ''),
                    'raw_response': result.get('raw_response', '')
                })

                pbar.update(1)

                # Batch save
                if len(results) >= args.batch_size:
                    results_df = pd.DataFrame(results)
                    if not output_file.exists():
                        results_df.to_csv(output_file, index=False)
                    else:
                        results_df.to_csv(output_file, mode='a', header=False, index=False)
                    results = []

            pbar.close()

        # Save any remaining results
        if results:
            results_df = pd.DataFrame(results)
            if not output_file.exists():
                results_df.to_csv(output_file, index=False)
            else:
                results_df.to_csv(output_file, mode='a', header=False, index=False)

        # Read full results for summary
        results_df = pd.read_csv(output_file)

        # Print summary
        success_rate = (results_df['answer'].notna().sum() / len(results_df)) * 100
        total_tokens = results_df['total_tokens'].sum()

        print(f"\nResults saved: {output_file}")
        print(f"  Total responses: {len(results_df):,}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Total tokens: {total_tokens:,}")
        print()

    print("=" * 70)
    print("All models complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
