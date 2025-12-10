import logging
import asyncio
import os
import json
import threading
import urllib.request
import urllib.parse
import telegram
import sys
import time
import datetime
import re
from difflib import get_close_matches
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    from googletrans import Translator
except Exception:
    Translator = None

try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
except Exception:
    import subprocess
    print("Required package 'python-telegram-bot' not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-telegram-bot>=20.0"])
        # retry import
        from telegram import Update
        from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
    except Exception as install_err:
        print("Automatic installation failed:", install_err)
        print("Please install manually: pip install python-telegram-bot -U")
        raise

from ctransformers import AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# YOUR TELEGRAM TOKEN
TOKEN = "8024688902:AAGkrTWQxm0raHy2tW1RTMntZJWvYoH1hGk"

# Global model variable
llm = None

def load_model():
    """Loads the TinyLlama model. This runs once at startup."""
    global llm
    print("Loading TinyLlama model... This may take a minute.")
    try:
        # Using a quantized GGUF version for efficiency on CPU
        # ctransformers will automatically download the model if not present
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            model_type="llama",
            context_length=1024,
            gpu_layers=0  # Force CPU
        )
        print("Model loaded successfully!")
    except Exception as e:
        logging.critical(f"Failed to load model: {e}")

def clean_response(text):
    """Remove incomplete sentences and filter out system prompt leakage."""
    if not text:
        return text
    
    # Filter out common prompt injection patterns and system message leakage
    unwanted_patterns = [
        r'<\|system\|>.*?<\|user\|>',
        r'<\|assistant\|>',
        r'</s>',
        r'You are.*?assistant\.',
        r'Think step-by-step.*?\.',
        r'Forall Agent.*?by Hrishikesh',
        r'System:.*?\n',
    ]
    
    # Remove unwanted patterns
    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    text = text.strip()
    
    # Remove trailing incomplete sentences
    sentences_end = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    
    # Try to find a complete sentence ending
    last_complete = -1
    for ending in sentences_end:
        idx = text.rfind(ending)
        if idx != -1:
            last_complete = max(last_complete, idx + 1)
    
    if last_complete > 0:
        return text[:last_complete].strip()
    
    # If no sentence ending found, return as is
    return text.strip()

def generate_text(prompt, max_new_tokens: int = 128, temperature: float = 0.7):
    """Generates text using the loaded LLM with configurable parameters."""
    if not llm:
        return "Model is still loading or failed to load. Please try again in a minute."
    
    try:
        # Enhanced system prompt for Forall Agent with better reasoning
        system_msg = """You are Forall Agent, an intelligent, witty, and insightful AI assistant created by Hrishikesh.

Core Traits:
- Sophisticated reasoning and deep analysis capabilities
- Logical, step-by-step thinking approach
- Conversational, warm, and genuinely helpful tone
- Contextual understanding and nuanced responses
- Honest about limitations, never makes up facts
- Creative, curious, and eager to help users learn

Response Guidelines:
- Be concise yet comprehensive
- Use simple language unless technical terms are needed
- Admit uncertainty - don't hallucinate
- Ask clarifying questions if needed
- Never be robotic or overly formal
- Structure complex answers with bullet points when helpful

Remember: Provide genuine value in every response."""
        
        formatted_prompt = f"<|system|>\n{system_msg}<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        # Generate response with timeout protection
        response = llm(formatted_prompt, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=1.1)
        
        if isinstance(response, (list, tuple)):
            response_text = "\n".join(map(str, response))
        else:
            response_text = str(response)
        
        # Validate response quality
        if not response_text or len(response_text.strip()) < 5:
            return "I'm having trouble forming a response. Could you rephrase your question?"
        
        # Clean and return
        cleaned = clean_response(response_text)
        if not cleaned or len(cleaned) < 5:
            return "Sorry, I couldn't generate a proper response. Please try again."
        
        return cleaned
        
    except Exception as e:
        logging.error(f"Generation Error: {e}")
        return "I encountered an error while thinking. Please try again in a moment."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome! I'm **Forall Agent**, an intelligent AI assistant.\n\n"
        "I was created by **Hrishikesh** to be your comprehensive conversational companion.\n\n"
        "I can:\n"
        "💬 Chat naturally with you about anything\n"
        "❓ Answer complex questions with deep reasoning\n"
        "📖 Write engaging stories\n"
        "😄 Tell witty jokes\n"
        "💪 Provide inspiration and motivation\n"
        "🔍 Search Wikipedia for information\n"
        "/ask <question> - Ask anything\n"
        "/story <topic> - Write a story\n"
        "/joke - Get a joke\n"
        "/motivate - Get inspired\n"
        "/wiki <topic> - Search Wikipedia\n"
        "/help - Show all commands",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "**Forall Agent - Commands**\n\n"
        "/ask <question> - Ask me any question\n"
        "/story <topic> - Generate a creative story\n"
        "/joke - Tell me a funny joke\n"
        "/motivate - Get an inspiring message\n"
        "/wiki <topic> - Search Wikipedia\n\n"
        "**Pro Tip:** You can also just chat with me naturally without commands! 💬",
        parse_mode="Markdown"
    )

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = " ".join(context.args).strip()
        if not user_input:
            await update.message.reply_text("Usage: /ask <question>")
            return

        status_msg = await update.message.reply_text("💭 Analyzing your question...")
        prompt = f"Question: {user_input}\n\nProvide a deep, thoughtful answer. Think step-by-step and explain your reasoning clearly."
        response = await asyncio.to_thread(generate_text, prompt, 500, 0.6)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=response)
    except Exception as e:
        logging.error(f"Ask command error: {e}")
        await update.message.reply_text("Sorry, I had trouble processing your question. Please try again.")

async def story(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = " ".join(context.args).strip()
        topic = user_input if user_input else "a random adventure"

        status_msg = await update.message.reply_text(f"📖 Writing a story about {topic}...")
        prompt = (
            f"Write an engaging and creative short story about '{topic}'. The story should be approximately "
            "300-400 words, with a clear beginning, middle, and satisfying ending. Make it interesting, imaginative, "
            "and emotionally engaging. Make sure the story has a complete and satisfying ending."
        )
        # First generation (larger token budget)
        response = await asyncio.to_thread(generate_text, prompt, 700, 0.75)

        # If the model cut off (no terminal punctuation), try to continue up to 3 times
        def is_complete_text(s: str) -> bool:
            s = s.strip()
            return bool(s) and s[-1] in '.!?'

        attempts = 0
        while attempts < 3 and not is_complete_text(response):
            attempts += 1
            cont_prompt = (
                "Continue the previous story and finish it with a satisfying ending. "
                "Do not repeat the previous text; pick up where it left off and conclude the narrative."
            )
            try:
                cont = await asyncio.to_thread(generate_text, cont_prompt + "\n\nPrevious text:\n" + response[-2000:], 200, 0.7)
                if cont and len(cont.strip()) > 0:
                    response = response.rstrip() + "\n\n" + cont.lstrip()
                else:
                    break
            except Exception:
                break

        # Final cleaning/trim
        response = clean_response(response)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=response)
    except Exception as e:
        logging.error(f"Story command error: {e}")
        await update.message.reply_text("Sorry, I had trouble writing a story. Please try again.")

async def joke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        status_msg = await update.message.reply_text("😄 Thinking of a joke (5 lines)...")

        async def try_generate(prompt, tokens=240, temp=0.6):
            try:
                return await asyncio.to_thread(generate_text, prompt, tokens, temp)
            except Exception as e:
                logging.error(f"Joke generation error: {e}")
                return None

        # Primary strict prompt
        prompt1 = (
            "Write a single joke split into EXACTLY FIVE short, non-empty lines.\n"
            "Rules:\n"
            "- Output exactly 5 lines only (use newline characters).\n"
            "- Do NOT use dialogue, character names, or 'Name:'.\n"
            "- Do NOT include numbering, bullets, or any extra text.\n"
            "- Each line should be brief and contribute to setup or punchline; the final line must contain the punchline.\n"
            "- Keep it family-friendly and witty.\n"
            "Now output the five lines, nothing else."
        )

        text = (await try_generate(prompt1)) or ''

        # If primary failed or returned placeholder/fallback, try a simpler prompt
        if not text or len(text.strip()) < 10 or 'couldn\'t think' in text.lower() or 'sorry' in text.lower():
            prompt2 = (
                "Give me a short, witty joke divided into five separate short lines. "
                "Do not use dialogue or names. Output only the five lines."
            )
            text = (await try_generate(prompt2, tokens=180, temp=0.65)) or ''

        # Normalize and clean response text
        text = (text or '').strip()
        # Remove surrounding quotes if any
        text = re.sub(r'^(["“”`]+)|(["“”`]+)$', '', text).strip()

        # Attempt to split into meaningful lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            parts = re.split(r'(?<=[.!?])\s+', text)
            lines = [p.strip() for p in parts if p.strip()]

        # If still fewer than 5, split by commas and semicolons
        if len(lines) < 5:
            frags = re.split(r'[;,]\s*', text)
            frags = [f.strip() for f in frags if f.strip()]
            combined = []
            for f in frags:
                if len(combined) >= 5:
                    break
                sub = re.split(r'(?<=[.!?])\s+', f)
                for s in sub:
                    if s.strip():
                        combined.append(s.strip())
                        if len(combined) >= 5:
                            break
            if combined:
                lines = combined

        # Final fallback: generate canned joke structure if still failing
        if len(lines) < 5:
            logging.warning("Joke generation falling back to canned template")
            setups = [
                "I tried doing a math workout", "Yesterday I forgot my umbrella", "My phone and I had a fight",
                "Heard about the new bakery on the moon", "Tried to teach my cat to code"
            ]
            middles = [
                "but it only did sums", "because the sky was confused", "it said 'battery low' and left",
                "they said the bread was out of this world", "it kept pressing the wrong keys"
            ]
            twists = [
                "Turns out, even numbers need rest.", "So I just stayed inside and made tea.", "Now it only charges at night.",
                "The croissants had no gravity either.", "Now it only naps on the keyboard."
            ]
            # Build 5-line joke: setup fragment, middle fragment, short lead-in, twist, punchline
            lines = [setups[0], middles[0], "Which was awkward for everyone", twists[0], twists[0]]

        # Ensure exactly 5 lines: pad or trim
        if len(lines) < 5:
            while len(lines) < 5:
                lines.append('...')
        else:
            lines = lines[:5]

        # Clean each line
        clean_lines = []
        for ln in lines:
            ln = re.sub(r'^[A-Za-z0-9_\- ]{1,30}:\s*', '', ln)
            ln = re.sub(r'["“”`]', '', ln)
            ln = re.sub(r'\s*-{2,}\s*', ' ', ln)
            ln = re.sub(r'\s*—\s*', ' ', ln)
            ln = re.sub(r'\s+', ' ', ln).strip()
            if len(ln) > 140:
                ln = ln[:137].rsplit(' ', 1)[0] + '...'
            clean_lines.append(ln)

        output = '\n'.join(clean_lines)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=output)
    except Exception as e:
        logging.error(f"Joke command error: {e}")
        await update.message.reply_text("Sorry, I couldn't think of a joke right now. Try again!")

async def motivate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        status_msg = await update.message.reply_text("💪 Finding inspiration...")
        # Friendly, casual encouragement per user's request
        prompt = (
            "You are speaking as a close friend. Keep the tone casual and supportive. Output a short motivational message that says:"
            " 'My friend, u can do it. You have to get through.' "
            "But expand it slightly to be sincere, encouraging, and actionable (one short paragraph). Do not be formal."
        )
        response = await asyncio.to_thread(generate_text, prompt, 140, 0.6)
        # Normalize and ensure casual phrasing contains the requested phrase
        text = (response or "").strip()
        if "my friend" not in text.lower() and "u can do it" not in text.lower():
            # Force a friendly fallback message matching user's style
            text = "my friend u can do it u have to get through"

        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=text)
    except Exception as e:
        logging.error(f"Motivate command error: {e}")
        await update.message.reply_text("Sorry, I had trouble with that. Keep pushing forward though - you've got this! 💪")

def fetch_wiki_summary(query):
    """Fetch Wikipedia summary with smart matching"""
    try:
        # Step 1: Search for possible titles
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=5&namespace=0&format=json"
        search_req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        search_res = urllib.request.urlopen(search_req, timeout=10).read().decode()
        search_data = json.loads(search_res)
        
        suggestions = search_data[1]  # list of possible titles
        
        if not suggestions:
            return {"success": False, "error": "Couldn't find anything for that. Try another topic!"}
        
        # Step 2: Auto-pick best match using difflib
        best_match = get_close_matches(query, suggestions, n=1, cutoff=0.6)
        if best_match:
            title = best_match[0]
        else:
            title = suggestions[0]  # fallback to first suggestion
        
        # Step 3: Get page summary
        page_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        page_req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        page_res = urllib.request.urlopen(page_req, timeout=10).read().decode()
        page_data = json.loads(page_res)
        
        if "extract" in page_data:
            extract = page_data["extract"][:2000]
            title = page_data.get("title", title)
            return {"success": True, "title": title, "extract": extract}
        
        return {"success": False, "error": "Page found but no summary available."}
    
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return {"success": False, "error": "Wikipedia blocked the request. Try again in a moment."}
        elif e.code == 404:
            return {"success": False, "error": "Page not found. Try another topic."}
        else:
            return {"success": False, "error": f"HTTP Error {e.code}"}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)[:100]}"}

async def wiki(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = " ".join(context.args).strip()
        if not query:
            await update.message.reply_text("Usage: /wiki <topic>")
            return

        status_msg = await update.message.reply_text("🔍 Searching Wikipedia...")
        result = await asyncio.to_thread(fetch_wiki_summary, query)
        
        if result["success"]:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=status_msg.message_id, 
                text=f"📚 {result['title']}\n\n{result['extract']}"
            )
        else:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id, 
                message_id=status_msg.message_id, 
                text=result["error"]
            )
    except Exception as e:
        logging.error(f"Wiki command error: {e}")
        await update.message.reply_text("Sorry, I couldn't search Wikipedia right now. Please try again later.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle free-form text messages with intelligent conversation"""
    try:
        user_message = update.message.text
        if not user_message or len(user_message.strip()) == 0:
            return
        
        user_message_lower = user_message.lower().strip()
        
        # Greeting detection - respond naturally
        greeting_patterns = r'\b(hi|hello|hey|yo|sup|good morning|good afternoon|good evening|howdy|greetings|welcome)\b'
        if re.search(greeting_patterns, user_message_lower):
            greetings = [
                "Hey! 👋 I'm Forall Agent. What's on your mind?",
                "Hello! 😊 I'm Forall Agent. How can I help you today?",
                "Hi there! 👋 Forall Agent here. Ready to chat or solve something interesting?",
                "Greetings! 🙂 I'm Forall Agent. What would you like to discuss?"
            ]
            await update.message.reply_text(greetings[hash(user_message) % len(greetings)])
            return
        
        # Detect question types for better responses
        question_markers = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you', 'could you', 'would you', 'tell me', 'explain', 'describe', 'what is', 'what are', 'how do']
        is_question = any(marker in user_message_lower for marker in question_markers)
        
        # Prepare enhanced prompt based on message type
        if is_question:
            prompt = f"Question: {user_message}\n\nThink through this step-by-step. Provide a thoughtful, comprehensive answer."
            tokens = 450
            temp = 0.6
        else:
            prompt = f"User said: {user_message}\n\nRespond conversationally and thoughtfully to what they said."
            tokens = 350
            temp = 0.7
        
        status_msg = await update.message.reply_text("💭 Thinking...")
        response = await asyncio.to_thread(generate_text, prompt, tokens, temp)
        
        # Validate response before sending
        if response and len(response) > 0:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=status_msg.message_id,
                text=response
            )
        else:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=status_msg.message_id,
                text="I had trouble processing that. Can you try again?"
            )
    
    except telegram.error.BadRequest as e:
        logging.error(f"Bad Request: {e}")
        try:
            await update.message.reply_text("Sorry, I encountered an error. Please try again.")
        except:
            pass
    except Exception as e:
        logging.error(f"Message handling error: {e}")
        try:
            await update.message.reply_text("Something went wrong. Please try again later.")
        except:
            pass

if __name__ == '__main__':
    load_model()
    
    try:
        application = ApplicationBuilder().token(TOKEN).build()

        application.add_handler(CommandHandler('start', start))
        application.add_handler(CommandHandler('help', help_command))
        application.add_handler(CommandHandler('ask', ask))
        application.add_handler(CommandHandler('story', story))
        application.add_handler(CommandHandler('joke', joke))
        application.add_handler(CommandHandler('motivate', motivate))
        application.add_handler(CommandHandler('wiki', wiki))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        print("Bot is running... Press Ctrl+C to stop")
        application.run_polling(allowed_updates=["message", "callback_query"])
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your TOKEN is correct and you have internet connection")
