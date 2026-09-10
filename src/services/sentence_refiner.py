"""
Multilingual Sign-to-Speech Grammar & Sentence Reconstruction Service.
Converts telegraphic sign gloss sequences (e.g., ["STORE", "ME", "GO", "BUY", "MILK"])
into natural, grammatically correct spoken sentences in English or Arabic.

Supports:
1. Local Quantized Multilingual SLM (Qwen2.5-0.5B-Instruct via llama-cpp-python).
2. Ultra-Fast Zero-Latency Deterministic Heuristic Grammar Engines (English & Arabic fallbacks).
3. Non-blocking asynchronous execution for Flet 0.86 event loops.
"""
import asyncio
import logging
from pathlib import Path
import re
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SLM_PATH = str(BASE_DIR / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf")


class SentenceRefinerService:
    def __init__(self, model_path: Optional[str] = None, preload: bool = True):
        self.model_path = model_path or DEFAULT_SLM_PATH
        self.llm = None
        self.is_llm_ready = False
        self._slm_init_attempted = False
        self._slm_init_lock = threading.Lock()
        if preload:
            threading.Thread(target=self._init_slm, daemon=True).start()

    def _init_slm(self):
        """Attempts to load local quantized SLM model if present on disk."""
        if self._slm_init_attempted:
            return
        target = Path(self.model_path)
        # A BITS/HTTP download exposes a growing destination file.  Never try to
        # mmap a partial GGUF: it wastes several seconds and would permanently
        # disable the model for this service instance after the failed load.
        min_qwen_bytes = 400_000_000
        if not target.exists() or (
            target.name.startswith("qwen2.5-0.5b")
            and target.stat().st_size < min_qwen_bytes
        ):
            logger.info(
                f"[SentenceRefiner] Qwen weights are not ready at '{target}'. "
                f"Operating in deterministic high-speed grammar mode."
            )
            return

        self._slm_init_attempted = True

        try:
            from llama_cpp import Llama
            logger.info(f"[SentenceRefiner] Loading local SLM weights from {target}...")
            self.llm = Llama(
                model_path=str(target),
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
            self.is_llm_ready = True
            logger.info("[SentenceRefiner] Local Qwen2.5 multilingual engine initialized successfully!")
        except Exception as e:
            logger.warning(f"[SentenceRefiner] Could not initialize llama-cpp engine: {e}")
            self.llm = None
            self.is_llm_ready = False

    async def refine_glosses_async(self, glosses: List[str], target_lang: str = "en") -> str:
        """Non-blocking async wrapper that executes SLM/heuristic refinement in a background thread."""
        return await asyncio.to_thread(self.refine_glosses, glosses, target_lang)

    def refine_glosses(self, glosses: List[str], target_lang: str = "en") -> str:
        """
        Converts a list of sign glosses into a grammatically polished sentence in target_lang ('en' or 'ar').
        """
        if not glosses:
            return ""

        cleaned_glosses = [g.strip() for g in glosses if g and g.strip()]
        if not cleaned_glosses:
            return ""

        # Normalize target_lang
        lang = "ar" if target_lang.lower().startswith("ar") else "en"

        # 1. Try local Qwen multilingual SLM if loaded
        # Loading a 0.5B model can take seconds; this method is called through
        # refine_glosses_async, so initialization never blocks camera capture or
        # the Flet UI event loop.
        if not self._slm_init_attempted:
            with self._slm_init_lock:
                self._init_slm()
        if self.is_llm_ready and self.llm is not None:
            try:
                refined = self._infer_slm(cleaned_glosses, target_lang=lang)
                if refined:
                    return refined
            except Exception as e:
                logger.error(f"[SentenceRefiner] SLM generation error: {e}. Falling back to heuristics.")

        # 2. High-Speed Deterministic Grammar Rule Engine fallback
        if lang == "ar":
            return self._heuristic_refinement_arabic(cleaned_glosses)
        return self._heuristic_refinement_english(cleaned_glosses)

    def _infer_slm(self, glosses: List[str], target_lang: str = "en") -> Optional[str]:
        raw_gloss = " ".join(glosses).upper()

        language = "Arabic" if target_lang == "ar" else "English"
        if target_lang == "ar":
            prompt = (
                "<|im_start|>system\n"
                "You are an expert sign language interpreter.\n"
                "Translate the sign language gloss sequence into a natural, grammatically correct Arabic sentence.\n"
                "Return ONLY the final translated Arabic sentence with proper Arabic punctuation.\n"
                "<|im_end|>\n<|im_start|>user\n"
                f"Glosses: {raw_gloss}\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            prompt = (
                "<|im_start|>system\n"
                "You are an expert American Sign Language (ASL) interpreter.\n"
                "Translate the sign language gloss sequence into a natural, grammatically correct English sentence.\n"
                "Fix pronouns (me -> I), verbs, and prepositions.\n"
                "Return ONLY the final translated sentence.\n"
                "<|im_end|>\n<|im_start|>user\n"
                f"Glosses: {raw_gloss}\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )

        if self.llm is not None:
            try:
                self.llm.reset()
            except Exception:
                pass

        output = self.llm(
            prompt,
            max_tokens=48,
            temperature=0.0,
            stop=["<|im_end|>", "\n"],
        )
        if self.llm is not None:
            try:
                self.llm.reset()
            except Exception:
                pass
        text = output["choices"][0]["text"].strip()
        # Clean artifacts, tags, quotes
        text = re.sub(r"<\|im_(?:start|end)\|>", "", text).strip()
        text = text.strip('"\'')

        if not text or text.upper() == raw_gloss:
            return None

        # Filter out chatbot refusal / preamble artifacts
        lower = text.lower()
        if any(bad in lower for bad in ("sorry", "understand", "here are", "options", "rewrite", "sentence:")):
            return None

        # For Arabic, ensure output actually contains Arabic characters
        if target_lang == "ar":
            has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
            if not has_arabic:
                return None

        return text

    def _heuristic_refinement_english(self, glosses: List[str]) -> str:
        """
        Syntactic transformation rules for sign glosses to standard English:
        - Replaces objective pronouns in subject position (me -> I)
        - Injects prepositions and copula verbs
        - Formats question structures
        - Fixes punctuation and capitalization
        """
        words = [w.lower() for w in glosses]
        n = len(words)
        result = []

        i = 0
        while i < n:
            w = words[i]

            # Rule: "thank you" / "thanks"
            if w in ("thank you", "thanks"):
                result.append("Thank you")
                i += 1
                continue

            # Rule: "my name" -> "My name is"
            if w == "my" and (i + 1 < n and words[i + 1] == "name"):
                result.append("My name is")
                i += 2
                continue

            # Rule: Subject pronoun normalization
            if w == "me" and (i == 0 or words[i - 1] in ("and", "then")):
                w = "I"

            # Rule: Question phrases ("where bathroom" -> "Where is the bathroom?")
            if w == "where" and (i + 1 < n):
                target = words[i + 1]
                result.append(f"Where is the {target}")
                i += 2
                continue

            # Rule: "want" / "need" + verb ("want eat" -> "want to eat")
            if w in ("want", "need") and (i + 1 < n and words[i + 1] in ("go", "eat", "drink", "buy", "sleep", "see", "help")):
                result.append(f"{w} to")
                i += 1
                continue

            # Rule: "go" + destination ("go store" -> "go to the store")
            if w == "go" and (i + 1 < n and words[i + 1] in ("store", "hospital", "bathroom", "school", "doctor", "supermarket", "bank")):
                dest = words[i + 1]
                dest_map = {
                    "store": "the store",
                    "hospital": "the hospital",
                    "bathroom": "the bathroom",
                    "school": "school",
                    "doctor": "the doctor",
                    "supermarket": "the supermarket",
                    "bank": "the bank",
                }
                result.append(f"go to {dest_map.get(dest, dest)}")
                i += 2
                continue

            # Rule: "how you" -> "How are you"
            if w == "how" and (i + 1 < n and words[i + 1] == "you"):
                result.append("How are you")
                i += 2
                continue

            result.append(w)
            i += 1

        sentence = " ".join(result).strip()
        if not sentence:
            return ""

        # Normalize spaces
        sentence = re.sub(r"\s+", " ", sentence)

        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]

        # Capitalize standalone "i"
        sentence = re.sub(r"\bi\b", "I", sentence)

        # Append appropriate sentence terminator
        if sentence.lower().startswith(("where", "what", "who", "when", "why", "how", "can you", "is")):
            if not sentence.endswith("?"):
                sentence += "?"
        elif not sentence.endswith((".", "!", "?")):
            sentence += "."

        return sentence

    def _heuristic_refinement_arabic(self, glosses: List[str]) -> str:
        """
        Syntactic transformation rules for sign glosses to standard Arabic.
        """
        ar_dict = {
            "hello": "مرحباً",
            "hi": "أهلاً",
            "thank you": "شكراً لك",
            "thanks": "شكراً",
            "please": "من فضلك",
            "help": "مساعدة",
            "me": "أنا",
            "i": "أنا",
            "you": "أنت",
            "want": "أريد",
            "need": "أحتاج",
            "go": "الذهاب",
            "store": "المتجر",
            "water": "الماء",
            "food": "الطعام",
            "eat": "أن آكل",
            "drink": "أن أشرب",
            "where": "أين",
            "bathroom": "دورة المياه",
            "hospital": "المستشفى",
            "school": "المدرسة",
            "doctor": "الطبيب",
            "yes": "نعم",
            "no": "لا",
            "good": "جيد",
            "how": "كيف",
            "name": "الاسم",
            "my": "خاصتي",
        }

        words = [w.lower().strip() for w in glosses]

        # Specific compound phrases
        joined = " ".join(words)
        if "thank you" in joined or "thanks" in joined:
            if "help" in joined:
                return "شكراً لك على المساعدة."
            return "شكراً لك."
        if "where" in words and ("bathroom" in words or "toilet" in words):
            return "أين هي دورة المياه؟"
        if "how" in words and "you" in words:
            return "كيف حالك؟"
        if ("me" in words or "i" in words) and "want" in words and "water" in words:
            return "أنا أريد شرب الماء."
        if ("me" in words or "i" in words) and "go" in words and "store" in words:
            return "أنا ذاهب إلى المتجر."

        # Word-by-word Arabic dictionary mapping
        ar_words = []
        for w in words:
            ar_words.append(ar_dict.get(w, w))

        sentence = " ".join(ar_words).strip()
        if not sentence:
            return ""

        if not sentence.endswith((".", "؟", "!")):
            if any(q in sentence for q in ("أين", "كيف", "هل", "ماذا")):
                sentence += "؟"
            else:
                sentence += "."

        return sentence
