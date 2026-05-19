from supabase_client import supabase
import base64
import io
import json
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Story Schema Cards",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Story Schema Cards")
st.caption(
    "Generate printable story schema cards for public-domain, "
    "open-source, or rights-cleared books."
)
# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------

def get_openai_client():
    api_key = None

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "OpenAI API key is missing. Add it to .streamlit/secrets.toml "
            "or Streamlit Cloud secrets."
        )
        st.stop()

    return OpenAI(api_key=api_key)


# ---------------------------------------------------------
# Prompt builder for schema card text
# ---------------------------------------------------------

def build_prompt(
    title,
    author,
    source_link,
    reader_level,
    deck_mode,
    spoiler_mode,
    card_focus,
    notes,
    chapter_context
):
    return f"""
You are an educational reading-support designer creating printable story schema cards.

The user wants schema cards for a public-domain, open-source, or rights-cleared book.

Book title: {title}
Author: {author}
Optional source link: {source_link if source_link else "Not provided"}
Reader level: {reader_level}
Deck mode: {deck_mode}
Spoiler mode: {spoiler_mode}
Card focus: {card_focus}
User notes: {notes if notes else "None"}

Optional chapter/context text:
{chapter_context[:4000] if chapter_context else "No chapter context provided."}

Purpose:
These cards support readers who can decode text but may struggle to visualize characters,
settings, scene actions, objects, symbols, or story structure.

Return JSON only.

Use this exact structure:

{{
  "book": {{
    "title": "",
    "author": "",
    "source_link": "",
    "public_domain_note": "",
    "recommended_card_count": 0,
    "complexity_level": "low | medium | high",
    "deck_mode": "",
    "spoiler_mode": ""
  }},
  "cards": [
    {{
      "card_type": "character | setting | scene | object | concept | group | episode",
      "label": "",
      "description": "",
      "why_it_matters": "",
      "chapter_reference": "",
      "image_search_query": "",
      "generic_image_fallback": "",
      "priority": "essential | useful | optional | deep",
      "spoiler_level": "low | medium | high",
      "sort_order": 1
    }}
  ]
}}

Rules:
- Do not quote the book.
- Do not copy publisher copy.
- Do not reference movie adaptations.
- Do not use copyrighted modern character likenesses.
- Prioritize reading comprehension.
- Include major characters.
- Include major settings.
- Include scene/action cards for moments that affect comprehension.
- Include symbolic or recurring objects when useful.
- Merge tiny background characters into group cards.
- For episodic works like The Odyssey, use episode cards.
- If spoiler mode is low, avoid major ending spoilers.
- Keep descriptions under 28 words.
- Keep why_it_matters under 20 words.
- image_search_query should help the user find public-domain or open-license images.
- generic_image_fallback should suggest a generic public-domain-friendly image if no book-specific image is available.
- If unsure about a detail, keep the card general rather than inventing plot facts.

Deck mode guidance:
- Sample: 8 essential cards
- Starter: 12 to 16 essential cards
- Standard: 20 to 35 cards using essential and useful items
- Full Schema: 40 to 70 cards with balanced coverage
- Deep Study: 70 to 110 cards with characters, settings, scenes, objects, concepts, and episodes

Card balance guidance:
- Characters: 25-40%
- Settings: 15-25%
- Scenes/actions: 25-35%
- Objects/symbols/concepts/groups: 10-25%

If the requested book has a complex plot, do not force an artificially tiny deck.
Create enough cards to support comprehension.
"""


# ---------------------------------------------------------
# Generate schema deck text
# ---------------------------------------------------------

def generate_schema_deck(
    title,
    author,
    source_link,
    reader_level,
    deck_mode,
    spoiler_mode,
    card_focus,
    notes,
    chapter_context
):
    client = get_openai_client()

    prompt = build_prompt(
        title=title,
        author=author,
        source_link=source_link,
        reader_level=reader_level,
        deck_mode=deck_mode,
        spoiler_mode=spoiler_mode,
        card_focus=card_focus,
        notes=notes,
        chapter_context=chapter_context
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        text={
            "format": {
                "type": "json_object"
            }
        }
    )

    text = response.output_text

    if not text:
        raise ValueError("No text returned from OpenAI.")

    return json.loads(text)


# ---------------------------------------------------------
# Balance cards by deck mode
# ---------------------------------------------------------

def balance_cards(cards, deck_mode):
    allowed_by_mode = {
        "Sample": ["essential"],
        "Starter": ["essential"],
        "Standard": ["essential", "useful"],
        "Full Schema": ["essential", "useful", "optional"],
        "Deep Study": ["essential", "useful", "optional", "deep"]
    }

    allowed = allowed_by_mode.get(deck_mode, ["essential", "useful"])

    filtered = [
        card for card in cards
        if card.get("priority") in allowed
    ]

    type_order = {
        "character": 1,
        "setting": 2,
        "scene": 3,
        "object": 4,
        "concept": 5,
        "group": 6,
        "episode": 7
    }

    filtered.sort(
        key=lambda card: (
            type_order.get(card.get("card_type"), 99),
            card.get("sort_order", 999)
        )
    )

    if deck_mode == "Sample":
        return filtered[:8]

    if deck_mode == "Starter":
        return filtered[:16]

    return filtered


# ---------------------------------------------------------
# Safe image prompt generation
# ---------------------------------------------------------

def sanitize_text_for_image_prompt(text):
    if not text:
        return ""

    safe_text = str(text)

    replacements = {
        "Alice": "storybook reader",
        "girl": "storybook figure",
        "child": "storybook figure",
        "boy": "storybook figure",
        "man": "storybook figure",
        "woman": "storybook figure",
        "rabbit wearing a waistcoat": "white rabbit beside a pocket watch",
        "wearing a waistcoat": "beside a pocket watch",
        "falls": "travels",
        "falling": "traveling",
        "down the rabbit hole": "near a round garden tunnel",
        "kill": "conflict",
        "killing": "conflict",
        "murder": "conflict",
        "execution": "royal command",
        "execute": "royal command",
        "blood": "red color",
        "behead": "royal command",
        "beheading": "royal command",
        "heads": "crowns",
        "weapon": "object",
        "knife": "object",
        "gun": "object",
        "violence": "conflict",
        "violent": "dramatic",
        "death": "serious moment",
        "dead": "still",
        "corpse": "figure",
        "hanging": "suspended object",
        "mad": "whimsical",
        "insane": "whimsical",
        "crazy": "whimsical",
        "attack": "conflict",
        "attacks": "conflict",
        "punishment": "rule",
        "threat": "dramatic command",
        "threatens": "commands",
        "screaming": "speaking",
        "angry": "stern"
    }

    for old, new in replacements.items():
        safe_text = safe_text.replace(old, new)
        safe_text = safe_text.replace(old.title(), new.title())
        safe_text = safe_text.replace(old.upper(), new.upper())

    return safe_text


def get_symbolic_subject(card):
    label = str(card.get("label", "")).lower()
    card_type = str(card.get("card_type", "")).lower()

    # Alice in Wonderland-specific safe symbols.
    if "white rabbit" in label:
        return "a white rabbit beside a pocket watch on a plain white background"

    if "rabbit hole" in label:
        return "a round garden tunnel entrance under a tree on a plain white background"

    if "alice" in label:
        return "an open storybook with a small blue ribbon bookmark on a plain white background"

    if "cheshire" in label or "cat" in label:
        return "a friendly striped cat sitting on a tree branch on a plain white background"

    if "queen" in label or "hearts" in label:
        return "a red heart playing card, a small crown, and red roses on a plain white background"

    if "tea" in label or "party" in label:
        return "a tea cup, teapot, and small table setting on a plain white background"

    if "bottle" in label or "drink" in label:
        return "a small glass bottle on a simple table on a plain white background"

    if "key" in label:
        return "a small golden key on a plain white background"

    if "door" in label:
        return "a small wooden door with a round handle on a plain white background"

    if "croquet" in label:
        return "a garden lawn with simple croquet hoops on a plain white background"

    if "caterpillar" in label:
        return "a blue caterpillar on a green leaf on a plain white background"

    if "cards" in label or "soldiers" in label:
        return "playing cards arranged in a neat row on a plain white background"

    if "duchess" in label:
        return "a small crown and teacup on a plain white background"

    if "mock turtle" in label:
        return "a friendly turtle near a shoreline on a plain white background"

    if "gryphon" in label:
        return "a simple mythical bird-lion creature silhouette on a plain white background"

    if "trial" in label:
        return "a wooden table, paper scroll, and small gavel on a plain white background"

    # Generic safe fallbacks by type.
    if card_type == "character":
        return "a simple symbolic object representing a story character on a plain white background"

    if card_type == "setting":
        return "a simple landscape symbol representing a story setting on a plain white background"

    if card_type == "scene":
        return "a simple arrangement of symbolic story objects on a plain white background"

    if card_type == "object":
        return "a single simple story object centered on a plain white background"

    if card_type == "concept":
        return "a simple educational icon representing an idea on a plain white background"

    if card_type == "episode":
        return "a simple map marker and open book icon on a plain white background"

    if card_type == "group":
        return "a small group of simple symbolic shapes on a plain white background"

    return "a simple open book icon on a plain white background"


def build_safe_image_prompt(card):
    subject = get_symbolic_subject(card)
    subject = sanitize_text_for_image_prompt(subject)

    return f"""
Create a simple printable educational flashcard image.

Subject:
{subject}

Style:
simple clean children's educational illustration, soft shapes, clear object, centered composition.

Strict rules:
- no text
- no letters
- no labels
- no logos
- no people
- no realistic humans
- no children
- no injury
- no danger
- no falling
- no violence
- no weapons
- no frightening imagery
- no movie adaptation references
- no publisher artwork imitation
- no named artist style
- plain white background
"""


def build_contextual_safe_image_prompt(card, chapter_context=""):
    client = get_openai_client()

    label = card.get("label", "")
    card_type = card.get("card_type", "")
    description = card.get("description", "")
    why_it_matters = card.get("why_it_matters", "")
    fallback = card.get("generic_image_fallback", "")
    existing_safe_subject = card.get("safe_image_subject", "")

    prompt = f"""
You are creating a safe image-generation prompt for a printable educational literature flashcard.

The final image will be used on a reading-support card for a public-domain book.

Card:
- Type: {card_type}
- Label: {label}
- Description: {description}
- Why it matters: {why_it_matters}
- Existing safe subject: {existing_safe_subject}
- Generic fallback: {fallback}

Relevant public-domain book context:
{chapter_context[:2500] if chapter_context else "No context provided."}

Task:
Create ONE safe image prompt that visually represents the card's story schema concept.

Important:
The image prompt must be customized to the book context, but it must avoid unsafe or easily-blocked wording.

Rules:
- Do not include the words: falling, falls, fall, danger, dangerous, injury, injured, violence, weapon, blood, kill, death, dead, execution, beheading.
- Do not depict a child in danger.
- Avoid realistic humans.
- Prefer symbolic objects, settings, props, silhouettes, landscapes, or visual metaphors.
- No text, letters, labels, logos, or captions inside the image.
- No movie adaptation references.
- No publisher artwork imitation.
- No named artist styles.
- Keep it classroom-friendly.
- Use a plain or simple background.
- Make it suitable for a printable flashcard.

Return JSON only:
{{
  "safe_image_prompt": ""
}}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        text={
            "format": {
                "type": "json_object"
            }
        }
    )

    data = json.loads(response.output_text)
    safe_prompt = data.get("safe_image_prompt", "").strip()

    if not safe_prompt:
        safe_prompt = build_safe_image_prompt(card)

    return safe_prompt


# ---------------------------------------------------------
# Placeholder image if image generation fails
# ---------------------------------------------------------

def wrap_text_for_placeholder(draw, text, font, max_width):
    words = str(text).split()
    lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            line = test_line
        else:
            if line:
                lines.append(line)
            line = word

    if line:
        lines.append(line)

    return lines


def create_placeholder_image(card):
    label = card.get("label", "Story Card")
    card_type = card.get("card_type", "schema")

    img = Image.new("RGB", (1024, 1024), color="white")
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()

    draw.rectangle(
        [(80, 80), (944, 944)],
        outline="black",
        width=6
    )

    draw.text(
        (120, 160),
        f"{str(card_type).upper()} CARD",
        fill="black",
        font=font
    )

    y = 260
    label_lines = wrap_text_for_placeholder(draw, str(label)[:80], font, 760)

    for line in label_lines[:4]:
        draw.text((120, y), line, fill="black", font=font)
        y += 45

    draw.text(
        (120, 520),
        "Image placeholder",
        fill="black",
        font=font
    )

    draw.text(
        (120, 580),
        "Review image prompt later.",
        fill="black",
        font=font
    )

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer.getvalue()


# ---------------------------------------------------------
# Generate image for card with contextual retry + fallback
# ---------------------------------------------------------

def generate_card_image(card, chapter_context=""):
    client = get_openai_client()

    try:
        safe_prompt = build_contextual_safe_image_prompt(
            card=card,
            chapter_context=chapter_context
        )
    except Exception:
        safe_prompt = build_safe_image_prompt(card)

    try:
        result = client.images.generate(
            model="gpt-image-1-mini",
            prompt=safe_prompt,
            size="1024x1024",
            quality="low",
            moderation="low",
            n=1
        )

        image_base64 = result.data[0].b64_json
        return base64.b64decode(image_base64)

    except Exception as first_error:
        label = str(card.get("label", "")).lower()
        card_type = str(card.get("card_type", "")).lower()

        # Super-safe custom fallbacks for common Alice Chapter 1 concepts.
        if "white rabbit" in label:
            fallback_subject = "a white rabbit beside a pocket watch, plain white background"
        elif "rabbit hole" in label or "down the rabbit" in label:
            fallback_subject = "a round garden tunnel entrance under a tree, plain white background"
        elif "alice" in label:
            fallback_subject = "an open storybook with a blue ribbon bookmark, plain white background"
        elif "chapter" in label or "scene" in card_type:
            fallback_subject = "a whimsical tunnel with floating books, jars, and maps, plain light background"
        else:
            fallback_subject = "an open book, bookmark, and small star, plain white background"

        fallback_prompt = f"""
Create a simple printable educational flashcard image.

Subject:
{fallback_subject}

Style:
simple clean children's educational illustration, centered composition, soft shapes.

Strict rules:
- no text
- no letters
- no logos
- no people
- no realistic humans
- no children
- no danger
- no falling
- no violence
- no weapons
- no frightening imagery
- plain white background
"""

        try:
            result = client.images.generate(
                model="gpt-image-1-mini",
                prompt=fallback_prompt,
                size="1024x1024",
                quality="low",
                moderation="low",
                n=1
            )

            image_base64 = result.data[0].b64_json
            return base64.b64decode(image_base64)

        except Exception as second_error:
            raise RuntimeError(
                f"Image generation failed twice. First: {first_error}. Second: {second_error}"
            )


# ---------------------------------------------------------
# Generate images for all cards
# ---------------------------------------------------------

def add_images_to_deck(deck, image_limit, chapter_context=""):
    cards = deck.get("cards", [])

    if not cards:
        return deck

    progress_bar = st.progress(0)
    status_text = st.empty()

    cards_to_generate = min(len(cards), image_limit)

    for index, card in enumerate(cards):
        progress = int((index + 1) / max(len(cards), 1) * 100)

        if index >= image_limit:
            card["generated_image_bytes"] = create_placeholder_image(card)
            card["image_status"] = "placeholder_used_due_to_image_limit"
            progress_bar.progress(progress)
            continue

        status_text.write(
            f"Generating image {index + 1} of {cards_to_generate}: {card.get('label')}"
        )

        try:
            image_bytes = generate_card_image(
                card=card,
                chapter_context=chapter_context
            )
            card["generated_image_bytes"] = image_bytes
            card["image_status"] = "generated"

        except Exception as e:
            card["generated_image_bytes"] = create_placeholder_image(card)
            card["image_status"] = f"placeholder_used_after_generation_failure: {e}"

        progress_bar.progress(progress)

    status_text.write("Image generation complete.")
    return deck


# ---------------------------------------------------------
# Openverse search fallback
# ---------------------------------------------------------

def get_openverse_search_url(query):
    if not query:
        query = "public domain book illustration"

    encoded = requests.utils.quote(query)
    return f"https://openverse.org/search/image?q={encoded}"


# ---------------------------------------------------------
# Save generated deck to Supabase
# ---------------------------------------------------------

def save_deck_to_supabase(deck, user_email, reader_level, deck_mode, spoiler_mode):
    book = deck.get("book", {})
    cards = deck.get("cards", [])

    book_result = supabase.table("books").insert({
        "title": book.get("title", ""),
        "author": book.get("author", ""),
        "source_text": "",
        "age_group": reader_level,
        "reading_level": reader_level
    }).execute()

    book_id = book_result.data[0]["id"]

    deck_result = supabase.table("decks").insert({
        "book_id": book_id,
        "deck_title": f"{book.get('title', 'Untitled')} Story Schema Deck",
        "deck_type": deck_mode,
        "status": "generated",
        "canva_status": "not_started"
    }).execute()

    deck_id = deck_result.data[0]["id"]

    card_rows = []

    for index, card in enumerate(cards):
        card_rows.append({
            "deck_id": deck_id,
            "card_order": index + 1,
            "front_text": card.get("label", ""),
            "back_text": card.get("description", ""),
            "category": card.get("card_type", ""),
            "difficulty": card.get("priority", ""),
            "image_prompt": card.get("image_search_query", ""),
            "image_status": card.get("image_status", "not_generated")
        })

    card_result = supabase.table("cards").insert(card_rows).execute()

    for index, saved_card in enumerate(card_result.data):
        cards[index]["supabase_card_id"] = saved_card["id"]

    return deck_id
def upload_card_image_to_supabase(deck_id, card_index, image_bytes):
    if not image_bytes:
        return None, None

    file_path = f"{deck_id}/card-{card_index + 1:03}.png"

    supabase.storage.from_("card-images").upload(
        path=file_path,
        file=image_bytes,
        file_options={
            "content-type": "image/png",
            "upsert": "true"
        }
    )

    public_url = supabase.storage.from_("card-images").get_public_url(file_path)

    return file_path, public_url


def save_card_images_to_supabase(deck_id, deck):
    cards = deck.get("cards", [])

    for index, card in enumerate(cards):
        image_bytes = card.get("generated_image_bytes")
        card_id = card.get("supabase_card_id")

        if not image_bytes or not card_id:
            continue

        image_path, image_url = upload_card_image_to_supabase(
            deck_id=deck_id,
            card_index=index,
            image_bytes=image_bytes
        )

        card["image_storage_path"] = image_path
        card["image_public_url"] = image_url

        supabase.table("cards").update({
            "image_storage_path": image_path,
            "image_public_url": image_url,
            "image_status": card.get("image_status", "generated")
        }).eq("id", card_id).execute()

    return deck


# ---------------------------------------------------------
# Export table
# ---------------------------------------------------------

def create_export_rows(deck, user_email, source_link, reader_level, deck_mode, spoiler_mode):
    rows = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    book = deck.get("book", {})
    cards = deck.get("cards", [])

    for card in cards:
        image_query = card.get("image_search_query", "")

        rows.append({
            "timestamp": timestamp,
            "user_email": user_email,
            "book_title": book.get("title"),
            "author": book.get("author"),
            "source_link": source_link,
            "reader_level": reader_level,
            "deck_mode": deck_mode,
            "spoiler_mode": spoiler_mode,
            "card_type": card.get("card_type"),
            "label": card.get("label"),
            "description": card.get("description"),
            "why_it_matters": card.get("why_it_matters"),
            "chapter_reference": card.get("chapter_reference"),
            "image_search_query": image_query,
            "generic_image_fallback": card.get("generic_image_fallback"),
            "openverse_search_url": get_openverse_search_url(image_query),
"image_public_url": card.get("image_public_url", ""),
"image_storage_path": card.get("image_storage_path", ""),
"generated_image_status": card.get("image_status", "not_generated"),
            "priority": card.get("priority"),
            "spoiler_level": card.get("spoiler_level"),
            "sort_order": card.get("sort_order"),
            "image_source_url": "",
            "image_creator": "",
            "image_license": "review_needed",
            "attribution_required": "",
            "attribution_text": "",
            "license_verified": "no",
            "canva_status": "not_started",
            "wix_shop_status": "not_started",
            "product_status": "raw_generation"
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Printable text
# ---------------------------------------------------------

def create_printable_text(deck):
    cards = deck.get("cards", [])
    book = deck.get("book", {})

    printable_text = ""
    printable_text += "STORY SCHEMA CARDS\n"
    printable_text += f"{book.get('title', '')}\n"
    printable_text += f"{book.get('author', '')}\n"
    printable_text += "\n"

    for card in cards:
        printable_text += "━━━━━━━━━━━━━━━━━━━━\n"
        printable_text += f"{card.get('card_type', '').upper()} CARD\n"
        printable_text += "━━━━━━━━━━━━━━━━━━━━\n\n"
        printable_text += f"{card.get('label', '')}\n\n"
        printable_text += f"{card.get('description', '')}\n\n"
        printable_text += "WHY IT MATTERS:\n"
        printable_text += f"{card.get('why_it_matters', '')}\n\n"

        if card.get("chapter_reference"):
            printable_text += "REFERENCE:\n"
            printable_text += f"{card.get('chapter_reference')}\n\n"

        printable_text += "CUT HERE ✂\n\n"

    return printable_text


# ---------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------

def draw_wrapped_text(
    pdf,
    text,
    x,
    y,
    max_width,
    font_name,
    font_size,
    line_height,
    max_lines=None
):
    if not text:
        return y

    pdf.setFont(font_name, font_size)

    words = str(text).split()
    line = ""
    lines_drawn = 0

    for word in words:
        test_line = f"{line} {word}".strip()
        width = pdf.stringWidth(test_line, font_name, font_size)

        if width <= max_width:
            line = test_line
        else:
            if max_lines is not None and lines_drawn >= max_lines:
                return y

            pdf.drawString(x, y, line)
            y -= line_height
            lines_drawn += 1
            line = word

    if line:
        if max_lines is None or lines_drawn < max_lines:
            pdf.drawString(x, y, line)
            y -= line_height

    return y


# ---------------------------------------------------------
# PDF builder
# ---------------------------------------------------------

def create_pdf(deck):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    page_width, page_height = letter

    margin_x = 24
    margin_y = 24
    gap_x = 18
    gap_y = 18

    card_width = (page_width - (2 * margin_x) - gap_x) / 2
    card_height = (page_height - (2 * margin_y) - gap_y) / 2

    cards = deck.get("cards", [])
    book = deck.get("book", {})

    for index, card in enumerate(cards):
        position = index % 4

        if index > 0 and position == 0:
            pdf.showPage()

        col = position % 2
        row = position // 2

        x = margin_x + col * (card_width + gap_x)
        y = page_height - margin_y - (row + 1) * card_height - row * gap_y

        pdf.setStrokeColor(colors.black)
        pdf.setDash(4, 3)
        pdf.rect(x, y, card_width, card_height)
        pdf.setDash()

        padding = 12
        inner_x = x + padding
        inner_y = y + card_height - padding
        inner_width = card_width - (2 * padding)

        pdf.setFillColor(colors.black)

        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(inner_x, inner_y - 8, str(card.get("card_type", "")).upper())

        pdf.setFont("Helvetica-Bold", 15)
        label = str(card.get("label", ""))
        pdf.drawString(inner_x, inner_y - 30, label[:34])

        text_y = inner_y - 58

        text_y = draw_wrapped_text(
            pdf=pdf,
            text=card.get("description", ""),
            x=inner_x,
            y=text_y,
            max_width=inner_width,
            font_name="Helvetica",
            font_size=10,
            line_height=13,
            max_lines=6
        )

        text_y -= 8

        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(inner_x, text_y, "Why it matters:")
        text_y -= 12

        text_y = draw_wrapped_text(
            pdf=pdf,
            text=card.get("why_it_matters", ""),
            x=inner_x,
            y=text_y,
            max_width=inner_width,
            font_name="Helvetica",
            font_size=9,
            line_height=11,
            max_lines=5
        )

        chapter = card.get("chapter_reference")
        if chapter:
            pdf.setFont("Helvetica-Oblique", 7)
            pdf.drawString(inner_x, y + 26, str(chapter)[:70])

        pdf.setFont("Helvetica", 6)
        pdf.setFillColor(colors.grey)
        footer = f"{book.get('title', '')} | Story Schema Cards"
        pdf.drawRightString(x + card_width - padding, y + 10, footer[:70])

    pdf.save()
    buffer.seek(0)

    return buffer.getvalue()


# ---------------------------------------------------------
# JSON safe helper
# ---------------------------------------------------------

def make_json_safe_deck(deck):
    return json.loads(json.dumps(deck, default=lambda value: None))


# ---------------------------------------------------------
# Sidebar input form
# ---------------------------------------------------------

st.sidebar.header("Generate a schema deck")

selected_title = st.sidebar.text_input(
    "Book title",
    placeholder="Example: Alice's Adventures in Wonderland"
)

selected_author = st.sidebar.text_input(
    "Author",
    placeholder="Example: Lewis Carroll"
)

source_link = st.sidebar.text_input(
    "Optional public-domain/source link",
    placeholder="Project Gutenberg, Internet Archive, Open Library, etc."
)

public_domain_confirmed = st.sidebar.checkbox(
    "I understand this should be a public-domain, open-source, or rights-cleared title."
)

reader_level = st.sidebar.selectbox(
    "Reader level",
    [
        "early elementary",
        "upper elementary",
        "middle school",
        "high school",
        "adult reader"
    ],
    index=1
)

deck_mode = st.sidebar.selectbox(
    "Deck mode",
    [
        "Sample",
        "Starter",
        "Standard",
        "Full Schema",
        "Deep Study"
    ],
    index=0
)

spoiler_mode = st.sidebar.selectbox(
    "Spoiler mode",
    [
        "low spoilers",
        "chapter-safe",
        "full study"
    ],
    index=0
)

card_focus = st.sidebar.multiselect(
    "Card focus",
    [
        "characters",
        "settings",
        "scenes",
        "objects",
        "concepts",
        "episodes",
        "balanced deck"
    ],
    default=["balanced deck"]
)

notes = st.sidebar.text_area(
    "Optional notes",
    placeholder="Example: Focus on confusing scenes, Victorian settings, or mythological figures."
)
chapter_context = st.sidebar.text_area(
    "Optional chapter/context text",
    placeholder=(
        "Paste a short public-domain excerpt or summary here. "
        "Example: Alice sees the White Rabbit, follows it across a field, "
        "and enters a strange tunnel with shelves, jars, maps, and books."
    ),
    height=180
)
generate_images = st.sidebar.checkbox(
    "Generate AI images for cards",
    value=True
)

image_limit = st.sidebar.number_input(
    "Maximum AI-generated cards (recommended: 4–20)",
    min_value=1,
    max_value=120,
    value=4,
    step=1
)
user_email = st.sidebar.text_input(
    "Tester email",
    placeholder="you@example.com"
)

generate_button = st.sidebar.button("Generate schema cards + PDF")


# ---------------------------------------------------------
# Main instructions
# ---------------------------------------------------------

with st.expander("What this tool does", expanded=True):
    st.write(
        """
        This tool generates story schema cards and a printable PDF.
        
        It creates card content, image search ideas, Openverse search links,
        a CSV export for Canva/Google Sheets, a JSON backup, and a printable text version.
        """
    )

with st.expander("Important source and license note"):
    st.write(
        """
        This is a working beta tool. Before selling polished decks, verify:
        
        - public-domain status of the book/translation/edition
        - image/license rules
        - attribution requirements
        - marketplace rules
        - privacy/payment/legal requirements
        """
    )


# ---------------------------------------------------------
# Generate deck
# ---------------------------------------------------------

if generate_button:
    if not selected_title.strip():
        st.error("Please enter a book title.")
        st.stop()

    if not selected_author.strip():
        st.error("Please enter an author.")
        st.stop()

    if not public_domain_confirmed:
        st.error(
            "Please confirm that this is intended for a public-domain, open-source, or rights-cleared title."
        )
        st.stop()

    try:
        with st.spinner("Generating schema card text..."):
            raw_deck = generate_schema_deck(
                title=selected_title,
                author=selected_author,
                source_link=source_link,
                reader_level=reader_level,
                deck_mode=deck_mode,
                spoiler_mode=spoiler_mode,
                card_focus=", ".join(card_focus),
                notes=notes,
                chapter_context=chapter_context
            )

            balanced_cards = balance_cards(
                raw_deck.get("cards", []),
                deck_mode
            )

            deck = {
                **raw_deck,
                "cards": balanced_cards
            }

            if "book" not in deck:
                deck["book"] = {}

            deck["book"]["title"] = deck["book"].get("title") or selected_title
            deck["book"]["author"] = deck["book"].get("author") or selected_author
            deck["book"]["source_link"] = source_link
            deck["book"]["deck_mode"] = deck_mode
            deck["book"]["spoiler_mode"] = spoiler_mode

        if generate_images:
            with st.spinner("Generating card images..."):
                deck = add_images_to_deck(
                    deck,
                    image_limit=image_limit,
                    chapter_context=chapter_context
                )
        else:
            for card in deck.get("cards", []):
                card["generated_image_bytes"] = create_placeholder_image(card)
                card["image_status"] = "placeholder_used_images_not_requested"

        with st.spinner("Preparing downloads..."):
            supabase_deck_id = save_deck_to_supabase(
                deck=deck,
                user_email=user_email,
                reader_level=reader_level,
                deck_mode=deck_mode,
                spoiler_mode=spoiler_mode
            )

            deck["supabase_deck_id"] = supabase_deck_id

            deck = save_card_images_to_supabase(
                deck_id=supabase_deck_id,
                deck=deck
            )

            export_df = create_export_rows(
                deck=deck,
                user_email=user_email,
                source_link=source_link,
                reader_level=reader_level,
                deck_mode=deck_mode,
                spoiler_mode=spoiler_mode
            )

            printable_text = create_printable_text(deck)
            pdf_bytes = create_pdf(deck)

        st.session_state["deck"] = deck
        st.session_state["export_df"] = export_df
        st.session_state["printable_text"] = printable_text
        st.session_state["pdf_bytes"] = pdf_bytes
        st.session_state["supabase_deck_id"] = supabase_deck_id

        st.success("Deck generated. PDF is ready to download.")

    except Exception as e:
        st.error(f"Something broke: {e}")


# ---------------------------------------------------------
# Display generated deck
# ---------------------------------------------------------

if "deck" in st.session_state:
    deck = st.session_state["deck"]
    export_df = st.session_state["export_df"]
    printable_text = st.session_state["printable_text"]
    pdf_bytes = st.session_state["pdf_bytes"]

    book = deck.get("book", {})
    cards = deck.get("cards", [])

    st.header(f"{book.get('title')} Schema Cards")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Cards", len(cards))
    col2.metric("Deck Mode", book.get("deck_mode", deck_mode))
    col3.metric("Spoiler Mode", book.get("spoiler_mode", spoiler_mode))
    col4.metric("Complexity", book.get("complexity_level", "unknown"))

    st.write(f"**Author:** {book.get('author')}")
    st.write(f"**Source link:** {book.get('source_link') or 'Not provided'}")
    st.write(
        f"**Public-domain note:** {book.get('public_domain_note', 'Review before commercial use.')}"
    )

    safe_title = (
        book.get("title", "schema-cards")
        .replace(" ", "-")
        .replace("/", "-")
        .replace(":", "")
        .replace("'", "")
        .lower()
    )

    st.subheader("Download Files")

    st.download_button(
        label="Download Print-Ready PDF",
        data=pdf_bytes,
        file_name=f"{safe_title}-schema-cards.pdf",
        mime="application/pdf"
    )

    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV for Canva / Google Sheets",
        data=csv_data,
        file_name=f"{safe_title}-schema-cards.csv",
        mime="text/csv"
    )

    json_safe_deck = make_json_safe_deck(deck)
    json_data = json.dumps(json_safe_deck, indent=2).encode("utf-8")

    st.download_button(
        label="Download JSON developer backup",
        data=json_data,
        file_name=f"{safe_title}-schema-cards.json",
        mime="application/json"
    )

    text_data = printable_text.encode("utf-8")

    st.download_button(
        label="Download Printable Text",
        data=text_data,
        file_name=f"{safe_title}-printable-text.txt",
        mime="text/plain"
    )

    st.subheader("Card Preview")

    for i, card in enumerate(cards, start=1):
        with st.container(border=True):
            st.markdown(f"### {i}. {card.get('label')}")
            st.write(f"**Type:** {card.get('card_type')}")
            st.write(f"**Description:** {card.get('description')}")
            st.write(f"**Why it matters:** {card.get('why_it_matters')}")
            st.write(f"**Chapter/reference:** {card.get('chapter_reference')}")
            st.write(f"**Image search query:** {card.get('image_search_query')}")
            st.write(f"**Generic fallback:** {card.get('generic_image_fallback')}")
            image_bytes = card.get("generated_image_bytes")

            if image_bytes:
                st.image(image_bytes, width=260)

            st.link_button(
                "Search open-license images",
                get_openverse_search_url(card.get("image_search_query", ""))
            )

    st.subheader("Export Table")
    st.dataframe(export_df, use_container_width=True)

    st.subheader("Printable Text Version")

    st.text_area(
        "Copy/paste printable roll-paper version",
        printable_text,
        height=450
    )