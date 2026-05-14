import base64
import io
import json
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


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
    "Generate printable story schema cards with images for public-domain, "
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
# Prompt builder
# ---------------------------------------------------------

def build_prompt(
    title,
    author,
    source_link,
    reader_level,
    deck_mode,
    spoiler_mode,
    card_focus,
    notes
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
      "image_prompt": "",
      "safe_image_subject": "",
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
- image_prompt should describe a simple generic educational illustration.
- safe_image_subject should be a very safe, neutral visual subject for image generation.
- image_search_query should help the user find public-domain or open-license images.
- generic_image_fallback should suggest a generic public-domain-friendly image if no book-specific image is available.
- If unsure about a detail, keep the card general rather than inventing plot facts.

Image prompt rules:
- simple classroom-friendly illustration
- no movie adaptation references
- no modern copyrighted character likenesses
- no named living artist styles
- no publisher artwork imitation
- no logos
- no text inside the image
- no violence
- no weapons
- no injury
- no frightening imagery
- clear single subject
- white or simple background
- suitable for a printable flashcard

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
# Generate schema deck
# ---------------------------------------------------------

def generate_schema_deck(
    title,
    author,
    source_link,
    reader_level,
    deck_mode,
    spoiler_mode,
    card_focus,
    notes
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
        notes=notes
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
# Safer image prompt builder
# ---------------------------------------------------------

def sanitize_text_for_image_prompt(text):
    if not text:
        return ""

    safe_text = str(text)

    replacements = {
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
        "falling": "floating",
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


def build_safe_image_prompt(card):
    card_type = card.get("card_type", "schema")
    label = card.get("label", "story element")

    safe_subject = (
        card.get("safe_image_subject")
        or card.get("generic_image_fallback")
        or card.get("image_search_query")
        or card.get("image_prompt")
        or label
    )

    safe_subject = sanitize_text_for_image_prompt(safe_subject)
    safe_label = sanitize_text_for_image_prompt(label)

    return f"""
Create a simple classroom-friendly illustration for a printable story schema flashcard.

Subject:
{safe_subject}

Card category:
{card_type}

Label concept:
{safe_label}

Visual requirements:
- simple educational illustration
- classic literature inspired
- no text in the image
- no letters
- no logos
- no violence
- no weapons
- no injury
- no frightening imagery
- no scary faces
- no modern movie adaptation references
- no publisher artwork imitation
- no named artist style
- no realistic child likeness
- no celebrity likeness
- clear single subject
- centered composition
- white or simple background
- suitable for elementary classroom printing
"""


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
# Generate image for card with retry + fallback
# ---------------------------------------------------------

def generate_card_image(card):
    client = get_openai_client()

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
        label = sanitize_text_for_image_prompt(card.get("label", "story element"))
        card_type = sanitize_text_for_image_prompt(card.get("card_type", "schema"))

        fallback_prompt = f"""
Create a simple neutral educational icon for a printable literature flashcard.

Subject:
A symbolic, classroom-safe visual for a {card_type} card.

Concept:
{label}

Visual requirements:
- no people if avoidable
- no text
- no letters
- no violence
- no scary imagery
- no copyrighted character likeness
- no movie references
- simple centered object or place
- white background
- elementary classroom friendly
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

def add_images_to_deck(deck, image_limit):
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
            image_bytes = generate_card_image(card)
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
            "image_prompt": card.get("image_prompt"),
            "safe_image_subject": card.get("safe_image_subject"),
            "image_search_query": image_query,
            "generic_image_fallback": card.get("generic_image_fallback"),
            "openverse_search_url": get_openverse_search_url(image_query),
            "generated_image_status": card.get("image_status", "not_generated"),
            "priority": card.get("priority"),
            "spoiler_level": card.get("spoiler_level"),
            "sort_order": card.get("sort_order"),
            "image_source_url": "",
            "image_creator": "",
            "image_license": "ai_generated_review_needed",
            "attribution_required": "",
            "attribution_text": "",
            "license_verified": "needs_review",
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
# Text wrapping helper for PDF
# ---------------------------------------------------------

def draw_wrapped_text(pdf, text, x, y, max_width, font_name, font_size, line_height, max_lines=None):
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

    # 4 cards per page, 2x2 grid.
    # These are "4x6-style" cards that fit letter paper.
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

        # Card border / cut line
        pdf.setStrokeColor(colors.black)
        pdf.setDash(4, 3)
        pdf.rect(x, y, card_width, card_height)
        pdf.setDash()

        padding = 12
        inner_x = x + padding
        inner_y = y + card_height - padding
        inner_width = card_width - (2 * padding)

        # Card type
        pdf.setFont("Helvetica-Bold", 8)
        pdf.setFillColor(colors.black)
        pdf.drawString(inner_x, inner_y - 8, str(card.get("card_type", "")).upper())

        # Label
        pdf.setFont("Helvetica-Bold", 15)
        label = str(card.get("label", ""))
        pdf.drawString(inner_x, inner_y - 30, label[:34])

        # Image area
        image_top = inner_y - 48
        image_height = 145
        image_width = inner_width
        image_x = inner_x
        image_y = image_top - image_height

        pdf.setStrokeColor(colors.lightgrey)
        pdf.rect(image_x, image_y, image_width, image_height)

        image_bytes = card.get("generated_image_bytes")

        if image_bytes:
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image.thumbnail((int(image_width * 2), int(image_height * 2)))

                image_buffer = io.BytesIO()
                image.save(image_buffer, format="PNG")
                image_buffer.seek(0)

                image_reader = ImageReader(image_buffer)

                pdf.drawImage(
                    image_reader,
                    image_x + 4,
                    image_y + 4,
                    width=image_width - 8,
                    height=image_height - 8,
                    preserveAspectRatio=True,
                    anchor="c",
                    mask="auto"
                )
            except Exception:
                pdf.setFont("Helvetica", 9)
                pdf.drawCentredString(
                    image_x + image_width / 2,
                    image_y + image_height / 2,
                    "Image unavailable"
                )
        else:
            pdf.setFont("Helvetica", 9)
            pdf.drawCentredString(
                image_x + image_width / 2,
                image_y + image_height / 2,
                "Image not generated"
            )

        # Description
        text_y = image_y - 18

        pdf.setFillColor(colors.black)
        text_y = draw_wrapped_text(
            pdf=pdf,
            text=card.get("description", ""),
            x=inner_x,
            y=text_y,
            max_width=inner_width,
            font_name="Helvetica",
            font_size=9,
            line_height=11,
            max_lines=4
        )

        # Why it matters
        text_y -= 4
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(inner_x, text_y, "Why it matters:")
        text_y -= 10

        text_y = draw_wrapped_text(
            pdf=pdf,
            text=card.get("why_it_matters", ""),
            x=inner_x,
            y=text_y,
            max_width=inner_width,
            font_name="Helvetica",
            font_size=8,
            line_height=10,
            max_lines=3
        )

        # Chapter/reference
        chapter = card.get("chapter_reference")
        if chapter:
            pdf.setFont("Helvetica-Oblique", 7)
            pdf.setFillColor(colors.black)
            pdf.drawString(inner_x, y + 10, str(chapter)[:60])

        # Book footer
        pdf.setFont("Helvetica", 6)
        pdf.setFillColor(colors.grey)
        footer = f"{book.get('title', '')} | Story Schema Cards"
        pdf.drawRightString(x + card_width - padding, y + 10, footer[:70])

    pdf.save()
    buffer.seek(0)

    return buffer.getvalue()


# ---------------------------------------------------------
# Make JSON safe for export
# ---------------------------------------------------------

def make_json_safe_deck(deck):
    safe_deck = json.loads(json.dumps(deck, default=lambda value: None))

    for card in safe_deck.get("cards", []):
        if "generated_image_bytes" in card:
            card["generated_image_bytes"] = "[removed from JSON export]"

    return safe_deck


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

generate_images = st.sidebar.checkbox(
    "Generate AI images for cards",
    value=True
)

image_limit = st.sidebar.number_input(
    "Max real AI images to generate",
    min_value=1,
    max_value=40,
    value=8,
    step=1,
    help="Cards beyond this limit get a placeholder image so the PDF still prints cleanly."
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
        This tool generates schema cards, optional AI images, and a printable PDF.
        
        The PDF is formatted as 4 flashcard-style cards per letter-size page.
        If an image fails, the app uses a printable placeholder so the deck still exports.
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
        
        AI-generated images should still be reviewed before commercial sale.
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
                notes=notes
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
                deck = add_images_to_deck(deck, image_limit=image_limit)
        else:
            for card in deck.get("cards", []):
                card["generated_image_bytes"] = create_placeholder_image(card)
                card["image_status"] = "placeholder_used_images_not_requested"

        with st.spinner("Building printable PDF..."):
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
            st.write(f"**Image status:** {card.get('image_status')}")

            image_bytes = card.get("generated_image_bytes")

            if image_bytes:
                st.image(image_bytes, width=260)
            else:
                st.write(f"**Image prompt:** {card.get('image_prompt')}")
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
