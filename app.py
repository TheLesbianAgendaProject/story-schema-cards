import json
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI


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
    "A beta tool that turns public-domain, open-source, or rights-cleared books "
    "into printable visual schema cards for readers who need help picturing the story."
)


# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

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
# Generate deck with OpenAI
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
# Image search links
# ---------------------------------------------------------

def get_openverse_search_url(query):
    if not query:
        query = "public domain book illustration"
    encoded = requests.utils.quote(query)
    return f"https://openverse.org/search/image?q={encoded}"


# ---------------------------------------------------------
# Create CSV/CMS rows
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
            "priority": card.get("priority"),
            "spoiler_level": card.get("spoiler_level"),
            "sort_order": card.get("sort_order"),
            "image_source_url": "",
            "image_creator": "",
            "image_license": "",
            "attribution_required": "",
            "attribution_text": "",
            "license_verified": "no",
            "canva_status": "not_started",
            "wix_shop_status": "not_started",
            "product_status": "raw_generation"
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Printable roll-paper text
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

        printable_text += "IMAGE IDEA:\n"
        printable_text += f"{card.get('image_search_query', '')}\n\n"

        printable_text += "CUT HERE ✂\n\n"

    return printable_text


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

user_email = st.sidebar.text_input(
    "Tester email",
    placeholder="you@example.com"
)

generate_button = st.sidebar.button("Generate schema cards")


# ---------------------------------------------------------
# Main instructions
# ---------------------------------------------------------

with st.expander("What this tool does", expanded=True):
    st.write(
        """
        This tool generates schema cards for books you identify as public-domain,
        open-source, or rights-cleared.
        
        It creates card content, image search ideas, Openverse search links,
        a CSV export for Canva/Google Sheets, a JSON backup, and a printable text version.
        """
    )

with st.expander("Important source and license note"):
    st.write(
        """
        This is a working beta tool. Before selling polished decks, verify:
        
        - public-domain status of the book/translation/edition
        - image license
        - attribution requirements
        - marketplace rules
        - privacy/payment/legal requirements
        
        The tool helps draft the deck. You still review and approve the final product.
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

    with st.spinner("Generating schema cards..."):
        try:
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

            export_df = create_export_rows(
                deck=deck,
                user_email=user_email,
                source_link=source_link,
                reader_level=reader_level,
                deck_mode=deck_mode,
                spoiler_mode=spoiler_mode
            )

            printable_text = create_printable_text(deck)

            st.session_state["deck"] = deck
            st.session_state["export_df"] = export_df
            st.session_state["printable_text"] = printable_text

            st.success("Deck generated. Download your files below.")

        except Exception as e:
            st.error(f"Something broke: {e}")


# ---------------------------------------------------------
# Display generated deck
# ---------------------------------------------------------

if "deck" in st.session_state:
    deck = st.session_state["deck"]
    export_df = st.session_state["export_df"]
    printable_text = st.session_state["printable_text"]

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
    st.write(f"**Public-domain note:** {book.get('public_domain_note', 'Review before commercial use.')}")

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

            st.link_button(
                "Search open-license images",
                get_openverse_search_url(card.get("image_search_query", ""))
            )

    st.subheader("Export Table")

    st.dataframe(export_df, use_container_width=True)

    safe_title = book.get("title", "schema-cards").replace(" ", "-").replace("/", "-").lower()

    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV for Canva / Google Sheets",
        data=csv_data,
        file_name=f"{safe_title}-schema-cards.csv",
        mime="text/csv"
    )

    json_data = json.dumps(deck, indent=2).encode("utf-8")

    st.download_button(
        label="Download JSON developer backup",
        data=json_data,
        file_name=f"{safe_title}-schema-cards.json",
        mime="application/json"
    )

    st.subheader("Printable Text Version")

    st.text_area(
        "Copy/paste printable roll-paper version",
        printable_text,
        height=450
    )

    text_data = printable_text.encode("utf-8")

    st.download_button(
        label="Download Printable Text",
        data=text_data,
        file_name=f"{safe_title}-printable-text.txt",
        mime="text/plain"
    )