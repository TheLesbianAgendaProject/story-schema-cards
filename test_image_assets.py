import io
import unittest

from PIL import Image

from image_assets import (
    crop_image,
    choose_existing_asset,
    normalize_image,
    normalize_match_text,
    similarity_score,
)


class ImageAssetTests(unittest.TestCase):
    def test_similarity_rewards_relevant_terms(self):
        relevant = similarity_score("white rabbit pocket watch", "white rabbit and pocket watch")
        unrelated = similarity_score("white rabbit pocket watch", "stormy ocean lighthouse")
        self.assertGreater(relevant, unrelated)

    def test_placeholders_are_not_reused(self):
        card = {
            "label": "White Rabbit",
            "card_type": "character",
            "image_search_query": "white rabbit pocket watch",
        }
        candidates = [{
            "front_text": "White Rabbit",
            "category": "character",
            "image_prompt": "white rabbit pocket watch",
            "image_storage_path": "deck/card-001.png",
            "image_status": "placeholder_used_due_to_image_limit",
        }]
        self.assertIsNone(choose_existing_asset(card, candidates))

    def test_exact_card_identity_reuses_saved_image_across_decks(self):
        card = {
            "label": "Heathcliff’s Childhood Abuse",
            "card_type": "scene",
            "description": "Heathcliff is mistreated after Mr. Earnshaw dies.",
            "image_search_query": "young Heathcliff at Wuthering Heights",
        }
        candidates = [{
            "front_text": "Heathcliff's Childhood Abuse",
            "back_text": "Hindley mistreats Heathcliff after their father dies.",
            "category": "scene",
            "image_prompt": "Heathcliff mistreated at Wuthering Heights",
            "image_storage_path": "older-deck/card-010.jpg",
            "image_status": "generated",
        }]
        match = choose_existing_asset(card, candidates)
        self.assertIsNotNone(match)
        self.assertEqual(match["image_storage_path"], "older-deck/card-010.jpg")
        self.assertEqual(match["match_score"], 1.0)

    def test_related_scene_wording_can_reuse_a_saved_image(self):
        card = {
            "label": "Heathcliff Returns",
            "card_type": "scene",
            "description": "Heathcliff returns wealthy after years away.",
            "image_search_query": "wealthy Heathcliff returns to Wuthering Heights",
        }
        candidates = [{
            "front_text": "Heathcliff’s Return to the Heights",
            "back_text": "A wealthy Heathcliff comes back after a long absence.",
            "category": "scene",
            "image_prompt": "wealthy Heathcliff returning to Wuthering Heights",
            "image_storage_path": "older-deck/card-021.jpg",
            "image_status": "generated",
        }]
        self.assertIsNotNone(choose_existing_asset(card, candidates))

    def test_category_mismatch_does_not_reuse_exact_label(self):
        card = {"label": "Wuthering Heights", "card_type": "setting"}
        candidates = [{
            "front_text": "Wuthering Heights",
            "category": "concept",
            "image_storage_path": "older-deck/card-030.jpg",
            "image_status": "generated",
        }]
        self.assertIsNone(choose_existing_asset(card, candidates))

    def test_match_normalization_handles_accents_and_possessives(self):
        self.assertEqual(normalize_match_text("Emily Brontë’s Novel"), "emily bronte novel")

    def test_normalize_image_reduces_large_dimensions(self):
        source = Image.new("RGBA", (1800, 900), (255, 0, 0, 128))
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        normalized = normalize_image(buffer.getvalue(), max_edge=600)
        with Image.open(io.BytesIO(normalized)) as result:
            self.assertEqual(result.mode, "RGB")
            self.assertLessEqual(max(result.size), 600)

    def test_crop_image_uses_card_aspect_ratio(self):
        source = Image.new("RGB", (1000, 1000), "blue")
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG")
        cropped = crop_image(buffer.getvalue(), focal_x=80, focal_y=20)
        with Image.open(io.BytesIO(cropped)) as result:
            self.assertAlmostEqual(result.width / result.height, 1.65, places=2)


if __name__ == "__main__":
    unittest.main()
