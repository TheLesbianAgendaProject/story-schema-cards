import io
import unittest

from PIL import Image

from image_assets import crop_image, choose_existing_asset, normalize_image, similarity_score


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
