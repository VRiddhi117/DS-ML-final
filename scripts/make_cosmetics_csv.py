print("Starting cosmetics CSV creation...")

import os
import csv
from datasets import load_dataset

OUT = "data/raw/cosmetics_reviews.csv"
MAX_ROWS = 50000  # big enough for the project

def main():
    print("Loading Amazon Beauty Reviews dataset...")
    os.makedirs("data/raw", exist_ok=True)

    dataset = load_dataset(
        "jhan21/amazon-beauty-reviews-dataset",
        split="train"
    )

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "review_id",
            "product_id",
            "user_id",
            "rating",
            "verified_purchase",
            "helpful_votes",
            "review_text"
        ])

        count = 0
        for r in dataset:
            review_id = str(r.get("timestamp", count))
            product_id = str(r.get("asin", "")).strip()
            user_id = str(r.get("reviewer_id", "")).strip()
            rating = r.get("rating")
            verified = r.get("verified_purchase")
            helpful = r.get("helpful_vote", 0)
            text = (r.get("text") or "").replace("\n", " ").strip()

            if not product_id:
                continue

            writer.writerow([
                review_id,
                product_id,
                user_id,
                rating,
                verified,
                helpful,
                text
            ])

            count += 1
            if count >= MAX_ROWS:
                break

    print(f"✅ Finished. Wrote {count} rows to {OUT}")

if __name__ == "__main__":
    main()
