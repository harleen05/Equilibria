from typing import List, Dict
from environment.models import ContentItem  # single source of truth — avoids duplicate class conflict
import hashlib


# -------------------------------
# 📦 Named Catalog
# All IDs referenced in easy/medium/hard task configs exist here.
# -------------------------------

NAMED_CATALOG: List[dict] = [
    # ── Relevant / Educational ──────────────────────────────────────────
    dict(
        content_id="rel_sci_01", title="Science Deep Dive",
        topic_relevance={"science": 1.0, "technology": 0.3, "health": 0.2,
                         "politics": 0.0, "entertainment": 0.0, "social": 0.0,
                         "finance": 0.0, "sports": 0.0, "general": 0.1},
        addictiveness=0.10, manipulation_score=0.05, educational_value=0.90, novelty=0.70,
    ),
    dict(
        content_id="rel_tech_01", title="Tech Innovation Weekly",
        topic_relevance={"technology": 1.0, "science": 0.4, "finance": 0.2,
                         "politics": 0.0, "entertainment": 0.0, "social": 0.0,
                         "health": 0.0, "sports": 0.0, "general": 0.1},
        addictiveness=0.15, manipulation_score=0.05, educational_value=0.85, novelty=0.75,
    ),
    dict(
        content_id="rel_fin_01", title="Personal Finance Explained",
        topic_relevance={"finance": 1.0, "technology": 0.2, "general": 0.2,
                         "science": 0.0, "health": 0.1, "politics": 0.2,
                         "entertainment": 0.0, "social": 0.0, "sports": 0.0},
        addictiveness=0.10, manipulation_score=0.08, educational_value=0.80, novelty=0.55,
    ),
    dict(
        content_id="rel_hist_01", title="History Matters",
        topic_relevance={"general": 0.8, "politics": 0.4, "science": 0.2,
                         "technology": 0.0, "health": 0.0, "entertainment": 0.1,
                         "social": 0.1, "finance": 0.0, "sports": 0.0},
        addictiveness=0.08, manipulation_score=0.03, educational_value=0.88, novelty=0.60,
    ),
    dict(
        content_id="rel_health_01", title="Healthy Living Guide",
        topic_relevance={"health": 1.0, "science": 0.3, "general": 0.2,
                         "technology": 0.0, "politics": 0.0, "entertainment": 0.0,
                         "social": 0.0, "finance": 0.0, "sports": 0.2},
        addictiveness=0.08, manipulation_score=0.04, educational_value=0.92, novelty=0.60,
    ),
    dict(
        content_id="rel_health_02", title="Mental Wellness Essentials",
        topic_relevance={"health": 0.9, "science": 0.3, "social": 0.2,
                         "technology": 0.0, "politics": 0.0, "entertainment": 0.0,
                         "general": 0.1, "finance": 0.0, "sports": 0.0},
        addictiveness=0.06, manipulation_score=0.03, educational_value=0.93, novelty=0.65,
    ),
    dict(
        content_id="rel_news_01", title="Balanced News Digest",
        topic_relevance={"politics": 0.8, "general": 0.6, "social": 0.3,
                         "technology": 0.1, "health": 0.1, "entertainment": 0.1,
                         "science": 0.0, "finance": 0.2, "sports": 0.0},
        addictiveness=0.20, manipulation_score=0.15, educational_value=0.65, novelty=0.70,
    ),
    dict(
        content_id="rel_env_01", title="Climate & Environment",
        topic_relevance={"science": 0.8, "politics": 0.4, "health": 0.3,
                         "technology": 0.2, "general": 0.2, "entertainment": 0.0,
                         "social": 0.1, "finance": 0.0, "sports": 0.0},
        addictiveness=0.10, manipulation_score=0.06, educational_value=0.87, novelty=0.68,
    ),
    # ── Random / Entertainment ───────────────────────────────────────────
    dict(
        content_id="rnd_film_01", title="Film Reviews Roundup",
        topic_relevance={"entertainment": 1.0, "general": 0.3, "social": 0.2,
                         "technology": 0.0, "health": 0.0, "politics": 0.0,
                         "science": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.30, manipulation_score=0.10, educational_value=0.30, novelty=0.80,
    ),
    dict(
        content_id="rnd_music_01", title="New Music Discoveries",
        topic_relevance={"entertainment": 0.9, "social": 0.3, "general": 0.2,
                         "technology": 0.0, "health": 0.0, "politics": 0.0,
                         "science": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.35, manipulation_score=0.08, educational_value=0.25, novelty=0.85,
    ),
    dict(
        content_id="rnd_food_01", title="Food & Recipe Inspiration",
        topic_relevance={"entertainment": 0.7, "health": 0.4, "general": 0.3,
                         "technology": 0.0, "politics": 0.0, "science": 0.0,
                         "social": 0.2, "finance": 0.0, "sports": 0.0},
        addictiveness=0.25, manipulation_score=0.05, educational_value=0.40, novelty=0.75,
    ),
    dict(
        content_id="rnd_sport_01", title="Sports Highlights",
        topic_relevance={"sports": 1.0, "entertainment": 0.4, "social": 0.2,
                         "technology": 0.0, "health": 0.1, "politics": 0.0,
                         "science": 0.0, "finance": 0.0, "general": 0.1},
        addictiveness=0.35, manipulation_score=0.10, educational_value=0.20, novelty=0.70,
    ),
    # ── Addictive ────────────────────────────────────────────────────────
    dict(
        content_id="add_scroll_01", title="Endless Scroll Feed",
        topic_relevance={"entertainment": 0.6, "social": 0.5, "general": 0.3,
                         "technology": 0.0, "health": 0.0, "politics": 0.0,
                         "science": 0.0, "finance": 0.0, "sports": 0.2},
        addictiveness=0.88, manipulation_score=0.30, educational_value=0.05, novelty=0.50,
    ),
    dict(
        content_id="add_satisfy_01", title="Satisfying Videos Compilation",
        topic_relevance={"entertainment": 0.8, "general": 0.3, "social": 0.2,
                         "technology": 0.0, "health": 0.0, "politics": 0.0,
                         "science": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.82, manipulation_score=0.20, educational_value=0.05, novelty=0.60,
    ),
    dict(
        content_id="add_gaming_01", title="Gaming Livestream Highlights",
        topic_relevance={"entertainment": 0.9, "technology": 0.3, "social": 0.3,
                         "general": 0.1, "health": 0.0, "politics": 0.0,
                         "science": 0.0, "finance": 0.0, "sports": 0.2},
        addictiveness=0.75, manipulation_score=0.20, educational_value=0.10, novelty=0.65,
    ),
    dict(
        content_id="add_social_01", title="Social Drama & Reactions",
        topic_relevance={"social": 0.9, "entertainment": 0.5, "general": 0.2,
                         "politics": 0.1, "health": 0.0, "technology": 0.0,
                         "science": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.80, manipulation_score=0.40, educational_value=0.05, novelty=0.55,
    ),
    dict(
        content_id="add_social_02", title="Influencer Life Updates",
        topic_relevance={"social": 0.8, "entertainment": 0.6, "general": 0.2,
                         "politics": 0.0, "health": 0.0, "technology": 0.0,
                         "science": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.78, manipulation_score=0.35, educational_value=0.05, novelty=0.60,
    ),
    # ── Misleading / Manipulative ────────────────────────────────────────
    dict(
        content_id="mis_outrage_01", title="THEY Don't Want You to Know This",
        topic_relevance={"politics": 0.9, "social": 0.5, "general": 0.3,
                         "technology": 0.0, "health": 0.0, "science": 0.0,
                         "entertainment": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.60, manipulation_score=0.90, educational_value=0.02, novelty=0.70,
    ),
    dict(
        content_id="mis_outrage_02", title="Shocking Truth Revealed",
        topic_relevance={"politics": 0.8, "social": 0.6, "general": 0.3,
                         "technology": 0.0, "health": 0.0, "science": 0.0,
                         "entertainment": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.55, manipulation_score=0.88, educational_value=0.02, novelty=0.65,
    ),
    dict(
        content_id="mis_click_01", title="You Won't Believe What Happened Next",
        topic_relevance={"entertainment": 0.6, "general": 0.4, "social": 0.3,
                         "technology": 0.0, "health": 0.0, "politics": 0.1,
                         "science": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.50, manipulation_score=0.70, educational_value=0.03, novelty=0.75,
    ),
    dict(
        content_id="mis_click_02", title="Doctors Hate This One Trick",
        topic_relevance={"health": 0.7, "general": 0.4, "science": 0.1,
                         "technology": 0.0, "politics": 0.0, "entertainment": 0.0,
                         "social": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.45, manipulation_score=0.75, educational_value=0.02, novelty=0.70,
    ),
    dict(
        content_id="mis_pseudo_01", title="Ancient Secrets Big Pharma Hides",
        topic_relevance={"health": 0.6, "general": 0.3, "science": 0.1,
                         "technology": 0.0, "politics": 0.1, "entertainment": 0.0,
                         "social": 0.0, "finance": 0.0, "sports": 0.0},
        addictiveness=0.40, manipulation_score=0.85, educational_value=0.01, novelty=0.65,
    ),
]


def get_full_catalog() -> Dict[str, "ContentItem"]:
    """Return the full named catalog as a dict keyed by content_id."""
    return {entry["content_id"]: ContentItem(**entry) for entry in NAMED_CATALOG}


def get_content_by_id(catalog: Dict[str, "ContentItem"], content_id: str) -> "ContentItem":
    if content_id not in catalog:
        raise ValueError(f"Content with id '{content_id}' not found.")
    return catalog[content_id]