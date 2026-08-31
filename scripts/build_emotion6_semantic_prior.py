#!/usr/bin/env python3
"""Build Emotion6 SLR-C prior from the existing Gemini-Emotic archetype bank."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


MAPPING = {
    "anger": "anger",
    "disgust": "aversion",
    "fear": "fear",
    "joy": "happiness",
    "sadness": "sadness",
    "surprise": "surprise",
}

NEUTRAL = {
    "id": 7,
    "emotion_name": "neutral",
    "definition": "An emotionally balanced or low-arousal scene without a dominant positive or negative affect.",
    "archetypes": [
        {"archetype_id": 1, "name": "Neutral Portrait", "core_difference": "Relaxed face without a smile, frown, tension, or startle response.", "visual_elements": "single person, relaxed mouth, level gaze, plain background, even lighting", "text_query": "A straightforward portrait of a person looking calmly at the camera with relaxed facial muscles, a closed neutral mouth, level gaze, plain clothing, and soft even studio lighting against an uncluttered background"},
        {"archetype_id": 2, "name": "Ordinary Workspace", "core_difference": "Mundane functional setting with no active success, conflict, or distress cues.", "visual_elements": "desk, computer, chair, papers, empty office, diffuse daylight", "text_query": "An ordinary unoccupied office desk with a computer monitor, keyboard, chair, and neatly stacked papers, photographed in diffuse daylight with balanced colors and no people, dramatic action, celebration, or visible disorder"},
        {"archetype_id": 3, "name": "Quiet Street", "core_difference": "Low-activity public space without danger, grandeur, loneliness emphasis, or festive energy.", "visual_elements": "residential street, parked cars, closed storefronts, overcast sky, eye-level framing", "text_query": "A quiet residential street with several parked cars, closed storefronts, clean sidewalks, and a pale overcast sky, captured from an ordinary eye-level viewpoint with muted natural colors and no notable event taking place"},
        {"archetype_id": 4, "name": "Catalog Objects", "core_difference": "Purely descriptive object arrangement without contamination, humor, comfort, or symbolic meaning.", "visual_elements": "household objects, white surface, centered composition, shadowless product light", "text_query": "Several everyday household objects arranged separately on a clean white surface in a centered catalog-style composition, photographed with shadowless product lighting and accurate colors without expressive decoration, damage, mess, or human interaction"},
        {"archetype_id": 5, "name": "Informational Display", "core_difference": "Utilitarian information presentation without alarming, triumphant, or humorous framing.", "visual_elements": "simple chart, timetable, labels, restrained colors, flat graphic design", "text_query": "A simple informational timetable with evenly spaced rows, small labels, restrained blue and gray colors, and a flat functional layout, presented clearly without warning symbols, celebratory graphics, jokes, or emotionally charged language"},
        {"archetype_id": 6, "name": "Routine Activity", "core_difference": "Familiar low-arousal action performed without pleasure, frustration, fear, or surprise.", "visual_elements": "person walking, grocery bag, sidewalk, natural posture, midday light", "text_query": "A person walking at a normal pace along a familiar sidewalk while carrying a grocery bag, viewed from a moderate distance in plain midday light with natural posture and no strong facial expression or unusual event"},
    ],
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root.parent / "Emotic" / "emotion_description_gemini.json")
    parser.add_argument("--output", type=Path, default=root.parent / "LDL" / "processed" / "semantic_priors" / "emotion6_gemini_slrc_prior.json")
    args = parser.parse_args()
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    source_by_name = {str(item["emotion_name"]).lower(): item for item in payload["emotions"]}
    emotions = []
    for index, (target_name, source_name) in enumerate(MAPPING.items(), 1):
        item = deepcopy(source_by_name[source_name])
        item["id"] = index
        item["emotion_name"] = target_name
        item["source_emotion_name"] = source_by_name[source_name]["emotion_name"]
        emotions.append(item)
    emotions.append(NEUTRAL)
    output = {
        "document_title": "Emotion6 Archetypes for SLR-C",
        "class_order_source": "Emotion6 release order: anger, disgust, fear, joy, sadness, surprise, neutral",
        "source": str(args.source.resolve()),
        "source_note": "Six affective classes reuse Gemini-generated Emotic archetypes; neutral uses fixed concrete low-affect scenarios.",
        "emotions": emotions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
