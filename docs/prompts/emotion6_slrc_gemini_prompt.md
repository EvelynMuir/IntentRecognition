# Gemini prompt: Emotion6 SLR-C prior

Copy the full prompt block below into Gemini. Return the generated JSON to
Codex; it will be validated and installed as the shared Emotion6 prior.

```text
Role
You are a top-tier computer-vision researcher, a prompt engineer specializing in
vision-language models such as CLIP, and an expert in visual emotion psychology.

Task
Build a fixed heterogeneous textual prior bank for label-distribution learning
on the Emotion6 image dataset. For every emotion class below, provide:
1. one concise canonical definition; and
2. six visually distinct scenario archetypes that translate the abstract
   emotion into concrete, CLIP-friendly photographic descriptions.

Fixed class order — never reorder, rename, add, merge, or omit classes
1. anger — strong displeasure, irritation, rage, resentment, or hostility
2. disgust — revulsion, aversion, contamination response, or strong dislike
3. fear — alarm, anxiety, perceived danger, threat, horror, or vulnerability
4. joy — happiness, delight, pleasure, playful enjoyment, or celebration
5. sadness — sorrow, unhappiness, grief, loss, loneliness, or disappointment
6. surprise — a sudden reaction to something unexpected, novel, or startling
7. neutral — emotionally balanced or low-arousal content without a dominant
   positive or negative affect

Dataset interpretation
- Emotion6 labels the emotion expressed by or evoked by the whole image, not
  only a photographed person's facial expression.
- Cover people, animals, objects, landscapes, events, indoor scenes, outdoor
  scenes, and graphic/web imagery when semantically appropriate.
- Neutral is a real seventh class, not a missing label. Neutral scenarios must
  describe ordinary, emotionally unmarked visual content rather than blank,
  corrupted, or meaningless images.

Scenario construction rules
- Produce exactly six archetypes per class.
- Across the six archetypes, cover all five axes:
  A. subjects and salient objects;
  B. actions, interactions, posture, and facial micro-expressions;
  C. setting and context;
  D. atmosphere, color, and affective tone;
  E. photographic style, framing, and lighting.
- Include at least four genuinely different settings per class.
- Make each text_query one natural English sentence of 25-55 words that can be
  passed directly to CLIP.
- Describe observable image content. Do not explain the ontology or say that an
  image "belongs to a class."
- Avoid abstract filler such as "an image conveying emotion" unless accompanied
  by concrete subjects, actions, setting, composition, color, and lighting.
- Make scenarios within each class diverse rather than paraphrases.
- Include both direct human/animal emotional expression and emotion evoked by
  scenes, objects, weather, composition, or events.
- Do not make every positive scene bright or every negative scene dark; preserve
  realistic intra-class visual variation.
- Do not mention Emotion6, CLIP, label IDs, probabilities, annotation, dataset,
  or classification inside text_query.

Boundary requirements
- anger vs disgust: anger should emphasize confrontation, obstruction, rage, or
  hostile action; disgust should emphasize repulsion, contamination, decay,
  offensive sensory material, or recoil.
- fear vs surprise: fear requires danger, vulnerability, dread, or threat;
  surprise requires unexpected novelty/startle and may be positive or neutral.
- joy vs surprise: joy emphasizes sustained pleasure, play, affection, success,
  or celebration; surprise emphasizes a sudden expectation violation.
- sadness vs neutral: sadness needs visible or scene-level evidence of loss,
  defeat, isolation, grief, or discouragement; neutral must avoid those cues.
- joy vs neutral: neutral may be calm or routine but must avoid clear smiling,
  celebration, playful interaction, reward, or pleasure cues.
- neutral must include at least these visual regimes across its six archetypes:
  a relaxed neutral portrait, an ordinary workspace, a quiet public place, a
  catalog-style object image, an informational graphic, and a routine activity.

Field requirements
- core_difference: state what visually distinguishes this archetype from the
  other five archetypes of the same class and from the nearest competing class.
- visual_elements: give a compact comma-separated list of observable anchors.
- text_query: integrate subjects, action/state, setting, composition, atmosphere,
  and lighting into a concrete photographic sentence.

Output constraints
- Return exactly one valid JSON object and nothing else.
- Do not use Markdown fences.
- Use double quotes, no comments, and no trailing commas.
- Follow this schema exactly:

{
  "document_title": "Emotion6 Emotion Archetypes for SLR-C",
  "class_order_source": "Emotion6 official seven-emotion order",
  "emotions": [
    {
      "id": 1,
      "emotion_name": "anger",
      "definition": "...",
      "archetypes": [
        {
          "archetype_id": 1,
          "name": "...",
          "core_difference": "...",
          "visual_elements": "...",
          "text_query": "..."
        }
      ]
    }
  ]
}

Before returning, silently verify all of the following:
- emotions has length 7 in the exact fixed order:
  anger, disgust, fear, joy, sadness, surprise, neutral;
- IDs are exactly 1-7;
- every class has exactly six archetypes with IDs 1-6;
- all 42 text_query values are unique;
- every text_query is concrete, photographic, grammatical, and 25-55 words;
- neutral descriptions contain no dominant affective cue;
- the complete output parses with a strict JSON parser.
```
