# Gemini prompt: Flickr-LDL / Twitter-LDL SLR-C prior

Copy everything inside the following prompt block into Gemini. The returned JSON
is consumed directly by `scripts/run_ldl_fdil.py`.

```text
Role
You are a top-tier computer-vision researcher, a prompt engineer specializing in
vision-language models such as CLIP, and an expert in visual emotion psychology.

Task
Build a fixed heterogeneous textual prior bank for label-distribution learning
on Flickr-LDL and Twitter-LDL. The same bank will be used for both datasets.
For every emotion class below, provide:
1. one concise canonical definition; and
2. six visually distinct scenario archetypes that translate the abstract
   emotion into concrete, CLIP-friendly photographic descriptions.

Fixed class order — never reorder, rename, add, merge, or omit classes
1. amusement — enjoyment, humor, or playful entertainment
2. anger — strong displeasure, irritation, rage, or hostility
3. awe — wonder or reverence caused by something impressive, sublime, or vast
4. contentment — calm satisfaction, comfort, serenity, or peaceful happiness
5. disgust — revulsion, aversion, contamination response, or strong dislike
6. excitement — energetic enthusiasm, anticipation, thrill, or stimulation
7. fear — alarm, anxiety, perceived danger, threat, or vulnerability
8. sadness — sorrow, unhappiness, grief, loss, loneliness, or disappointment

Scenario construction rules
- Produce exactly six archetypes per class.
- Cover all five axes across the six archetypes:
  A. subjects and salient objects;
  B. actions, interactions, posture, and facial micro-expressions;
  C. setting and context;
  D. atmosphere, color, and affective tone;
  E. photographic style, framing, and lighting.
- Include at least four genuinely different settings per class, mixing people,
  animals, objects, landscapes, events, indoor scenes, and outdoor scenes when
  semantically appropriate.
- Flickr-LDL and Twitter-LDL label the emotion evoked by the whole image, not
  only a photographed person's facial expression. Include both expressed and
  image-evoked emotions.
- Make each text_query a single natural English sentence of 25-55 words that
  could be passed directly to CLIP. It must describe visible content, not explain
  the label ontology.
- Avoid abstract filler such as "an image conveying emotion" unless accompanied
  by concrete subjects, actions, setting, composition, color, and lighting.
- Make scenarios within a class diverse rather than paraphrases.
- Explicitly sharpen nearby boundaries where visual ambiguity is common:
  amusement vs excitement; awe vs fear; contentment vs amusement; anger vs
  disgust; sadness vs contentment.
- Do not make all positive emotions bright or all negative emotions dark.
  Preserve realistic intra-class variation.
- Do not mention Flickr-LDL, Twitter-LDL, CLIP, label IDs, probabilities, or
  "classification" inside text_query.
- core_difference must state what visually distinguishes that archetype from
  the other five archetypes of the same class and from its nearest emotion.
- visual_elements must be a compact comma-separated list of observable anchors.

Output constraints
- Return exactly one valid JSON object and nothing else.
- Do not wrap it in Markdown fences.
- Use double quotes; no comments; no trailing commas.
- Follow this schema exactly:

{
  "document_title": "Flickr-LDL and Twitter-LDL Emotion Archetypes for SLR-C",
  "class_order_source": "Flickr-LDL/Twitter-LDL Plutchik eight-emotion order",
  "emotions": [
    {
      "id": 1,
      "emotion_name": "amusement",
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

Before returning, silently verify:
- emotions has length 8 in the fixed order;
- every class has exactly 6 archetypes with IDs 1-6;
- every text_query is concrete, photographic, grammatical, and unique;
- the output parses with a strict JSON parser.
```
