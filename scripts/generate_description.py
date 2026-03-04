DESCRIPTION_PROMPT = """
# Role
You are a top-tier Computer Vision (CV) expert, a Prompt Engineer specializing in multimodal models (e.g., CLIP, BLIP), and an expert in visual psychology. 

# Task
I am working on a "Human Intent Image Classification" task using real-world web images. Intent is often expressed implicitly in images, presenting high "intra-class variance" (one intent can have hundreds of different visual manifestations) and "inter-class similarity" (different intents might appear in visually similar scenes). 
Your task is to leverage your extensive world knowledge to generate highly rich, concrete, and visually discriminative descriptions (Visual Anchors) for a specific intent category. These descriptions will serve as text queries to help a Vision-Language model accurately classify images.

# Target Intent
The current intent category to describe is: "[Insert specific category name here, e.g., Exploration - Being curious and adventurous. Having an exciting, stimulating life.]"

# Output Guidelines
To fully cover the implicit visual elements, please construct your description strictly across the following 5 dimensions. Translate the abstract intent into concrete visual anchors:

1. Visual Subjects & Objects: 
List 3 to 5 typical objects or subject combinations that strongly imply this intent (e.g., for "exploration," do not just say a backpack; include a topographical map, an off-road vehicle, a DSLR camera, or mud-splattered boots).

2. Actions & Interactions: 
Describe specific body language, micro-expressions, or ongoing physical interactions using strong action verbs (e.g., leaning in to observe closely, raising hands in triumph, staring intently at a screen).

3. Settings & Context: 
To address "intra-class variance," provide at least 3 distinct typical scenarios (e.g., indoor vs. outdoor, daily life vs. extreme environments) where this intent naturally occurs.

4. Emotion, Vibe & Atmosphere: 
What is the overall atmosphere conveyed by the image? What are the specific facial expressions or mood indicators? (e.g., a serene gaze, a warm and cozy domestic vibe, high-energy excitement).

5. Photography Style & Lighting: 
What visual style typically characterizes web images of this intent? (e.g., high-contrast natural sunlight, warm indoor ambient lighting, vibrant and saturated colors, candid documentary style).

# Format Requirement
First, output a 50-80 word [Comprehensive Summary]. This summary should seamlessly integrate the visual elements above into a natural, cohesive, and highly visual paragraph. This will serve as my core Query.
Second, provide a concise list breaking down the keywords/phrases for the 5 dimensions mentioned above.

Crucial Note: While generating, actively consider the core visual differences between this intent and other highly similar intents (e.g., distinguishing "Enjoying life" from "Having an easy and comfortable life"), and heavily emphasize those unique visual discriminators in your description.
"""


import json
import os
import sys
from pathlib import Path
from time import sleep
from typing import List

from google import genai
from pydantic import BaseModel, Field
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.models.intentonomy_clip_vit_slot_module import INTENTONOMY_DESCRIPTIONS

# 请在此处填入你的 Gemini API Key，或者通过环境变量 GEMINI_API_KEY 设置
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

# 1. 确保输出目录存在
output_dir = "/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "intent_description_gemini.json")

# 2. 定义 Prompt 填充函数
def get_full_prompt(intent_name):
    return DESCRIPTION_PROMPT.replace(
        "[Insert specific category name here, e.g., Exploration - Being curious and adventurous. Having an exciting, stimulating life.]",
        intent_name
    )


class VisualAnchorDimension(BaseModel):
    name: str = Field(..., description="One of the 5 required dimensions")
    keywords: List[str] = Field(..., description="Keywords/phrases for this dimension")


class MatchResult(BaseModel):
    intent: str = Field(..., description="The intent category being described")
    summary: str = Field(..., description="50-80 word comprehensive summary")
    dimensions: List[VisualAnchorDimension] = Field(
        ..., description="Exactly 5 dimension entries in order"
    )

# 3. 遍历 descriptions (共28个类别) 并调用 API
all_results = []

descriptions = INTENTONOMY_DESCRIPTIONS

for idx, intent in enumerate(tqdm(descriptions, desc="Generating Intent Descriptions")):
    prompt = get_full_prompt(intent)
    
    max_retries = 3
    success = False
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
                config={
                    "tools": [
                        {"google_search": {}},
                    ],
                    "response_mime_type": "application/json",
                    "response_json_schema": MatchResult.model_json_schema(),
                },
            )
            content = response.text.strip()
            try:
                parsed = json.loads(content) if content else {}
            except json.JSONDecodeError:
                parsed = {"raw": content}
            all_results.append({
                "id": idx + 1,
                "intent": intent,
                "visual_anchor": parsed
            })
            success = True
            break
        except Exception as e:
            print(f"\n[Error] Intent: {intent[:30]}... Attempt {attempt+1} failed: {e}")
            sleep(2) # 失败后稍作等待
    
    if not success:
        print(f"\n[Critical] Failed to generate for: {intent}")
        all_results.append({"id": idx + 1, "intent": intent, "error": "API Failure"})
    
    # 限制频率，防止被封
    sleep(0.5)

# 4. 保存结果到文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n任务完成！共处理 {len(all_results)} 条意图，结果已保存至: {output_file}")
