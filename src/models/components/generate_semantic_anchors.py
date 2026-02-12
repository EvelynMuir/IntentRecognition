"""
生成语义锚点库 (Semantic Anchors) 脚本

为5个维度（COCO, Places365, Emotion, AVA, Stanford 40 Actions）生成CLIP文本特征，
并应用扩充策略（噪声复制或同义词扩展）以达到目标codebook大小。
"""
import os
import torch
import clip
from tqdm import tqdm
from typing import List, Dict, Tuple


# ==================== 类别列表定义 ====================

# COCO 80类
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Places365 场景类别（从place365.txt提取）
def load_places365_classes(file_path: str = "/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/place365.txt") -> List[str]:
    """从place365.txt文件加载Places365类别列表"""
    classes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 格式: /a/airfield 0
                    # 提取类别路径（去掉数字索引）
                    parts = line.split()
                    if len(parts) >= 1:
                        class_path = parts[0]
                        # 去掉开头的"/"，并将斜杠替换为下划线，使类别名更自然
                        # 例如: /a/airfield -> a_airfield
                        # 例如: /a/apartment_building/outdoor -> a_apartment_building_outdoor
                        class_name = class_path.strip('/').replace('/', '_')
                        classes.append(class_name)
        print(f"Loaded {len(classes)} Places365 classes from {file_path}")
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, using default classes")
        classes = [
            "a_airport_terminal", "b_bedroom", "f_forest", "k_kitchen", "l_living_room", 
            "o_office", "r_restaurant", "s_street", "b_beach", "m_mountain"
        ]
    return classes

# Emotion 7类基本情绪及其同义词
EMOTION_SYNONYMS = {
    "Happiness": ["joy", "delight", "cheerful", "happy", "pleased", "glad", "merry", "jovial", "blissful", "ecstatic", "elated", "euphoric", "content", "satisfied"],
    "Sadness": ["sorrow", "grief", "melancholy", "unhappy", "depressed", "downcast", "dejected", "despondent", "miserable", "woeful", "mournful", "gloomy"],
    "Anger": ["rage", "fury", "wrath", "irritated", "annoyed", "furious", "enraged", "livid", "incensed", "outraged", "indignant"],
    "Fear": ["terror", "dread", "anxiety", "panic", "frightened", "scared", "afraid", "worried", "nervous", "apprehensive", "terrified"],
    "Surprise": ["astonishment", "amazement", "shock", "wonder", "bewilderment", "stunned", "astounded", "startled", "taken aback"],
    "Disgust": ["revulsion", "repulsion", "loathing", "aversion", "distaste", "repugnance", "contempt", "abhorrence"],
    "Neutral": ["calm", "composed", "serene", "peaceful", "tranquil", "balanced", "equanimous", "unemotional", "impassive"]
}

# AVA 14类风格属性
AVA_STYLE_ATTRIBUTES = [
    # --- 光影与色调 (Lighting & Tone) ---
    "Black and White", 
    "High Dynamic Range (HDR)", 
    "Analog",          # 胶片感、颗粒感
    "Infrared",        # 红外摄影、奇异色调
    "Emotive",         # 情绪化、低调或高调照明
    "Horror",          # 暗黑、高对比、压抑

    # --- 构图与视角 (Composition & Perspective) ---
    "Macro",           # 微距
    "Panoramic",       # 全景
    "Fish Eye",        # 鱼眼畸变
    "Lensbaby",        # 移轴、边缘模糊
    "Candid",          # 抓拍、不看镜头、自然构图
    
    # --- 纹理与后期 (Texture & Post-processing) ---
    "Abstract",        # 抽象、无明确主体
    "Blur",            # 模糊、动感模糊
    "Digital Art",     # 插画感、非自然纹理
    "Overlays",        # 双重曝光、纹理叠加
    "Fashion",         # 精致修图、完美光影
    "Advertisement"    # 高饱和、商业质感
]

# Stanford 40 Actions 40类动作
STANFORD40_ACTIONS = [
    "applauding", "blowing bubbles", "brushing hair", "brushing teeth", "cleaning the floor",
    "climbing", "cooking", "cutting", "cutting vegetables", "drinking", "feeding a horse",
    "fishing", "fixing a bike", "fixing a car", "gardening", "holding an umbrella",
    "jumping", "looking through a microscope", "looking through a telescope", "playing guitar",
    "playing violin", "pouring liquid", "pushing a cart", "reading", "phoning",
    "riding a bike", "riding a horse", "rowing a boat", "running", "shooting an arrow",
    "smoking", "taking photos", "texting message", "throwing frisby", "using a computer",
    "walking the dog", "washing dishes", "watching TV", "waving hands", "writing on a board"
]


# ==================== 扩充策略函数 ====================

def augment_with_noise(base_embedding: torch.Tensor, num_codes: int, noise_scale: float = 0.05) -> torch.Tensor:
    """
    通过添加噪声扩充embedding
    
    Args:
        base_embedding: 基础embedding [1, dim] 或 [dim]
        num_codes: 需要生成的code数量
        noise_scale: 噪声缩放因子
    
    Returns:
        扩充后的embeddings [num_codes, dim]
    """
    if base_embedding.dim() == 1:
        base_embedding = base_embedding.unsqueeze(0)  # [1, dim]
    
    dim = base_embedding.shape[-1]
    augmented = []
    
    for i in range(num_codes):
        noise = torch.randn_like(base_embedding) * noise_scale
        augmented_embedding = base_embedding + noise
        # 重新归一化
        augmented_embedding = augmented_embedding / augmented_embedding.norm(dim=-1, keepdim=True)
        augmented.append(augmented_embedding.squeeze(0))
    
    return torch.stack(augmented)  # [num_codes, dim]


def expand_emotion_synonyms(emotion_synonyms: Dict[str, List[str]], target_size: int = 128) -> List[str]:
    """
    通过同义词扩展情绪类别列表
    
    Args:
        emotion_synonyms: 情绪同义词字典
        target_size: 目标codebook大小
    
    Returns:
        扩展后的情绪词列表
    """
    expanded = []
    for emotion, synonyms in emotion_synonyms.items():
        expanded.extend(synonyms)
    
    # 如果还不够，重复一些词直到达到目标大小
    while len(expanded) < target_size:
        expanded.extend(expanded[:target_size - len(expanded)])
    
    return expanded[:target_size]


# ==================== 生成锚点函数 ====================

def generate_anchors_for_factor(
    class_names: List[str],
    clip_model,
    clip_tokenize,
    prompt_template: str,
    target_size: int,
    use_noise_augmentation: bool = False,
    noise_scale: float = 0.05,
    device: str = "cuda"
) -> Tuple[torch.Tensor, List[str]]:
    """
    为单个factor生成锚点embeddings
    
    Args:
        class_names: 类别名称列表
        clip_model: CLIP模型
        clip_tokenize: CLIP tokenize函数
        prompt_template: Prompt模板，如 "a photo of {}" 或 "a photo of a person {}"
        target_size: 目标codebook大小
        use_noise_augmentation: 是否使用噪声扩充
        noise_scale: 噪声缩放因子
        device: 计算设备
    
    Returns:
        (embeddings tensor [target_size, dim], 使用的文本列表)
    """
    embeddings_list = []
    texts_used = []
    
    with torch.no_grad():
        for name in tqdm(class_names, desc=f"Processing {len(class_names)} classes"):
            # 构建prompt
            prompt = prompt_template.format(name)
            texts_used.append(prompt)
            
            # 编码文本
            text_tokens = clip_tokenize([prompt]).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化
            
            embeddings_list.append(text_features.cpu().squeeze(0))
    
    # 如果使用噪声扩充
    if use_noise_augmentation:
        num_classes = len(class_names)
        num_codes_per_class = target_size // num_classes
        remainder = target_size % num_classes
        
        augmented_embeddings = []
        for i, base_emb in enumerate(embeddings_list):
            num_codes = num_codes_per_class + (1 if i < remainder else 0)
            augmented = augment_with_noise(base_emb.unsqueeze(0), num_codes, noise_scale)
            augmented_embeddings.append(augmented)
            # 扩展文本列表
            for j in range(num_codes - 1):
                texts_used.append(texts_used[i] + f"_noise_{j+1}")
        
        embeddings = torch.cat(augmented_embeddings, dim=0)  # [target_size, dim]
    else:
        embeddings = torch.stack(embeddings_list)  # [num_classes, dim]
        
        # 如果数量不够，通过重复或噪声扩充
        if embeddings.shape[0] < target_size:
            num_needed = target_size - embeddings.shape[0]
            # 使用噪声扩充剩余部分
            additional = augment_with_noise(embeddings[0:1], num_needed, noise_scale)
            embeddings = torch.cat([embeddings, additional], dim=0)
            for i in range(num_needed):
                texts_used.append(texts_used[0] + f"_aug_{i}")
    
    return embeddings[:target_size], texts_used[:target_size]


def generate_all_anchors(
    clip_model_name: str = "ViT-L/14",
    output_path: str = "semantic_anchors.pth",
    device: str = None
) -> Dict[str, torch.Tensor]:
    """
    生成所有5个维度的语义锚点
    
    Args:
        clip_model_name: CLIP模型名称
        output_path: 输出文件路径
        device: 计算设备
    
    Returns:
        包含5个维度锚点的字典
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    print(f"加载CLIP模型: {clip_model_name}")
    
    # 加载CLIP模型
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
    clip_model.eval()
    
    anchors = {}
    texts_dict = {}
    
    # 1. COCO (Object) - 256 codes, 每类~3个
    print("\n" + "="*60)
    print("生成 COCO (Object) 锚点...")
    print("="*60)
    coco_embeddings, coco_texts = generate_anchors_for_factor(
        class_names=COCO_CLASSES,
        clip_model=clip_model,
        clip_tokenize=clip.tokenize,
        prompt_template="a photo of {}",
        target_size=256,
        use_noise_augmentation=True,
        noise_scale=0.05,
        device=device
    )
    anchors["coco"] = coco_embeddings
    texts_dict["coco"] = coco_texts
    print(f"COCO embeddings shape: {coco_embeddings.shape}")
    
    # 2. Places365 (Scene) - 512 codes
    print("\n" + "="*60)
    print("生成 Places365 (Scene) 锚点...")
    print("="*60)
    # 从文件加载Places365类别
    places365_classes = load_places365_classes()
    places_embeddings, places_texts = generate_anchors_for_factor(
        class_names=places365_classes,
        clip_model=clip_model,
        clip_tokenize=clip.tokenize,
        prompt_template="a photo of {}",
        target_size=512,
        use_noise_augmentation=False,  # 365类，需要扩充到512
        device=device
    )
    anchors["places"] = places_embeddings
    texts_dict["places"] = places_texts
    print(f"Places365 embeddings shape: {places_embeddings.shape}")
    
    # 3. Emotion - 128 codes, 通过同义词扩展
    print("\n" + "="*60)
    print("生成 Emotion 锚点...")
    print("="*60)
    emotion_words = expand_emotion_synonyms(EMOTION_SYNONYMS, target_size=128)
    emotion_embeddings, emotion_texts = generate_anchors_for_factor(
        class_names=emotion_words,
        clip_model=clip_model,
        clip_tokenize=clip.tokenize,
        prompt_template="a photo of {}",
        target_size=128,
        use_noise_augmentation=False,
        device=device
    )
    anchors["emotion"] = emotion_embeddings
    texts_dict["emotion"] = emotion_texts
    print(f"Emotion embeddings shape: {emotion_embeddings.shape}")
    
    # 4. AVA (Style) - 256 codes, 每类~18个
    print("\n" + "="*60)
    print("生成 AVA (Style) 锚点...")
    print("="*60)
    ava_embeddings, ava_texts = generate_anchors_for_factor(
        class_names=AVA_STYLE_ATTRIBUTES,
        clip_model=clip_model,
        clip_tokenize=clip.tokenize,
        prompt_template="a {} photo",  # AVA使用不同的prompt格式
        target_size=256,
        use_noise_augmentation=True,
        noise_scale=0.05,
        device=device
    )
    anchors["ava"] = ava_embeddings
    texts_dict["ava"] = ava_texts
    print(f"AVA embeddings shape: {ava_embeddings.shape}")
    
    # 5. Stanford 40 Actions - 256 codes
    print("\n" + "="*60)
    print("生成 Stanford 40 Actions 锚点...")
    print("="*60)
    actions_embeddings, actions_texts = generate_anchors_for_factor(
        class_names=STANFORD40_ACTIONS,
        clip_model=clip_model,
        clip_tokenize=clip.tokenize,
        prompt_template="a photo of a person {}",  # 注意：Action使用不同的prompt格式
        target_size=256,
        use_noise_augmentation=False,  # 40类，可能需要一些扩充
        device=device
    )
    anchors["actions"] = actions_embeddings
    texts_dict["actions"] = actions_texts
    print(f"Actions embeddings shape: {actions_embeddings.shape}")
    
    # 保存锚点
    print("\n" + "="*60)
    print("保存锚点文件...")
    print("="*60)
    torch.save(anchors, output_path)
    print(f"锚点已保存到: {output_path}")
    
    # 保存文本列表（用于参考）
    texts_path = output_path.replace(".pth", "_texts.pth")
    torch.save(texts_dict, texts_path)
    print(f"文本列表已保存到: {texts_path}")
    
    # 打印总结
    print("\n" + "="*60)
    print("生成完成！总结:")
    print("="*60)
    for factor, emb in anchors.items():
        print(f"{factor:10s}: shape {emb.shape}")
    
    return anchors


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成语义锚点库")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14", 
                       help="CLIP模型名称 (默认: ViT-L/14)")
    parser.add_argument("--output", type=str, default="semantic_anchors.pth",
                       help="输出文件路径 (默认: semantic_anchors.pth)")
    parser.add_argument("--device", type=str, default=None,
                       help="计算设备 (默认: 自动选择)")
    
    args = parser.parse_args()
    
    generate_all_anchors(
        clip_model_name=args.clip_model,
        output_path=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()

