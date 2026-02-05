"""
CLIP Text Anchors for Factor Attention

生成5个数据集的CLIP文本anchor，每个anchor是该数据集所有类别prompt embedding的平均值。
"""
from typing import List, Tuple
import torch
import clip
from pathlib import Path


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

# Emotion 7类（仅基础类别，不扩充）
EMOTION_CLASSES = [
    "Happiness",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise",
    "Disgust",
    "Neutral"
]

# AVA 14类风格属性
AVA_STYLE_ATTRIBUTES = [
    "Black and White", 
    "High Dynamic Range (HDR)", 
    "Analog",
    "Infrared",
    "Emotive",
    "Horror",
    "Macro",
    "Panoramic",
    "Fish Eye",
    "Lensbaby",
    "Candid",
    "Abstract",
    "Blur",
    "Digital Art",
    "Overlays",
    "Fashion",
    "Advertisement"
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


def generate_text_anchors(
    clip_model,
    clip_tokenize,
    device: str = "cpu"
) -> Tuple[torch.Tensor, List[str]]:
    """
    生成5个数据集的CLIP文本anchor
    
    Args:
        clip_model: CLIP模型
        clip_tokenize: CLIP tokenize函数
        device: 计算设备
    
    Returns:
        anchors: [5, embedding_dim] 的tensor，每个anchor是一个数据集的平均embedding
        anchor_names: 5个anchor的名称列表
    """
    anchors = []
    anchor_names = []
    
    # 定义5个数据集及其prompt模板
    datasets = [
        {
            "name": "COCO",
            "classes": COCO_CLASSES,
            "prompt_template": "a photo of {}"
        },
        {
            "name": "Places365",
            "classes": load_places365_classes(),
            "prompt_template": "a photo of {}"
        },
        {
            "name": "Emotion",
            "classes": EMOTION_CLASSES,
            "prompt_template": "a photo of {}"
        },
        {
            "name": "AVA",
            "classes": AVA_STYLE_ATTRIBUTES,
            "prompt_template": "a {} photo"
        },
        {
            "name": "Stanford40",
            "classes": STANFORD40_ACTIONS,
            "prompt_template": "a photo of a person {}"
        }
    ]
    
    clip_model.eval()
    with torch.no_grad():
        for dataset in datasets:
            name = dataset["name"]
            classes = dataset["classes"]
            prompt_template = dataset["prompt_template"]
            
            # 为每个类别生成prompt并编码
            embeddings_list = []
            for class_name in classes:
                prompt = prompt_template.format(class_name)
                text_tokens = clip_tokenize([prompt]).to(device)
                text_features = clip_model.encode_text(text_tokens)
                # 归一化（CLIP的标准做法）
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings_list.append(text_features.squeeze(0))
            
            # 计算平均embedding
            embeddings_tensor = torch.stack(embeddings_list)  # [num_classes, embedding_dim]
            anchor = embeddings_tensor.mean(dim=0)  # [embedding_dim]
            # 归一化
            anchor = anchor / anchor.norm(dim=-1, keepdim=True)
            
            anchors.append(anchor)
            anchor_names.append(name)
            print(f"Generated anchor for {name}: {len(classes)} classes, embedding dim: {anchor.shape[0]}")
    
    anchors_tensor = torch.stack(anchors)  # [5, embedding_dim]
    return anchors_tensor, anchor_names

