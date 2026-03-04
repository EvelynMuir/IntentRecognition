#!/usr/bin/env python3
"""
PatchSlotClassifier 快速演示

展示如何使用干净版本的Patch-Slot分类器
"""

import torch
import sys

sys.path.insert(0, "/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra")

from src.models.intentonomy_clip_vit_slot_module import (
    PatchSlotClassifier,
    SlotAttention,
)


def demo_slot_attention():
    """演示纯Slot Attention"""
    print("=" * 70)
    print("演示 1: 纯Slot Attention（无intent conditioning）")
    print("=" * 70)
    
    # 参数
    batch_size = 2
    num_tokens = 196  # 14x14 patch grid
    hidden_dim = 768  # ViT-B/32
    num_slots = 4
    
    # 创建slot attention
    slot_attn = SlotAttention(
        dim=hidden_dim,
        num_slots=num_slots,
        iters=3,
    )
    
    # 模拟patch tokens
    tokens = torch.randn(batch_size, num_tokens, hidden_dim)
    
    print(f"\n输入:")
    print(f"  - Patch tokens: {tokens.shape}")
    print(f"  - 每个token维度: {hidden_dim}")
    print(f"  - Patch grid: 14×14 = {num_tokens}")
    
    # 执行slot attention
    slots = slot_attn(tokens)
    
    print(f"\n输出:")
    print(f"  - Slots: {slots.shape}")
    print(f"  - 学习到 {num_slots} 个slots")
    print(f"  - 每个slot维度: {hidden_dim}")
    
    # 分析slots
    slot_norms = torch.norm(slots, dim=-1)
    print(f"\n第一张图的slot norms:")
    for i, norm in enumerate(slot_norms[0]):
        print(f"  - Slot {i+1}: {norm.item():.4f}")
    
    print("\n✅ Slot Attention演示完成！\n")


def demo_patch_slot_classifier():
    """演示完整的PatchSlotClassifier"""
    print("=" * 70)
    print("演示 2: PatchSlotClassifier（完整流程）")
    print("=" * 70)
    
    # 参数
    batch_size = 2
    num_tokens = 196
    hidden_dim = 768
    num_intents = 28
    num_slots = 4
    
    # 创建分类器
    classifier = PatchSlotClassifier(
        dim=hidden_dim,
        num_intents=num_intents,
        num_slots=num_slots,
        slot_iters=3,
    )
    
    # 模拟输入
    tokens = torch.randn(batch_size, num_tokens, hidden_dim)
    cls_embed = torch.randn(batch_size, hidden_dim)
    
    print(f"\n输入:")
    print(f"  - Patch tokens: {tokens.shape}")
    print(f"  - CLS embedding: {cls_embed.shape}")
    
    # Forward pass
    logits, slots = classifier(
        tokens=tokens,
        cls_embed=cls_embed,
        return_slots=True,
    )
    
    print(f"\n输出:")
    print(f"  - Logits: {logits.shape}")
    print(f"  - Predicted intents: {num_intents}")
    print(f"  - Slots: {slots.shape}")
    
    # 分析logits
    probs = torch.sigmoid(logits)
    print(f"\n第一张图的前5个intent概率:")
    for i, prob in enumerate(probs[0, :5]):
        print(f"  - Intent {i+1}: {prob.item():.4f}")
    
    # 显示参数量
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 可训练: {trainable_params:,}")
    
    print("\n✅ PatchSlotClassifier演示完成！\n")


def demo_normalization_effect():
    """演示normalization的效果"""
    print("=" * 70)
    print("演示 3: Normalization效果")
    print("=" * 70)
    
    batch_size = 2
    num_tokens = 196
    hidden_dim = 768
    num_intents = 28
    num_slots = 4
    
    classifier = PatchSlotClassifier(
        dim=hidden_dim,
        num_intents=num_intents,
        num_slots=num_slots,
        slot_iters=2,
    )
    
    # 创建不同范数的输入
    tokens_small = torch.randn(batch_size, num_tokens, hidden_dim) * 0.1
    tokens_large = torch.randn(batch_size, num_tokens, hidden_dim) * 10.0
    cls_embed = torch.randn(batch_size, hidden_dim)
    
    print(f"\n输入范数对比:")
    print(f"  - Small tokens norm: {torch.norm(tokens_small[0, 0]).item():.4f}")
    print(f"  - Large tokens norm: {torch.norm(tokens_large[0, 0]).item():.4f}")
    
    # 分别处理
    logits_small, _ = classifier(tokens_small, cls_embed, return_slots=False)
    logits_large, _ = classifier(tokens_large, cls_embed, return_slots=False)
    
    print(f"\n输出logits范围:")
    print(f"  - Small tokens: [{logits_small.min():.4f}, {logits_small.max():.4f}]")
    print(f"  - Large tokens: [{logits_large.min():.4f}, {logits_large.max():.4f}]")
    
    # 计算差异
    diff = (logits_small - logits_large).abs().mean()
    print(f"\n平均差异: {diff.item():.6f}")
    print(f"说明: Normalization使输出对输入范数变化不敏感")
    
    print("\n✅ Normalization演示完成！\n")


def demo_comparison():
    """对比不同配置"""
    print("=" * 70)
    print("演示 4: 不同配置对比")
    print("=" * 70)
    
    batch_size = 2
    num_tokens = 196
    hidden_dim = 768
    num_intents = 28
    
    configs = [
        {"num_slots": 2, "slot_iters": 2},
        {"num_slots": 4, "slot_iters": 3},
        {"num_slots": 8, "slot_iters": 3},
    ]
    
    tokens = torch.randn(batch_size, num_tokens, hidden_dim)
    cls_embed = torch.randn(batch_size, hidden_dim)
    
    print(f"\n配置对比（输入相同）:")
    print(f"{'配置':<20} {'参数量':<15} {'输出范围':<25}")
    print("-" * 60)
    
    for config in configs:
        classifier = PatchSlotClassifier(
            dim=hidden_dim,
            num_intents=num_intents,
            **config,
        )
        
        params = sum(p.numel() for p in classifier.parameters())
        logits, _ = classifier(tokens, cls_embed, return_slots=False)
        
        config_str = f"K={config['num_slots']}, iters={config['slot_iters']}"
        output_str = f"[{logits.min():.3f}, {logits.max():.3f}]"
        
        print(f"{config_str:<20} {params:>12,}   {output_str:<25}")
    
    print("\n说明:")
    print("  - 更多slots → 更多参数，可能捕获更多信息")
    print("  - 更多迭代 → 更精细的slot refinement")
    
    print("\n✅ 配置对比演示完成！\n")


def main():
    print("\n" + "=" * 70)
    print("PatchSlotClassifier 交互式演示")
    print("=" * 70 + "\n")
    
    print("这个演示展示了干净版本的Patch-Slot分类器如何工作")
    print("包括：纯Slot Attention、完整分类流程、Normalization效果等\n")
    
    # 运行所有演示
    try:
        demo_slot_attention()
        input("按Enter继续...")
        
        demo_patch_slot_classifier()
        input("按Enter继续...")
        
        demo_normalization_effect()
        input("按Enter继续...")
        
        demo_comparison()
        
        print("=" * 70)
        print("🎉 所有演示完成！")
        print("=" * 70)
        print("\n下一步:")
        print("  1. 运行单元测试: python test_single_layer_unit.py")
        print("  2. 训练模型: python src/train.py experiment=intentonomy_clip_vit_slot_single_layer")
        print("  3. 查看文档: docs/single_layer_patch_slot_classifier.md\n")
        
    except KeyboardInterrupt:
        print("\n\n演示被中断。")
        sys.exit(0)


if __name__ == "__main__":
    main()
