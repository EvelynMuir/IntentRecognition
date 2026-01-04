import wandb

print("-" * 30)
# 1. 检查本地读取到的 API Key
try:
    api = wandb.Api()
    key = api.api_key
    if not key:
        print("❌ 错误：未检测到任何 API Key。请运行 wandb login")
        exit()
    print(f"🔑 当前使用的 API Key: {key[:4]}......{key[-4:]}")
except Exception as e:
    print(f"❌ 读取 Key 失败: {e}")
    exit()

# 2. 向服务器询问“我是谁”
print("📡 正在连接 W&B 服务器验证身份...")
try:
    # 获取当前用户信息
    viewer = api.viewer
    username = viewer.get('username')
    teams = [t['name'] for t in viewer.get('teams', [])]
    
    print(f"👤 当前登录用户名:  【 {username} 】")
    print(f"🏢 该用户所属团队:  {teams}")
    
    # 3. 检查是否有权访问目标 Entity
    target_entity = "yintang-beihang-university"
    
    if target_entity == username:
        print(f"✅ 目标 Entity 是你的个人账号，权限正常。")
    elif target_entity in teams:
        print(f"✅ 你在团队 '{target_entity}' 中，权限正常。")
    else:
        print(f"❌ 警告：你当前登录的是 '{username}'，但你不在团队 '{target_entity}' 中！")
        print(f"   这就是报 403 Forbidden 的原因。")

except Exception as e:
    print(f"❌ 身份验证失败 (401/403): {e}")
    print("   这意味着你的 API Key 可能已经失效或被重置。")

print("-" * 30)