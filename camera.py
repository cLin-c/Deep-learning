from omni.isaac.core import World
from omni.isaac.core.prims import Camera
from random import uniform

def create_random_camera(world):
    # 随机位置
    x = uniform(-10, 10)
    y = uniform(-10, 10)
    z = uniform(1, 5)
    
    # 随机朝向
    pitch = uniform(-30, 30)
    yaw = uniform(-180, 180)
    roll = 0
    
    # 创建摄像头
    camera = Camera(
        prim_path="/World/RandomCamera", 
        position=(x, y, z),
        rotation=(pitch, yaw, roll)
    )
    
    world.scene.add(camera)
    print(f"摄像头位置：({x}, {y}, {z}), 朝向：({pitch}, {yaw}, {roll})")
    return camera

# 初始化世界
world = World(stage_units_in_meters=1.0)

# 放置随机摄像头
camera = create_random_camera(world)

# 启动渲染
world.render()
