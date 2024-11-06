import numpy as np
import matplotlib.pyplot as plt
array = np.array[-5.,  5.,  5., -5.,  3.,  6.,  1., -6.,  6., -3.,  6., -1.,  6.,
                    -2., -5., -6., -6.,  5., -6., -1., -6.,  2., -6., -4.,  0., -6.,
                        6.,  1.,  5.,  6.,  4., -6., -1., -6., -4., -6., -6.,  1.,  6.,
                        5., -6., -2.,  6.,  2.,  6., -4.,  6.,  6., -6., -5.,  2.,  6.,
                        0.,  6., -1.,  6., -2.,  6.,  6.,  0., -3., -6.,  4.,  6.,  6.,
                    -6., -6.,  4., -6.,  3.,  6.,  3., -6., -6.,  3., -6., -6., -3.,
                        5., -6.,  1.,  6., -6.,  0., -2., -6.,  6.,  4.,  6., -5.,  2.,
                    -6., -1.,  0., -1.,  1., -2.,  2., -4.,  2.,  3., -5.,  2., -2.,
                    -1.,  2., -4.,  5., -4.,  6.,  1.,  1., -3.,  1., -3., -3., -1.,
                    -3., -3.,  6.,  2.,  1.,  1., -3., -6.,  6., -5.,  6.,  3., -2.,
                    -5.,  2., -3.,  0.,  1.,  2.,  3.,  1.,  5., -2.,  0.,  3.,  0.,
                    -3.,  4., -2.,  0.,  4., -2., -3.]

def obs_to_array(coordinates):
    # 提取agent、終點和牆壁的座標
    agent = coordinates[:2]
    goal = coordinates[2:4]
    walls = coordinates[4:]

    # 創建一個13x13的矩陣，初始化為0（空白空間）
    maze = np.zeros((13, 13), dtype=int)

    # 將座標轉換為矩陣索引的函數
    def coord_to_index(x, z):
        # 將x座標範圍從[-6, 6]映射到[0, 12]，右正左負
        # 將z座標範圍從[-6, 6]映射到[12, 0]，上正下負
        x_idx = int(np.round((x + 6) * 12 / 12))
        z_idx = 12 - int(np.round((z + 6) * 12 / 12))
        return max(0, min(12, x_idx)), max(0, min(12, z_idx))

    # 放置牆壁
    for i in range(0, len(walls), 2):
        x, z = walls[i], walls[i+1]
        idx_x, idx_z = coord_to_index(x, z)
        maze[idx_z, idx_x] = 3  # 牆壁用3表示

    # 放置agent
    agent_x, agent_z = coord_to_index(agent[0], agent[1])
    maze[agent_z, agent_x] = 1  # agent用1表示

    # 放置終點
    goal_x, goal_z = coord_to_index(goal[0], goal[1])
    maze[goal_z, goal_x] = 2  # 終點用2表示

    # 添加邊界牆壁
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 3

    return maze

# def visualize_maze(maze):
#     fig, ax = plt.subplots(figsize=(12, 12))
#     cmap = plt.cm.colors.ListedColormap(['white', 'red', 'green', 'gray'])
#     bounds = [0, 1, 2, 3, 4]
#     norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

#     ax.imshow(maze, cmap=cmap, norm=norm)

#     # 添加網格線
#     ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, 13, 1), minor=True)
#     ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
#     ax.tick_params(which="minor", size=0)

#     # 設置主刻度標籤
#     ax.set_xticks(np.arange(0, 13, 1))
#     ax.set_yticks(np.arange(0, 13, 1))
#     ax.set_xticklabels(range(-6, 7))
#     ax.set_yticklabels(range(6, -7, -1))

#     # 添加標籤
#     for i in range(13):
#         for j in range(13):
#             if maze[i, j] == 1:
#                 ax.text(j, i, 'A', ha='center', va='center', color='white', fontweight='bold')
#             elif maze[i, j] == 2:
#                 ax.text(j, i, 'G', ha='center', va='center', color='black', fontweight='bold')

#     ax.set_title("2D Maze Projection (13x13)")
#     ax.set_xlabel("X-axis")
#     ax.set_ylabel("Z-axis")

#     plt.show()
obs_to_array(array)
if __name__ == '__main__':
    obs_to_array()
    # visualize_maze()