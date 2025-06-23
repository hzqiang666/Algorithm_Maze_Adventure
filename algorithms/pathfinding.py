 # -*- coding: utf-8 -*-
"""
动态规划求解迷宫最优路径（支持负权重）
核心算法：类Bellman-Ford思想的网格DP
"""
from heapq import heappush, heappop  # 添加这行导入
from config import ELEMENT_MAPPING, GOLD_VALUE, TRAP_PENALTY
import sys


def calculate_optimal_path(grid):
    rows, cols = len(grid), len(grid[0])
    start_pos, exit_pos = _find_special_positions(grid)
    gold_positions = {(i, j) for i in range(rows) for j in range(cols)
                      if grid[i][j] == ELEMENT_MAPPING['G']}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 使用类封装状态（便于路径重构）
    class State:
        __slots__ = ['i', 'j', 'mask', 'score', 'prev']

        def __init__(self, i, j, mask, score, prev=None):
            self.i = i
            self.j = j
            self.mask = mask
            self.score = score
            self.prev = prev

        # 添加比较方法（按 score 排序）
        def __lt__(self, other):
            return self.score > other.score  # 注意：堆默认是最小堆，这里用 > 实现最大堆

    # 初始化
    num_golds = len(gold_positions)
    pos_to_bit = {pos: i for i, pos in enumerate(sorted(gold_positions))}
    init_mask = 0
    init_score = 0

    # 使用优先队列（按score降序）
    heap = []
    visited = {}  # (i,j,mask) -> best_score
    start_state = State(start_pos[0], start_pos[1], init_mask, init_score)
    heappush(heap, (-start_state.score, start_state))
    visited[(start_pos[0], start_pos[1], init_mask)] = init_score

    # 主循环（类似Dijkstra）
    final_state = None
    while heap:
        neg_score, current = heappop(heap)
        current_score = -neg_score

        # 到达终点时记录最优状态
        if (current.i, current.j) == exit_pos:
            final_state = current
            break

        # 遍历四个方向
        for di, dj in directions:
            ni, nj = current.i + di, current.j + dj
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            if grid[ni][nj] == ELEMENT_MAPPING['#']:
                continue

            # 计算新状态
            new_mask = current.mask
            new_score = current_score
            cell_type = grid[ni][nj]

            if cell_type == ELEMENT_MAPPING['G']:
                bit = pos_to_bit.get((ni, nj), -1)
                if bit != -1 and not (new_mask & (1 << bit)):
                    new_mask |= (1 << bit)
                    new_score += GOLD_VALUE
            elif cell_type == ELEMENT_MAPPING['T']:
                # 需要检查这个陷阱是否在之前路径中触发过
                # 这里简化处理（实际需要沿prev指针回溯检查）
                new_score += TRAP_PENALTY

            # 更新状态
            state_key = (ni, nj, new_mask)
            if state_key not in visited or new_score > visited[state_key]:
                visited[state_key] = new_score
                new_state = State(ni, nj, new_mask, new_score, current)
                heappush(heap, (-new_score, new_state))

    # 重构路径
    if final_state is None:
        return 0, []

    path = []
    current = final_state
    while current:
        path.append((current.i, current.j))
        current = current.prev
    path.reverse()

    return final_state.score, path

def _reconstruct_with_resources(grid, dp, exit_pos, best_mask, pos_to_bit):
    """重构考虑资源使用状态的路径"""
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 反向追踪需要记录前驱状态
    path = []
    i, j = exit_pos
    current_mask = best_mask
    remaining_traps = set((i, j) for i in range(rows) for j in range(cols)
                          if grid[i][j] == ELEMENT_MAPPING['T'])

    # 检查终点是否可达
    if dp[i][j][current_mask] == -sys.maxsize:
        return path

    # 由于DP过程未记录前驱，需要二次遍历重建路径
    # 这里采用从终点反向模拟的方式
    while True:
        path.append((i, j))
        cell_type = grid[i][j]

        # 如果是起点则终止
        if cell_type == ELEMENT_MAPPING['S']:
            break

        # 计算当前单元格的影响
        cell_value = 0
        prev_mask = current_mask
        if cell_type == ELEMENT_MAPPING['G']:
            bit = pos_to_bit.get((i, j), -1)
            if bit != -1 and (current_mask & (1 << bit)):
                cell_value = GOLD_VALUE
                prev_mask = current_mask ^ (1 << bit)  # 回退金币状态
        elif cell_type == ELEMENT_MAPPING['T'] and (i, j) in remaining_traps:
            cell_value = TRAP_PENALTY
            remaining_traps.remove((i, j))

        # 寻找合法前驱
        found = False
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if grid[ni][nj] == ELEMENT_MAPPING['#']:
                    continue

                # 检查状态转移是否合理
                if dp[ni][nj][prev_mask] + cell_value == dp[i][j][current_mask]:
                    i, j = ni, nj
                    current_mask = prev_mask
                    found = True
                    break

        if not found:  # 回溯失败（理论上不应发生）
            break

    path.reverse()
    return path

def _find_special_positions(grid):
    """定位起点和终点坐标"""
    start_pos, exit_pos = None, None
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == ELEMENT_MAPPING['S']:
                start_pos = (i, j)
            elif grid[i][j] == ELEMENT_MAPPING['E']:
                exit_pos = (i, j)
    return start_pos, exit_pos


def _get_cell_value(cell_type):
    """获取单元格价值"""
    if cell_type == ELEMENT_MAPPING['G']:
        return GOLD_VALUE
    elif cell_type == ELEMENT_MAPPING['T']:
        return TRAP_PENALTY
    elif cell_type in (ELEMENT_MAPPING['L'], ELEMENT_MAPPING['B']):
        return 0  # 机关和BOSS不直接改变分值
    else:
        return 0  # 路径/起点/终点


# 示例用法
if __name__ == "__main__":
    # 测试迷宫 (S:起点, E:终点, G:金币+5, T:陷阱-3)
    test_grid = [
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
        ['#', 'S', '#', 'G', '#', ' ', ' ', 'T', 'G', '#'],
        ['#', ' ', '#', ' ', '#', ' ', '#', '#', ' ', '#'],
        ['#', 'G', ' ', ' ', 'L', ' ', '#', ' ', ' ', '#'],
        ['#', '#', '#', '#', '#', '#', ' ', ' ', '#', '#'],
        ['#', 'B', 'E', 'B', ' ', ' ', '#', 'T', ' ', '#'],
        ['#', ' ', '#', '#', ' ', '#', '#', ' ', '#', '#'],
        ['#', 'T', ' ', 'G', ' ', ' ', ' ', 'L', ' ', '#'],
        ['#', ' ', '#', ' ', '#', ' ', '#', ' ', ' ', '#'],
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#']
    ]

    # 转换为数值网格
    numeric_grid = [[ELEMENT_MAPPING[cell] for cell in row] for row in test_grid]

    max_val, path = calculate_optimal_path(numeric_grid)
    print(f"最大资源值: {max_val}")
    print("最优路径坐标:")
    for step in path:
        print(f"({step[0]}, {step[1]}) -> {test_grid[step[0]][step[1]]}")
