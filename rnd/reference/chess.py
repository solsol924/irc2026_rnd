import math
import time
import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
from typing import List, Tuple, Optional

# =========================
# 설정값
# =========================

# ---- 이동/회전 정의 ----
DEGREES = 6.0  # 선형 스텝 종료 시 좌(+CCW) 회전량(기준값)
ROT_DEG = 13.0  # 회전 스텝 종료 시 헤딩 변화(좌 +13°, 우 -13°)

# 로컬 이동 벡터 (선형 4종 + 회전 2종 + 큰 직진 1종)
MOVE_VECTORS = [
    np.array([4.0, 0.5]),   # 0
    np.array([-4.0, 0.5]),  # 1
    np.array([0.0, 4.0]),   # 2
    np.array([0.0, -4.0]),  # 3
    np.array([-1.0, 1.0]),  # 4: 좌회전 스텝
    np.array([1.0, 1.0]),   # 5: 우회전 스텝
    np.array([-3.0, 12.0]), # 6: 크게 직진 (linear로 분류)
]

# 각도 변화(계획=무오차, 실행=±10% 오차)
# 선형 스텝은 항상 +DEGREES, 회전 스텝은 ±ROT_DEG
MOVE_DELTAS = [DEGREES, DEGREES, DEGREES, DEGREES, +ROT_DEG, -ROT_DEG, DEGREES]

# ---- 도장/목표/제약 ----
OFFSET = np.array([0.0, 3.0])  # 도장 오프셋(바디 정면)
TARGET = np.array([4.0, 34.0])
RADIUS = 1.0  # 성공(도장 중심이 이 반경 이내)
FORBIDDEN_RADIUS = 2.0  # 금지 구역(몸/경로 선분 모두 금지)

# ---- 탐색/실행 ----
CONSERVATIVE_FACTOR = 1.2  # A* 휴리스틱용 보수 반경 확대
ASTAR_MAX_DEPTH = 40       # A* 깊이 (스텝 수 기준)
GLOBAL_MAX_STEPS = 400     # 전체 안전장치
REPLAN_INTERVAL = 5        # 5스텝마다 재관측/재계획
NOISE_SCALE = 0.10         # 이동/각도 ±10% 오차

# ---- A* 안전장치/로그 ----
ASTAR_MAX_EXPANSIONS = 200_000
ASTAR_MAX_TIME_SEC = 3.0
DEBUG_PROGRESS = True
DEBUG_EVERY_N = 5_000

# ---- 에피소드 하트비트(선택) ----
EPISODE_MAX_TIME_SEC = 60.0
HB_INTERVAL_SEC = 1.0

# ---- 헤딩 화살표 표시 ----
HEAD_ARROW_LEN = 2.0
HEAD_ARROW_EVERY = 1  # n스텝마다 표시(1=매 스텝)

# ---- 큰 직진 정렬/가성비 게이트 + 액션 비용 ----
BIG_STRIDE_IDX = 6
ALIGN_THRESH_DEG = 10.0   # 이 각도 이내면 큰 직진 허용
BIG_STRIDE_H_GAIN = 4.0   # 큰 직진이 보수거리 h를 이만큼 이상 줄이면 허용
MOVE_COSTS = [1, 1, 1, 1, 3, 3, 2]  # [선형4, 좌, 우, 큰직진]


# =========================
# 유틸
# =========================
def rot_deg(deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def noisy_vector(vec: np.ndarray) -> np.ndarray:
    return np.array([
        vec[0] * random.uniform(1.0 - NOISE_SCALE, 1.0 + NOISE_SCALE),
        vec[1] * random.uniform(1.0 - NOISE_SCALE, 1.0 + NOISE_SCALE),
    ])


def noisy_deg(delta_deg: float) -> float:
    return delta_deg * random.uniform(1.0 - NOISE_SCALE, 1.0 + NOISE_SCALE)


def paint_center(pos: np.ndarray, theta_deg: float) -> np.ndarray:
    return pos + rot_deg(theta_deg).dot(OFFSET)


def conservative_distance(pcenter: np.ndarray) -> float:
    inflated = RADIUS * CONSERVATIVE_FACTOR
    return max(0.0, np.linalg.norm(pcenter - TARGET) - inflated)


def success(pcenter: np.ndarray) -> bool:
    return np.linalg.norm(pcenter - TARGET) <= RADIUS


def in_forbidden(pos: np.ndarray) -> bool:
    return np.linalg.norm(pos - TARGET) <= FORBIDDEN_RADIUS


def segment_circle_intersect(p0: np.ndarray, p1: np.ndarray,
                             center: np.ndarray, radius: float) -> bool:
    """선분 p0->p1 이 원(center, radius)와 교차/접촉하면 True"""
    v = p1 - p0
    vv = float(v.dot(v))
    if vv == 0.0:
        return np.linalg.norm(p0 - center) <= radius
    t = float((center - p0).dot(v)) / vv
    t = max(0.0, min(1.0, t))
    closest = p0 + t * v
    return np.linalg.norm(closest - center) <= radius


def segment_hits_forbidden(p0: np.ndarray, p1: np.ndarray) -> bool:
    return segment_circle_intersect(p0, p1, TARGET, FORBIDDEN_RADIUS)


def can_see_target(pos: np.ndarray, theta_deg: float) -> bool:
    """전방(FOV=바디 프레임 y>=0)만 가시"""
    rel = rot_deg(-theta_deg).dot(TARGET - pos)
    return rel[1] >= 0.0


def heading_error_deg(pos: np.ndarray, theta_deg: float) -> float:
    """현재 헤딩과 타깃 방향 벡터 사이의 최소각(절대값, 도)"""
    to_tgt = TARGET - pos
    if np.allclose(to_tgt, 0):
        return 0.0
    fwd = rot_deg(theta_deg).dot(np.array([0.0, 1.0]))
    num = float(np.dot(fwd, to_tgt))
    den = float(np.linalg.norm(fwd) * np.linalg.norm(to_tgt)) + 1e-12
    cosv = max(-1.0, min(1.0, num / den))
    return abs(math.degrees(math.acos(cosv)))


def h_value(pos: np.ndarray, theta_deg: float) -> float:
    """현재 상태의 보수거리 h"""
    return conservative_distance(paint_center(pos, theta_deg))


# =========================
# A*: 금지구역 회피 + 보수 휴리스틱 + 액션 비용
# =========================
def plan_astar(pos0: np.ndarray, theta0_deg: float,
               max_depth: int = ASTAR_MAX_DEPTH
               ) -> Optional[Tuple[List[int], List[np.ndarray], np.ndarray]]:

    start_time = time.time()
    expansions = 0
    counter = 0
    heap = []  # (f, tie, g_cost, depth, seq, pos, theta)

    heapq.heappush(heap, (0.0, counter, 0.0, 0, [], pos0.copy(), float(theta0_deg)))

    if DEBUG_PROGRESS:
        print(f"[A*] start from pos={pos0}, theta={theta0_deg:.2f}")

    while heap:
        if (time.time() - start_time) > ASTAR_MAX_TIME_SEC:
            if DEBUG_PROGRESS:
                print(f"[A*] timeout after {expansions} expansions")
            return None

        if expansions >= ASTAR_MAX_EXPANSIONS:
            if DEBUG_PROGRESS:
                print(f"[A*] expansion cap reached: {expansions}")
            return None

        f, _, g_cost, depth, seq, pos, theta = heapq.heappop(heap)
        if depth >= max_depth:
            continue

        base_h = h_value(pos, theta)

        for mv in range(len(MOVE_VECTORS)):

            # 큰 직진: 정렬이 충분하거나 h 이득이 충분할 때만 허용
            if mv == BIG_STRIDE_IDX and heading_error_deg(pos, theta) > ALIGN_THRESH_DEG:
                step_world_tmp = rot_deg(theta).dot(MOVE_VECTORS[mv])
                tmp_pos = pos + step_world_tmp
                if in_forbidden(tmp_pos) or segment_hits_forbidden(pos, tmp_pos):
                    continue
                tmp_theta = theta + MOVE_DELTAS[mv]
                new_h_tmp = conservative_distance(paint_center(tmp_pos, tmp_theta))
                if (base_h - new_h_tmp) < BIG_STRIDE_H_GAIN:
                    continue

            new_seq = seq + [mv]
            step_world = rot_deg(theta).dot(MOVE_VECTORS[mv])
            new_pos = pos + step_world

            if in_forbidden(new_pos) or segment_hits_forbidden(pos, new_pos):
                continue

            new_theta = theta + MOVE_DELTAS[mv]
            pc = paint_center(new_pos, new_theta)
            d = conservative_distance(pc)
            expansions += 1

            if DEBUG_PROGRESS and (expansions % DEBUG_EVERY_N == 0):
                print(f"[A*] expanded {expansions} nodes (depth={depth+1})")

            if d <= 0.0:
                # 경로 복원
                planned_positions = [pos0.copy()]
                cur_pos = pos0.copy()
                cur_theta = theta0_deg
                for mv2 in new_seq:
                    cur_pos = cur_pos + rot_deg(cur_theta).dot(MOVE_VECTORS[mv2])
                    planned_positions.append(cur_pos.copy())
                    cur_theta += MOVE_DELTAS[mv2]
                if DEBUG_PROGRESS:
                    print(f"[A*] goal reached with seq_len={len(new_seq)}, expansions={expansions}")
                return new_seq, planned_positions, pc

            new_g = g_cost + MOVE_COSTS[mv]
            new_f = new_g + d
            counter += 1
            heapq.heappush(heap, (new_f, counter, new_g, depth + 1, new_seq, new_pos, new_theta))

    if DEBUG_PROGRESS:
        print(f"[A*] failed with {expansions} expansions (heap exhausted)")
    return None


# =========================
# 실행(노이즈 포함): 헤딩 기록 버전
# =========================
def execute_prefix_with_noise_with_headings(
    seq: List[int], k: int, pos: np.ndarray, theta_deg: float
) -> Tuple[np.ndarray, float, int, List[np.ndarray], bool, List[float]]:
    """
    seq의 앞에서 최대 k스텝까지 노이즈 포함 실행.
    이동/각도 오차 포함.
    반환: (새 pos, 새 theta, 실행 스텝 수, 경로, 충돌여부, 헤딩로그)
    """
    executed = 0
    path = [pos.copy()]
    headings = [float(theta_deg)]
    cur_pos = pos.copy()
    cur_theta = float(theta_deg)
    collided = False

    for mv in seq[:k]:
        step_local = noisy_vector(MOVE_VECTORS[mv])
        step_world = rot_deg(cur_theta).dot(step_local)
        next_pos = cur_pos + step_world
        if in_forbidden(next_pos) or segment_hits_forbidden(cur_pos, next_pos):
            collided = True
            break
        cur_pos = next_pos
        path.append(cur_pos.copy())
        cur_theta += noisy_deg(MOVE_DELTAS[mv])
        headings.append(float(cur_theta))
        executed += 1

    return cur_pos, cur_theta, executed, path, collided, headings


# =========================
# "뒤라서 안 보이면" 한 스텝 휙 (탐색)
# =========================
def pick_peek_step(pos: np.ndarray, theta_deg: float) -> int:
    """다음 1스텝으로 타깃을 전방으로 가장 잘 가져오는 move 선택"""
    best_mv, best_score = None, -1e18
    base_h = h_value(pos, theta_deg)

    for mv in range(len(MOVE_VECTORS)):
        if mv == BIG_STRIDE_IDX and heading_error_deg(pos, theta_deg) > ALIGN_THRESH_DEG:
            step_world_tmp = rot_deg(theta_deg).dot(MOVE_VECTORS[mv])
            tmp_pos = pos + step_world_tmp
            if in_forbidden(tmp_pos) or segment_hits_forbidden(pos, tmp_pos):
                continue
            tmp_theta = theta_deg + MOVE_DELTAS[mv]
            new_h_tmp = conservative_distance(paint_center(tmp_pos, tmp_theta))
            if (base_h - new_h_tmp) < BIG_STRIDE_H_GAIN:
                continue

        pos1 = pos + rot_deg(theta_deg).dot(MOVE_VECTORS[mv])
        if in_forbidden(pos1) or segment_hits_forbidden(pos, pos1):
            continue
        theta1 = theta_deg + MOVE_DELTAS[mv]
        rel1 = rot_deg(-theta1).dot(TARGET - pos1)
        score = rel1[1]  # 전방성
        if score > best_score:
            best_score = score
            best_mv = mv

    return 0 if best_mv is None else best_mv


# =========================
# 메인 루프 (5스텝 재관측 + 헤딩 화살표 렌더)
# =========================
def run_episode(seed: Optional[int] = None, verbose: bool = True, plot: bool = True) -> bool:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pos = np.array([0.0, 0.0], dtype=float)
    theta = 0.0
    total_steps = 0
    full_path = [pos.copy()]
    full_headings = [float(theta)]
    plan: Optional[List[int]] = None
    big_stride_used = 0

    episode_start = time.time()
    last_hb = episode_start

    if DEBUG_PROGRESS:
        print(f"[EP] start. pos={pos}, theta={theta:.2f}")

    while total_steps < GLOBAL_MAX_STEPS:
        now = time.time()
        if (now - episode_start) > EPISODE_MAX_TIME_SEC:
            if verbose:
                print(f"[EP] timeout at steps={total_steps}")
            break

        if (now - last_hb) >= HB_INTERVAL_SEC:
            see = can_see_target(pos, theta)
            plan_len = (len(plan) if plan else 0)
            print(f"[HB] t={now - episode_start:.1f}s steps={total_steps} "
                  f"pos=({pos[0]:.2f},{pos[1]:.2f}) theta={theta:.2f} "
                  f"see={see} plan_len={plan_len} big6={big_stride_used}")
            last_hb = now

        sense_time = (total_steps % REPLAN_INTERVAL == 0) or (not plan)

        if sense_time:
            if can_see_target(pos, theta):
                res = plan_astar(pos, theta, ASTAR_MAX_DEPTH)
                if res is None:
                    if verbose:
                        print("[A*] unavailable → peek 1 step")
                    mv = pick_peek_step(pos, theta)
                    pos2, theta2, ex, path, collided, heads = execute_prefix_with_noise_with_headings(
                        [mv], 1, pos, theta)
                    if ex == 1 and mv == BIG_STRIDE_IDX:
                        big_stride_used += 1
                    full_path.extend(path[1:])
                    full_headings.extend(heads[1:])
                    total_steps += ex
                    pos, theta = pos2, theta2
                    if collided and verbose:
                        print("[WARN] peek-step collision")

                    pc_now = paint_center(pos, theta)
                    if success(pc_now):
                        if verbose:
                            print("[SUCCESS] steps:", total_steps,
                                  "pos:", pos, "theta:", theta,
                                  "big6:", big_stride_used)
                        if plot:
                            draw_result(full_path, full_headings, pc_now,
                                        title=f"SUCCESS (peek-step, big6={big_stride_used})")
                        return True
                    else:
                        plan = None
                else:
                    plan, _, _ = res
            else:
                mv = pick_peek_step(pos, theta)
                pos2, theta2, ex, path, collided, heads = execute_prefix_with_noise_with_headings(
                    [mv], 1, pos, theta)
                if ex == 1 and mv == BIG_STRIDE_IDX:
                    big_stride_used += 1
                full_path.extend(path[1:])
                full_headings.extend(heads[1:])
                total_steps += ex
                pos, theta = pos2, theta2
                if collided and verbose:
                    print("[WARN] peek-step collision")

                pc_now = paint_center(pos, theta)
                if success(pc_now):
                    if verbose:
                        print("[SUCCESS] steps:", total_steps,
                              "pos:", pos, "theta:", theta,
                              "big6:", big_stride_used)
                    if plot:
                        draw_result(full_path, full_headings, pc_now,
                                    title=f"SUCCESS (peek-step, big6={big_stride_used})")
                    return True
                continue

        if plan:
            k = min(REPLAN_INTERVAL, len(plan))
            pos2, theta2, ex, path, collided, heads = execute_prefix_with_noise_with_headings(
                plan, k, pos, theta)
            for mv in plan[:ex]:
                if mv == BIG_STRIDE_IDX:
                    big_stride_used += 1
            full_path.extend(path[1:])
            full_headings.extend(heads[1:])
            total_steps += ex
            pos, theta = pos2, theta2
            plan = plan[ex:]

            if collided:
                if verbose:
                    print("[WARN] collision during execution → replan")
                plan = None

            pc_now = paint_center(pos, theta)
            if success(pc_now):
                if verbose:
                    print("[SUCCESS] steps:", total_steps,
                          "pos:", pos, "theta:", theta,
                          "big6:", big_stride_used)
                if plot:
                    draw_result(full_path, full_headings, pc_now,
                                title=f"SUCCESS (big6={big_stride_used})")
                return True
            else:
                plan = None
                continue

    if verbose:
        pc = paint_center(pos, theta)
        print("[FAIL] steps:", total_steps,
              "| last paint center dist:",
              np.linalg.norm(pc - TARGET),
              "big6:", big_stride_used)
    if plot:
        draw_result(full_path, full_headings,
                    paint_center(pos, theta),
                    title=f"FAIL (big6={big_stride_used})")
    return False


# =========================
# 시각화: 경로 + 헤딩 화살표
# =========================
def draw_result(path_xy: List[np.ndarray], headings: List[float],
                pc: np.ndarray, title: str = ""):
    xs = [p[0] for p in path_xy]
    ys = [p[1] for p in path_xy]

    plt.figure(figsize=(7, 7))
    plt.plot(xs, ys, marker='o', label="executed route")

    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.annotate(str(i), (x, y),
                     textcoords="offset points",
                     xytext=(4, 4), fontsize=8)

    # 헤딩 화살표
    sel_idx = list(range(0, len(path_xy),
                         max(1, HEAD_ARROW_EVERY)))
    qx, qy, qu, qv = [], [], [], []
    for i in sel_idx:
        th = headings[i] if i < len(headings) else headings[-1]
        fwd = rot_deg(th).dot(np.array([0.0, HEAD_ARROW_LEN]))
        qx.append(xs[i])
        qy.append(ys[i])
        qu.append(fwd[0])
        qv.append(fwd[1])

    if qx:
        plt.quiver(qx, qy, qu, qv,
                   angles='xy', scale_units='xy', scale=1,
                   label="heading")

    # 도장 원
    circle = plt.Circle((pc[0], pc[1]), RADIUS,
                        fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    plt.plot([pc[0]], [pc[1]], 'x',
             label="paint center (actual)")

    # 타깃 및 금지구역
    plt.plot([TARGET[0]], [TARGET[1]], 's',
             label=f"target {TARGET}")
    forb = plt.Circle((TARGET[0], TARGET[1]), FORBIDDEN_RADIUS,
                      fill=False, linestyle=':',
                      label="forbidden zone (body/segment)")
    plt.gca().add_patch(forb)

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================
# (선택) 몬테카를로 시뮬레이션
# =========================
def monte_carlo(trials: int = 20, plot_one: bool = False):
    succ = 0
    for t in range(trials):
        ok = run_episode(seed=None,
                         verbose=(t == 0),
                         plot=(plot_one and t == 0))
        if ok:
            succ += 1
    print(f"[Monte Carlo] {succ}/{trials} 성공")


# =========================
# 실행
# =========================
if __name__ == "__main__":
    print(f"[BOOT] chess2 starting… TARGET={TARGET}, OFFSET={OFFSET}")
    run_episode(seed=None, verbose=True, plot=True)
    # monte_carlo(trials=50, plot_one=False)
