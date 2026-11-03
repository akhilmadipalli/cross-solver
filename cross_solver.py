#!/usr/bin/env python3
"""
cross_solver.py

Finds the shortest (optimal, fewest moves in HTM) white cross for a 3x3 scramble.

Assumptions
-----------
* The scramble is applied with WHITE on top and GREEN in front
  (standard WCA orientation: red right, orange left, blue back, yellow bottom).
* The cross is then solved with WHITE on the bottom. Every front color
  (green, red, blue, orange) is tried, and the best one is reported.

Usage
-----
    python cross_solver.py "R U R2 F U2 R2 F2 D2 F R2 D2 F' U2 D' B2 U L R' B' F' R"
    python cross_solver.py            (then paste the scramble when asked)

How it works
------------
Only the 4 white edges matter for the cross. Each edge can sit in one of
12 slots with 2 flips, so the whole cross puzzle has
12 * 11 * 10 * 9 * 2^4 = 190,080 states. The program does one breadth-first
search from the solved cross to label every state with its exact distance,
then walks "downhill" from the scrambled state. That guarantees an optimal
solution for every orientation.
"""

import sys
from collections import deque

# ---------------------------------------------------------------------------
# Geometry
# Axes: 0 = x (R face is +x), 1 = y (U face is +y), 2 = z (F face is +z)
# ---------------------------------------------------------------------------
FACES = {
    "U": (1, 1), "D": (1, -1),
    "R": (0, 1), "L": (0, -1),
    "F": (2, 1), "B": (2, -1),
}
ROTATION_AXIS = {"x": 0, "y": 1, "z": 2}  # x follows R, y follows U, z follows F
SUFFIX_TURNS = {"": 1, "2": 2, "'": -1}
MOVE_NAMES = [f + s for f in "UDRLFB" for s in ("", "2", "'")]

FACE_OF_NORMAL = {
    (0, 1, 0): "U", (0, -1, 0): "D",
    (1, 0, 0): "R", (-1, 0, 0): "L",
    (0, 0, 1): "F", (0, 0, -1): "B",
}


def rot90(v, axis, turns):
    """Rotate vector v by turns * 90 degrees counterclockwise (right-hand rule) about axis."""
    x, y, z = v
    for _ in range(turns % 4):
        if axis == 0:
            x, y, z = x, -z, y
        elif axis == 1:
            x, y, z = z, y, -x
        else:
            x, y, z = -y, x, z
    return (x, y, z)


def parse(token):
    """Split 'R2' into ('R', 2), "U'" into ('U', -1)."""
    base, suffix = token[0], token[1:]
    if suffix not in SUFFIX_TURNS or (base not in FACES and base not in ROTATION_AXIS):
        raise ValueError(f"Unrecognized move: {token!r}")
    return base, SUFFIX_TURNS[suffix]


def turn_edge(edge, base, q):
    """Apply a face move or a whole-cube rotation to one edge (pos, white_normal, other_color)."""
    pos, normal, color = edge
    if base in ROTATION_AXIS:
        axis, sign = ROTATION_AXIS[base], 1
    else:
        axis, sign = FACES[base]
        if pos[axis] != sign:  # edge is not in the turning layer
            return edge
    turns = -q * sign  # clockwise when viewed from that face = negative rotation
    return (rot90(pos, axis, turns), rot90(normal, axis, turns), color)


# ---------------------------------------------------------------------------
# Cube model (only what the cross needs: 4 white edges + 6 centers)
# ---------------------------------------------------------------------------
class CrossCube:
    def __init__(self):
        # White top, green front
        self.centers = {
            (0, 1, 0): "white", (0, -1, 0): "yellow",
            (0, 0, 1): "green", (0, 0, -1): "blue",
            (1, 0, 0): "red", (-1, 0, 0): "orange",
        }
        up = (0, 1, 0)
        self.edges = [
            ((0, 1, 1), up, "green"),
            ((1, 1, 0), up, "red"),
            ((0, 1, -1), up, "blue"),
            ((-1, 1, 0), up, "orange"),
        ]

    def copy(self):
        c = CrossCube.__new__(CrossCube)
        c.centers = dict(self.centers)
        c.edges = list(self.edges)
        return c

    def apply(self, alg):
        for token in alg.split():
            base, q = parse(token)
            self.edges = [turn_edge(e, base, q) for e in self.edges]
            if base in ROTATION_AXIS:
                axis = ROTATION_AXIS[base]
                self.centers = {rot90(n, axis, -q): col for n, col in self.centers.items()}
        return self

    def color_on(self, face):
        for n, col in self.centers.items():
            if FACE_OF_NORMAL[n] == face:
                return col


# ---------------------------------------------------------------------------
# Search tables
# ---------------------------------------------------------------------------
def all_edge_states():
    """Every (position, white-sticker normal) an edge can have: 12 slots x 2 flips = 24."""
    states = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                pos = (x, y, z)
                if sum(c != 0 for c in pos) != 2:
                    continue
                for axis in range(3):
                    if pos[axis] != 0:
                        normal = [0, 0, 0]
                        normal[axis] = pos[axis]
                        states.append((pos, tuple(normal)))
    return states


EDGE_STATES = all_edge_states()
EDGE_INDEX = {s: i for i, s in enumerate(EDGE_STATES)}
N = len(EDGE_STATES)  # 24

# MOVE_TABLE[m][i] = edge-state index after move m
MOVE_TABLE = []
for name in MOVE_NAMES:
    base, q = parse(name)
    MOVE_TABLE.append([
        EDGE_INDEX[turn_edge((p, n, None), base, q)[:2]] for (p, n) in EDGE_STATES
    ])

DOWN = (0, -1, 0)
TARGET_SLOTS = [(0, -1, 1), (1, -1, 0), (0, -1, -1), (-1, -1, 0)]  # DF, DR, DB, DL
SOLVED = tuple(EDGE_INDEX[(p, DOWN)] for p in TARGET_SLOTS)


def encode(s):
    return ((s[0] * N + s[1]) * N + s[2]) * N + s[3]


def decode(code):
    code, d = divmod(code, N)
    code, c = divmod(code, N)
    a, b = divmod(code, N)
    return (a, b, c, d)


def build_distance_table():
    dist = bytearray([255]) * (N ** 4)
    start = encode(SOLVED)
    dist[start] = 0
    queue = deque([start])
    while queue:
        code = queue.popleft()
        a, b, c, d = decode(code)
        nd = dist[code] + 1
        for t in MOVE_TABLE:
            nxt = ((t[a] * N + t[b]) * N + t[c]) * N + t[d]
            if dist[nxt] == 255:
                dist[nxt] = nd
                queue.append(nxt)
    return dist


def cross_state(cube):
    """Order the 4 white edges by the bottom slot they belong in (DF, DR, DB, DL)."""
    state = [None] * 4
    for pos, normal, color in cube.edges:
        center = next(n for n, col in cube.centers.items() if col == color)
        target = (center[0], -1, center[2])
        state[TARGET_SLOTS.index(target)] = EDGE_INDEX[(pos, normal)]
    return tuple(state)


def solve(state, dist):
    """Walk downhill through the distance table; every step is optimal."""
    solution = []
    code = encode(state)
    while dist[code] != 0:
        a, b, c, d = decode(code)
        for name, t in zip(MOVE_NAMES, MOVE_TABLE):
            nxt = ((t[a] * N + t[b]) * N + t[c]) * N + t[d]
            if dist[nxt] == dist[code] - 1:
                solution.append(name)
                code = nxt
                break
    return solution


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ORIENTATIONS = ["x2", "x2 y", "x2 y2", "x2 y'"]  # white to the bottom, then each front color


def main():
    scramble = " ".join(sys.argv[1:]).strip() or input("Enter scramble: ").strip()
    scramble = scramble.replace("’", "'")  # allow curly apostrophes from copy/paste
    if any(tok[0] in ROTATION_AXIS for tok in scramble.split()):
        sys.exit("Scrambles should only contain face turns (U D R L F B), not rotations.")

    print("Building cross distance table (one time, a few seconds)...", file=sys.stderr)
    dist = build_distance_table()

    scrambled = CrossCube().apply(scramble)

    results = []
    for rotation in ORIENTATIONS:
        cube = scrambled.copy().apply(rotation)
        moves = solve(cross_state(cube), dist)
        # Double check: applying the solution must give a solved cross
        assert cross_state(cube.copy().apply(" ".join(moves))) == SOLVED
        results.append((len(moves), rotation, cube, moves))

    length = results[0][0]
    print()
    print(f"Scramble : {scramble}")
    print("           (applied with white on top, green in front)")
    print(f"Optimal white cross: {length} move{'s' if length != 1 else ''}")
    print()
    print("Turning the whole cube does not change the cross length, so every")
    print("front color below is equally short. Pick the one you like best.")
    print()

    for rank, (_, rotation, cube, moves) in enumerate(results, 1):
        front, right = cube.color_on("F"), cube.color_on("R")
        print(f"{rank}. {front.upper()} in front")
        print(f"   Starting position : white on bottom, yellow on top, {front} in front, {right} on the right")
        print(f"   How to get there  : from the scramble orientation, rotate {rotation}")
        print(f"   Cross solution    : {' '.join(moves) if moves else '(already solved)'}")
        print()


if __name__ == "__main__":
    main()
