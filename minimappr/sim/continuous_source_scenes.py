"""Continuous-source tracking scenes (localization overhaul E6).

A scene is the stream of solves the sidecar publishes for a few sources over
a short span, plus the order in which Python receives them. Positions are
truth plus seeded Gaussian fix error; the solve-to-track path is what is
under test, so no audio is synthesised.

Delivery follows the solve contract (``docs/localization_solve_contract.md``
section 1):

- every event-triggered solve is published at once as a ``localization_result``
  with its own ``solve_id``;
- a classifier render covers the latest event solve on its buffer and repeats
  that ``solve_id``; it is published with a ``localization_result`` carrying the
  same solve, and both arrive ``render_latency_seconds`` after the solve, so
  later positions of the same source are already in when the label arrives.

Cadence (all from Settings defaults, passed in by the caller as
``SceneTiming``): a continuous source re-arms the live trigger on every
dispatch, so it produces one event solve per trigger cooldown, landing on the
next firmware upload boundary. A bird produces one solve per call. Renders
follow the sidecar's classifier-render minimum interval; the first covers the
source's first solve.

Builders take ``reporter_ids``: the two node ids whose packets report the
cluster's solves (reporter and cluster are separate in the contract).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Vec3 = tuple[float, float, float]

SOURCE_KIND_DRONE = "drone"
SOURCE_KIND_BIRD = "bird"
DRONE_LABEL = "drone"
ROBIN_LABEL = "american robin"
WREN_LABEL = "house wren"

DELIVERY_EVENT_LOCALIZATION_RESULT = "event_localization_result"
DELIVERY_RENDER_LOCALIZATION_RESULT = "render_localization_result"
DELIVERY_COVERING_CLASSIFIER_RENDER = "covering_classifier_render"

# Firmware uploads every 100 ms; the trigger is evaluated per upload, so a
# re-armed trigger fires up to one upload period after its cooldown ends.
FIRMWARE_UPLOAD_PERIOD_SECONDS = 0.1
# Sidecar compute + SSE hop for an event solve; small next to the cadence.
EVENT_PUBLISH_LATENCY_SECONDS = 0.05
# Render span close + classifier inference before the render is published.
DEFAULT_RENDER_LATENCY_SECONDS = 2.0

# Typical small multirotor cruise; 6.4 m between solves at the default cadence.
DRONE_CRUISE_SPEED_MPS = 8.0
DRONE_FIX_ERROR_STD_M = 1.0
BIRD_FIX_ERROR_STD_M = 1.0
# Hover holds station within a few decimetres (GPS-hold quadcopter).
HOVER_WANDER_AMPLITUDE_M = 0.4
HOVER_WANDER_PERIOD_SECONDS = 7.0
BIRD_CALL_GAP_RANGE_SECONDS = (3.0, 6.0)

SHORT_SCENE_SECONDS = 20.0
SHORT_HOVER_SECONDS = 16.0
LONG_SCENE_SECONDS = 120.0

# Site layout of tests/localization_solve_manifest_fixtures.py: planar array at
# (30, 25, 1.5) and a point node at (33, 21, 4). Flight lines pass 40-55 m off.
FLY_BY_LINE_Y_M = 75.0
FLY_BY_ALTITUDE_M = 30.0
FLY_BY_CLOSEST_APPROACH_X_M = 30.0
HOVER_POSITION_M: Vec3 = (55.0, 70.0, 25.0)
CROSSING_POINT_XY_M = (30.0, 70.0)
# 15 m of altitude separation: crossing in plan view, separable in 3-D.
CROSSING_ALTITUDES_M = (30.0, 45.0)
ROBIN_PERCH_M: Vec3 = (20.0, 40.0, 6.0)
WREN_PERCH_M: Vec3 = (60.0, 35.0, 4.0)
# Second robin 30 m from the first: same label, same cluster, distinct birds.
SECOND_ROBIN_PERCH_M: Vec3 = (50.0, 45.0, 5.0)
# Counter-singing robins: one call a second each, answering 0.3 s apart.
COUNTER_SINGING_CALL_GAP_SECONDS = 1.0
COUNTER_SINGING_ANSWER_DELAY_SECONDS = 0.3
COUNTER_SINGING_CALLS_PER_BIRD = 8

SAME_LABEL_CALLERS_SEED = 65
FLY_BY_SEED = 61
HOVER_SEED = 62
CROSSING_SEED = 63
DRONE_WITH_BIRDS_SEED = 64


@dataclass(frozen=True, slots=True)
class SceneTiming:
    solve_interval_seconds: float
    render_interval_seconds: float
    render_latency_seconds: float = DEFAULT_RENDER_LATENCY_SECONDS
    upload_period_seconds: float = FIRMWARE_UPLOAD_PERIOD_SECONDS

    @property
    def largest_solve_interval_seconds(self) -> float:
        return self.solve_interval_seconds + self.upload_period_seconds


@dataclass(frozen=True, slots=True)
class SceneSource:
    source_id: str
    label: str
    kind: str
    reporting_node_id: str


@dataclass(frozen=True, slots=True)
class SceneSolve:
    solve_id: str
    source_id: str
    # Window centre, seconds after scene start.
    solve_offset_seconds: float
    true_position_m: Vec3
    true_velocity_mps: Vec3
    measured_position_m: Vec3
    position_std_m: float


@dataclass(frozen=True, slots=True)
class SceneDelivery:
    delivery_offset_seconds: float
    solve_id: str
    kind: str


@dataclass(frozen=True, slots=True)
class ContinuousSourceScene:
    name: str
    sources: dict[str, SceneSource]
    solves: tuple[SceneSolve, ...]
    deliveries: tuple[SceneDelivery, ...]
    timing: SceneTiming

    def solve(self, solve_id: str) -> SceneSolve:
        return next(solve for solve in self.solves if solve.solve_id == solve_id)

    def solves_for(self, source_id: str) -> list[SceneSolve]:
        return [solve for solve in self.solves if solve.source_id == source_id]

    def source_ids_of_kind(self, kind: str) -> list[str]:
        return [source_id for source_id, source in self.sources.items() if source.kind == kind]


TrajectoryFunction = Callable[[float], tuple[np.ndarray, np.ndarray]]


def straight_line_trajectory(start_m: Vec3, velocity_mps: Vec3) -> TrajectoryFunction:
    start = np.asarray(start_m, dtype=np.float64)
    velocity = np.asarray(velocity_mps, dtype=np.float64)
    return lambda offset_seconds: (start + velocity * offset_seconds, velocity.copy())


def hover_trajectory(station_m: Vec3) -> TrajectoryFunction:
    station = np.asarray(station_m, dtype=np.float64)
    angular_rate = 2.0 * np.pi / HOVER_WANDER_PERIOD_SECONDS

    def position_and_velocity(offset_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        phase = angular_rate * offset_seconds
        wander = HOVER_WANDER_AMPLITUDE_M * np.array([np.sin(phase), np.cos(phase), 0.0])
        wander_velocity = HOVER_WANDER_AMPLITUDE_M * angular_rate * np.array([np.cos(phase), -np.sin(phase), 0.0])
        return station + wander, wander_velocity

    return position_and_velocity


def continuous_source_solve_offsets(
    rng: np.random.Generator, timing: SceneTiming, *, start_seconds: float, end_seconds: float
) -> list[float]:
    """One solve per cooldown, each on the first upload boundary after it."""
    offsets: list[float] = []
    offset = start_seconds
    while offset <= end_seconds:
        offsets.append(offset)
        boundary_delay_seconds = timing.upload_period_seconds * int(rng.integers(0, 2))
        offset += timing.solve_interval_seconds + boundary_delay_seconds
    return offsets


def bird_call_offsets(rng: np.random.Generator, *, start_seconds: float, end_seconds: float) -> list[float]:
    offsets: list[float] = []
    offset = start_seconds
    while offset <= end_seconds:
        offsets.append(offset)
        offset += float(rng.uniform(*BIRD_CALL_GAP_RANGE_SECONDS))
    return offsets


def _vec3(values) -> Vec3:
    return (float(values[0]), float(values[1]), float(values[2]))


def solves_along_trajectory(
    rng: np.random.Generator,
    *,
    source_id: str,
    offsets_seconds: list[float],
    trajectory: TrajectoryFunction,
    position_std_m: float,
) -> list[SceneSolve]:
    solves = []
    for index, offset_seconds in enumerate(offsets_seconds):
        position, velocity = trajectory(offset_seconds)
        measured = position + rng.normal(0.0, position_std_m, size=3)
        solves.append(
            SceneSolve(
                solve_id=f"{source_id}-solve-{index:03d}",
                source_id=source_id,
                solve_offset_seconds=float(offset_seconds),
                true_position_m=_vec3(position),
                true_velocity_mps=_vec3(velocity),
                measured_position_m=_vec3(measured),
                position_std_m=position_std_m,
            )
        )
    return solves


def covered_solves(solves: list[SceneSolve], render_interval_seconds: float) -> list[SceneSolve]:
    """Solves a source's classifier renders cover: its first, then one per render interval."""
    covered: list[SceneSolve] = []
    for solve in solves:
        if not covered or solve.solve_offset_seconds - covered[-1].solve_offset_seconds >= render_interval_seconds:
            covered.append(solve)
    return covered


def deliveries_for(solves: list[SceneSolve], timing: SceneTiming) -> list[SceneDelivery]:
    """Event delivery for every solve plus the render pair for each covered solve."""
    deliveries = [
        SceneDelivery(solve.solve_offset_seconds + EVENT_PUBLISH_LATENCY_SECONDS, solve.solve_id,
                      DELIVERY_EVENT_LOCALIZATION_RESULT)
        for solve in solves
    ]
    for solve in covered_solves(solves, timing.render_interval_seconds):
        render_offset_seconds = solve.solve_offset_seconds + timing.render_latency_seconds
        deliveries.append(SceneDelivery(render_offset_seconds, solve.solve_id, DELIVERY_RENDER_LOCALIZATION_RESULT))
        deliveries.append(SceneDelivery(render_offset_seconds, solve.solve_id, DELIVERY_COVERING_CLASSIFIER_RENDER))
    return deliveries


def assemble_scene(
    name: str, sources: list[SceneSource], solves_by_source: dict[str, list[SceneSolve]], timing: SceneTiming
) -> ContinuousSourceScene:
    all_solves = sorted(
        (solve for solves in solves_by_source.values() for solve in solves),
        key=lambda solve: solve.solve_offset_seconds,
    )
    all_deliveries = [delivery for solves in solves_by_source.values() for delivery in deliveries_for(solves, timing)]
    # Stable sort: a render's localization_result stays ahead of its render.
    all_deliveries.sort(key=lambda delivery: delivery.delivery_offset_seconds)
    return ContinuousSourceScene(
        name=name,
        sources={source.source_id: source for source in sources},
        solves=tuple(all_solves),
        deliveries=tuple(all_deliveries),
        timing=timing,
    )


def _fly_by_trajectory(duration_seconds: float, *, line_y_m: float, altitude_m: float) -> TrajectoryFunction:
    half_track_m = 0.5 * DRONE_CRUISE_SPEED_MPS * duration_seconds
    start = (FLY_BY_CLOSEST_APPROACH_X_M - half_track_m, line_y_m, altitude_m)
    return straight_line_trajectory(start, (DRONE_CRUISE_SPEED_MPS, 0.0, 0.0))


def drone_fly_by(
    timing: SceneTiming, *, duration_seconds: float = SHORT_SCENE_SECONDS, reporter_ids: tuple[str, str]
) -> ContinuousSourceScene:
    rng = np.random.default_rng(FLY_BY_SEED)
    drone = SceneSource("drone-1", DRONE_LABEL, SOURCE_KIND_DRONE, reporter_ids[0])
    solves = solves_along_trajectory(
        rng,
        source_id=drone.source_id,
        offsets_seconds=continuous_source_solve_offsets(rng, timing, start_seconds=0.0, end_seconds=duration_seconds),
        trajectory=_fly_by_trajectory(duration_seconds, line_y_m=FLY_BY_LINE_Y_M, altitude_m=FLY_BY_ALTITUDE_M),
        position_std_m=DRONE_FIX_ERROR_STD_M,
    )
    return assemble_scene("drone_fly_by", [drone], {drone.source_id: solves}, timing)


def hovering_drone(
    timing: SceneTiming, *, duration_seconds: float = SHORT_HOVER_SECONDS, reporter_ids: tuple[str, str]
) -> ContinuousSourceScene:
    rng = np.random.default_rng(HOVER_SEED)
    drone = SceneSource("drone-1", DRONE_LABEL, SOURCE_KIND_DRONE, reporter_ids[1])
    solves = solves_along_trajectory(
        rng,
        source_id=drone.source_id,
        offsets_seconds=continuous_source_solve_offsets(rng, timing, start_seconds=0.0, end_seconds=duration_seconds),
        trajectory=hover_trajectory(HOVER_POSITION_M),
        position_std_m=DRONE_FIX_ERROR_STD_M,
    )
    return assemble_scene("hovering_drone", [drone], {drone.source_id: solves}, timing)


def crossing_drones(
    timing: SceneTiming, *, duration_seconds: float = SHORT_SCENE_SECONDS, reporter_ids: tuple[str, str]
) -> ContinuousSourceScene:
    """Perpendicular tracks crossing in plan view at mid-scene; the cluster
    solves each drone in turn, so their solves interleave half a cadence apart."""
    rng = np.random.default_rng(CROSSING_SEED)
    eastbound = SceneSource("drone-east", DRONE_LABEL, SOURCE_KIND_DRONE, reporter_ids[0])
    northbound = SceneSource("drone-north", DRONE_LABEL, SOURCE_KIND_DRONE, reporter_ids[1])
    half_track_m = 0.5 * DRONE_CRUISE_SPEED_MPS * duration_seconds
    crossing_x_m, crossing_y_m = CROSSING_POINT_XY_M
    trajectories = {
        eastbound.source_id: straight_line_trajectory(
            (crossing_x_m - half_track_m, crossing_y_m, CROSSING_ALTITUDES_M[0]), (DRONE_CRUISE_SPEED_MPS, 0.0, 0.0)
        ),
        northbound.source_id: straight_line_trajectory(
            (crossing_x_m, crossing_y_m - half_track_m, CROSSING_ALTITUDES_M[1]), (0.0, DRONE_CRUISE_SPEED_MPS, 0.0)
        ),
    }
    first_solve_offsets = {eastbound.source_id: 0.0, northbound.source_id: 0.5 * timing.solve_interval_seconds}
    solves_by_source = {
        source_id: solves_along_trajectory(
            rng,
            source_id=source_id,
            offsets_seconds=continuous_source_solve_offsets(
                rng, timing, start_seconds=first_solve_offsets[source_id], end_seconds=duration_seconds
            ),
            trajectory=trajectory,
            position_std_m=DRONE_FIX_ERROR_STD_M,
        )
        for source_id, trajectory in trajectories.items()
    }
    return assemble_scene("crossing_drones", [eastbound, northbound], solves_by_source, timing)


def drone_with_birds_calling(
    timing: SceneTiming, *, duration_seconds: float = SHORT_SCENE_SECONDS, reporter_ids: tuple[str, str]
) -> ContinuousSourceScene:
    """The fly-by with a robin and a wren calling 35-40 m off the flight line."""
    rng = np.random.default_rng(DRONE_WITH_BIRDS_SEED)
    drone = SceneSource("drone-1", DRONE_LABEL, SOURCE_KIND_DRONE, reporter_ids[0])
    robin = SceneSource("robin-1", ROBIN_LABEL, SOURCE_KIND_BIRD, reporter_ids[1])
    wren = SceneSource("wren-1", WREN_LABEL, SOURCE_KIND_BIRD, reporter_ids[0])
    solves_by_source = {
        drone.source_id: solves_along_trajectory(
            rng,
            source_id=drone.source_id,
            offsets_seconds=continuous_source_solve_offsets(rng, timing, start_seconds=0.0, end_seconds=duration_seconds),
            trajectory=_fly_by_trajectory(duration_seconds, line_y_m=FLY_BY_LINE_Y_M, altitude_m=FLY_BY_ALTITUDE_M),
            position_std_m=DRONE_FIX_ERROR_STD_M,
        )
    }
    for bird, perch_m, first_call_seconds in ((robin, ROBIN_PERCH_M, 0.3), (wren, WREN_PERCH_M, 1.1)):
        solves_by_source[bird.source_id] = solves_along_trajectory(
            rng,
            source_id=bird.source_id,
            offsets_seconds=bird_call_offsets(rng, start_seconds=first_call_seconds, end_seconds=duration_seconds),
            trajectory=straight_line_trajectory(perch_m, (0.0, 0.0, 0.0)),
            position_std_m=BIRD_FIX_ERROR_STD_M,
        )
    return assemble_scene("drone_with_birds_calling", [drone, robin, wren], solves_by_source, timing)


def two_same_label_callers_in_one_cluster(
    timing: SceneTiming, *, call_count: int = COUNTER_SINGING_CALLS_PER_BIRD, reporter_ids: tuple[str, str]
) -> ContinuousSourceScene:
    """Two robins counter-singing 30 m apart, both solved by one cluster and
    reported by one node, so only position tells them apart."""
    rng = np.random.default_rng(SAME_LABEL_CALLERS_SEED)
    first = SceneSource("robin-a", ROBIN_LABEL, SOURCE_KIND_BIRD, reporter_ids[0])
    second = SceneSource("robin-b", ROBIN_LABEL, SOURCE_KIND_BIRD, reporter_ids[0])
    solves_by_source = {}
    for bird, perch_m, first_call_seconds in (
        (first, ROBIN_PERCH_M, 0.0),
        (second, SECOND_ROBIN_PERCH_M, COUNTER_SINGING_ANSWER_DELAY_SECONDS),
    ):
        offsets = [first_call_seconds + index * COUNTER_SINGING_CALL_GAP_SECONDS for index in range(call_count)]
        solves_by_source[bird.source_id] = solves_along_trajectory(
            rng,
            source_id=bird.source_id,
            offsets_seconds=offsets,
            trajectory=straight_line_trajectory(perch_m, (0.0, 0.0, 0.0)),
            position_std_m=BIRD_FIX_ERROR_STD_M,
        )
    return assemble_scene("two_same_label_callers_in_one_cluster", [first, second], solves_by_source, timing)


SCENE_BUILDERS = {
    "drone_fly_by": drone_fly_by,
    "hovering_drone": hovering_drone,
    "crossing_drones": crossing_drones,
    "drone_with_birds_calling": drone_with_birds_calling,
}
